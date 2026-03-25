"""
LanceDBStore — LanceDB-backed vector store as an alternative to FAISS.

The *filters* parameter in similarity_search is designed for future
file-level access control: store ``allowed_roles`` in each chunk's metadata
and pass ``{"allowed_roles": "admin"}`` (or similar) at query time to restrict
results to chunks the requesting user is permitted to see.
"""
from typing import Any

import numpy as np


class LanceDBStore:
    """Vector store backed by LanceDB."""

    def __init__(self, db_path: str, table_name: str) -> None:
        """
        Args:
            db_path: Path to the LanceDB database directory.
            table_name: Name of the LanceDB table to use (created on first write).
        """
        self.db_path = db_path
        self.table_name = table_name
        self._db = None
        self._table = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_db(self):
        """Open (or reuse) the LanceDB connection."""
        if self._db is None:
            import lancedb  # imported lazily so the rest of the codebase
                            # works even if lancedb is not installed
            self._db = lancedb.connect(self.db_path)
        return self._db

    def _open_table(self):
        """Return the table handle, opening it if it is not already open."""
        if self._table is None:
            db = self._open_db()
            if self.table_name in db.table_names():
                self._table = db.open_table(self.table_name)
        return self._table

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Embed and persist a batch of text chunks.

        Args:
            chunks: Raw text content for each chunk.
            embeddings: Pre-computed embedding vectors, one per chunk.
            metadatas: Arbitrary metadata dicts (e.g. source, page, allowed_roles).
        """
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise ValueError("chunks, embeddings, and metadatas must have the same length.")

        rows = [
            {"text": text, "vector": emb, **meta}
            for text, emb, meta in zip(chunks, embeddings, metadatas)
        ]

        db = self._open_db()
        if self.table_name in db.table_names():
            self._table = db.open_table(self.table_name)
            self._table.add(rows)
        else:
            self._table = db.create_table(self.table_name, data=rows)

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict = None,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks for *query_embedding*.

        Args:
            query_embedding: Embedding vector of the query.
            k: Number of results to return.
            filters: Optional metadata filters applied before ranking.
                     Example: ``{"allowed_roles": "analyst"}`` restricts
                     results to chunks whose ``allowed_roles`` field matches.
                     Supports equality checks on scalar metadata fields.

        Returns:
            List of dicts, each containing at least ``text``, ``score``,
            and any metadata fields stored with the chunk.
        """
        table = self._open_table()
        if table is None:
            raise RuntimeError("No table found. Call add_documents() first.")

        query = table.search(query_embedding).metric("cosine").limit(k)

        if filters:
            filter_clauses = " AND ".join(
                f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}"
                for key, value in filters.items()
            )
            query = query.where(filter_clauses)

        results = query.to_list()

        # Rename LanceDB's internal distance field to a normalised score
        output = []
        for row in results:
            row = dict(row)
            # cosine metric: distance in [0, 1], so sim = 1 - distance
            distance = row.pop("_distance", None)
            row["score"] = 1.0 - distance if distance is not None else None
            output.append(row)

        return output

    def save(self) -> None:
        """
        No-op for LanceDB — data is persisted automatically on every write.
        Included for API parity with FAISSStore.
        """

    def load(self) -> None:
        """
        Open an existing LanceDB table.  Call this before similarity_search
        when the process restarts and add_documents has not been called.
        """
        table = self._open_table()
        if table is None:
            raise RuntimeError(
                f"Table '{self.table_name}' not found in '{self.db_path}'. "
                "Run add_documents() first."
            )
