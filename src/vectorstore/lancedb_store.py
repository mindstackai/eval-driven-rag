"""
LanceDBStore — LanceDB-backed vector store as an alternative to FAISS.

The *filters* parameter in similarity_search is designed for future
file-level access control: store ``allowed_roles`` in each chunk's metadata
and pass ``{"allowed_roles": "admin"}`` (or similar) at query time to restrict
results to chunks the requesting user is permitted to see.
"""
import json
from typing import Any

import numpy as np

# Fixed schema columns written as top-level LanceDB fields.
# Any metadata key NOT in this set is serialised into `extra_metadata` (JSON
# string) so that varying PDF metadata never breaks the schema on append.
_CORE_COLUMNS = {"source", "page", "allowed_roles", "chunk_id", "ingest_source", "ingest_date"}


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

        def _normalize(meta: dict) -> dict:
            """Return a row dict with a fixed schema.

            Core columns (source, page, allowed_roles, chunk_id,
            ingest_source, ingest_date) are promoted to top-level fields.
            Everything else — including arbitrary PDF metadata like
            'moddate', 'ptex.fullbanner', etc. — is packed into
            ``extra_metadata`` as a JSON string so the schema never varies
            between files.
            """
            core = {}
            extra = {}
            for k, v in meta.items():
                # Normalise key: replace dots (LanceDB forbids them in column names)
                norm_key = k.replace(".", "_")
                if norm_key in _CORE_COLUMNS:
                    core[norm_key] = v
                else:
                    extra[norm_key] = v
            core["extra_metadata"] = json.dumps(extra) if extra else "{}"
            return core

        rows = [
            {"text": text, "vector": emb, **_normalize(meta)}
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
        prefilter: bool = False,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks for *query_embedding*.

        Args:
            query_embedding: Embedding vector of the query.
            k: Number of results to return.
            filters: Optional metadata filters.
                     Scalar value  → equality check:  ``{"allowed_roles": "analyst"}``
                     List value    → IN check:         ``{"allowed_roles": ["analyst", "public"]}``
            prefilter: When True, the WHERE clause is applied *before* ANN search
                       (better for high-selectivity roles like admin/exec that match
                       very few chunks).  When False (default), filtering happens
                       after ANN candidate retrieval.

        Returns:
            List of dicts, each containing at least ``text``, ``score``,
            and any metadata fields stored with the chunk.
        """
        table = self._open_table()
        if table is None:
            raise RuntimeError("No table found. Call add_documents() first.")

        query = table.search(query_embedding).limit(k)

        if filters:
            clauses = []
            for key, value in filters.items():
                if isinstance(value, list):
                    quoted = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v) for v in value
                    )
                    clauses.append(f"{key} IN ({quoted})")
                elif isinstance(value, str):
                    clauses.append(f"{key} = '{value}'")
                else:
                    clauses.append(f"{key} = {value}")
            filter_str = " AND ".join(clauses)
            query = query.where(filter_str, prefilter=prefilter)

        results = query.to_list()

        # Rename LanceDB's internal distance field to a normalised score
        output = []
        for row in results:
            row = dict(row)
            # LanceDB returns L2 distance by default; convert to a similarity-ish score
            distance = row.pop("_distance", None)
            row["score"] = 1.0 / (1.0 + distance) if distance is not None else None
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
