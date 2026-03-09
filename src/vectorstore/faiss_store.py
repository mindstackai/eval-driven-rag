"""
FAISSStore — wraps the LangChain FAISS vector store for use within the
vectorstore abstraction layer. The original free functions (build_faiss,
update_faiss, load_faiss) in src/embed_store.py remain untouched for
backwards compatibility with the existing Streamlit app and ingest pipeline.
"""
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()


class FAISSStore:
    """Thin class wrapper around LangChain's FAISS vector store."""

    def __init__(self, index_path: str, embeddings: Any) -> None:
        """
        Args:
            index_path: Directory path where the FAISS index is (or will be) saved.
            embeddings: A LangChain Embeddings instance used to encode documents
                        and queries.
        """
        self.index_path = index_path
        self.embeddings = embeddings
        self._store: FAISS | None = None

    def add_documents(self, docs: list) -> None:
        """
        Add documents to the store, building a new index if one does not yet
        exist, or updating the existing one.

        Args:
            docs: List of LangChain Document objects to embed and store.
        """
        if self._store is None and Path(self.index_path).exists():
            self._store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        if self._store is None:
            self._store = FAISS.from_documents(docs, self.embeddings)
        else:
            self._store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> list:
        """
        Return the top-k most similar documents for *query*.

        Args:
            query: Plain-text query string.
            k: Number of results to return.

        Returns:
            List of LangChain Document objects ranked by similarity.
        """
        if self._store is None:
            raise RuntimeError("No index loaded. Call add_documents() or load() first.")
        return self._store.similarity_search(query, k=k)

    def save(self) -> None:
        """Persist the FAISS index to *self.index_path*."""
        if self._store is None:
            raise RuntimeError("Nothing to save — the store is empty.")
        self._store.save_local(self.index_path)

    def load(self) -> None:
        """Load a previously saved FAISS index from *self.index_path*."""
        self._store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def as_retriever(self, k: int = 4):
        """
        Return a LangChain retriever configured for top-k retrieval.

        Args:
            k: Number of documents to retrieve per query.
        """
        if self._store is None:
            raise RuntimeError("No index loaded. Call add_documents() or load() first.")
        return self._store.as_retriever(search_kwargs={"k": k})
