"""
LanceDB-backed retriever with role-based access control.

LanceDBRetriever wraps LanceDBStore in a LangChain-compatible interface
so the rest of the codebase (app, eval) can call .invoke() or
.similarity_search_with_relevance_scores() without knowing the store type.
"""
from langchain_core.documents import Document

from src.config_manager import get_config
from src.vectorstore.lancedb_store import LanceDBStore


class LanceDBRetriever:
    """LangChain-compatible retriever backed by LanceDB."""

    def __init__(self, store: LanceDBStore, embeddings, k: int, user_roles: list[str] = None):
        self.store = store
        self.embeddings = embeddings
        self.k = k
        self.user_roles = user_roles

    def _to_document(self, row: dict) -> Document:
        text = row.get("text", "")
        meta = {k: v for k, v in row.items() if k not in ("text", "vector", "score")}
        return Document(page_content=text, metadata=meta)

    def invoke(self, query: str) -> list[Document]:
        """Return top-k Documents, filtered by the user's roles."""
        vec = self.embeddings.embed_query(query)
        filters = {"allowed_roles": self.user_roles} if self.user_roles else None
        results = self.store.similarity_search(vec, self.k, filters=filters)
        return [self._to_document(r) for r in results]

    def similarity_search_with_relevance_scores(self, query: str, k: int) -> list[tuple]:
        """Return (Document, score) pairs — used by the eval pipeline."""
        vec = self.embeddings.embed_query(query)
        filters = {"allowed_roles": self.user_roles} if self.user_roles else None
        results = self.store.similarity_search(vec, k, filters=filters)
        return [(self._to_document(r), r.get("score", 0.0)) for r in results]


def get_retriever(user_id: str = None, k: int = None) -> LanceDBRetriever:
    """
    Return a LanceDBRetriever scoped to the given user's accessible roles.

    Args:
        user_id: If provided, only chunks whose allowed_roles matches the
                 user's expanded role list are returned.
                 If None, no role filter is applied (open access).
        k: Number of results to return. Defaults to config top_k.
    """
    config = get_config()
    embeddings = config.get_embeddings()

    store = LanceDBStore(
        db_path=config.get_lancedb_path(),
        table_name=config.get_lancedb_table(),
    )
    store.load()

    if k is None:
        k = config.get("top_k", 3)

    user_roles = None
    if user_id:
        from src.auth import get_user_roles
        user_roles = get_user_roles(user_id)

    return LanceDBRetriever(store, embeddings, k, user_roles)
