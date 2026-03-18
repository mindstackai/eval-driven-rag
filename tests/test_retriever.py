"""
tests/test_retriever.py — Integration tests for LanceDBRetriever.

Uses real OpenAI Embeddings API (requires OPENAI_API_KEY in .env).
Embeddings are computed once per module (scope="module") to minimise cost.

Coverage:
  - Role-based filtering: admin / analyst / public / unauthenticated
  - Return types: Document objects, metadata keys present, scores are floats
  - k parameter: limits results, gracefully returns fewer when set exceeds pool
  - Auth integration: get_user_roles() expansion wired to LanceDBRetriever
"""
import pytest
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

import src.auth as auth
from src.auth import get_user_roles, set_user_role
from src.retriever import LanceDBRetriever
from src.vectorstore.lancedb_store import LanceDBStore


# ── Test data ─────────────────────────────────────────────────────────────────

CHUNKS = {
    "public_doc":  "This document covers general product information for all users.",
    "analyst_doc": "This report contains quarterly financial analysis and market trends.",
    "admin_doc":   "This file holds confidential executive strategy and internal policies.",
}
ROLES = {
    "public_doc":  "public",
    "analyst_doc": "analyst",
    "admin_doc":   "admin",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embeddings():
    """Real OpenAI embeddings — initialised once per module to save API calls."""
    from src.config_manager import get_config
    return get_config().get_embeddings()


@pytest.fixture(scope="module")
def populated_store(tmp_path_factory, embeddings):
    """Temp LanceDB with three chunks (one per role), embedded with OpenAI."""
    db_dir = tmp_path_factory.mktemp("retriever_db")
    store = LanceDBStore(db_path=str(db_dir), table_name="test")

    keys = list(CHUNKS.keys())
    texts = [CHUNKS[k] for k in keys]
    vecs = embeddings.embed_documents(texts)
    metas = [{"allowed_roles": ROLES[k], "source": k} for k in keys]

    store.add_documents(texts, vecs, metas)
    return store


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    """Redirect auth I/O to a temp roles.yaml so tests don't touch the real file."""
    monkeypatch.setattr(auth, "ROLES_PATH", tmp_path / "roles.yaml")


def _make_retriever(store, embeddings, user_roles, k=10) -> LanceDBRetriever:
    return LanceDBRetriever(store=store, embeddings=embeddings, k=k, user_roles=user_roles)


def _roles_seen(docs: list[Document]) -> set[str]:
    return {d.metadata["allowed_roles"] for d in docs}


# ── Role filtering ─────────────────────────────────────────────────────────────

class TestRoleFiltering:
    def test_admin_roles_see_all_chunks(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["admin", "analyst", "public"])
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"admin", "analyst", "public"}

    def test_analyst_roles_exclude_admin(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["analyst", "public"])
        docs = retriever.invoke("document information")
        assert "admin" not in _roles_seen(docs)
        assert {"analyst", "public"}.issubset(_roles_seen(docs))

    def test_public_role_sees_only_public(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["public"])
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"public"}

    def test_no_filter_returns_all_roles(self, populated_store, embeddings):
        """user_roles=None disables filtering — used for unauthenticated access."""
        retriever = _make_retriever(populated_store, embeddings, user_roles=None)
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"admin", "analyst", "public"}

    def test_admin_cannot_see_nonexistent_role(self, populated_store, embeddings):
        """Requesting a role that has no matching chunks returns empty for that role."""
        retriever = _make_retriever(populated_store, embeddings, ["superadmin"])
        docs = retriever.invoke("document information")
        assert len(docs) == 0


# ── Return types ───────────────────────────────────────────────────────────────

class TestReturnTypes:
    def test_invoke_returns_document_objects(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["public"])
        docs = retriever.invoke("document")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_required_metadata_keys(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["public"])
        docs = retriever.invoke("document")
        for doc in docs:
            assert "allowed_roles" in doc.metadata
            assert "source" in doc.metadata

    def test_documents_have_page_content(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["public"])
        docs = retriever.invoke("document")
        assert all(isinstance(d.page_content, str) and len(d.page_content) > 0 for d in docs)

    def test_similarity_search_returns_doc_score_tuples(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["public"])
        results = retriever.similarity_search_with_relevance_scores("document", k=5)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert score >= 0.0


# ── k parameter ────────────────────────────────────────────────────────────────

class TestKParameter:
    def test_k_caps_number_of_results(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["admin", "analyst", "public"], k=2)
        docs = retriever.invoke("document information")
        assert len(docs) <= 2

    def test_k_larger_than_matching_pool_returns_all_matches(self, populated_store, embeddings):
        """Only 1 public chunk exists — k=10 should still return just 1."""
        retriever = _make_retriever(populated_store, embeddings, ["public"], k=10)
        docs = retriever.invoke("document information")
        assert len(docs) == 1

    def test_k1_returns_single_best_match(self, populated_store, embeddings):
        retriever = _make_retriever(populated_store, embeddings, ["admin", "analyst", "public"], k=1)
        docs = retriever.invoke("confidential executive strategy")
        assert len(docs) == 1
        # Most semantically similar chunk to "confidential executive strategy" should be admin_doc
        assert docs[0].metadata["source"] == "admin_doc"


# ── Auth integration ───────────────────────────────────────────────────────────

class TestAuthIntegration:
    """Verify that get_user_roles() expands correctly and wires into the retriever."""

    def test_admin_user_sees_all_three_roles(self, populated_store, embeddings):
        set_user_role("alice", "admin")
        roles = get_user_roles("alice")                          # ["admin", "analyst", "public"]
        retriever = _make_retriever(populated_store, embeddings, roles)
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"admin", "analyst", "public"}

    def test_analyst_user_excludes_admin_chunks(self, populated_store, embeddings):
        set_user_role("bob", "analyst")
        roles = get_user_roles("bob")                           # ["analyst", "public"]
        retriever = _make_retriever(populated_store, embeddings, roles)
        docs = retriever.invoke("document information")
        assert "admin" not in _roles_seen(docs)

    def test_public_user_sees_only_public(self, populated_store, embeddings):
        set_user_role("carol", "public")
        roles = get_user_roles("carol")                         # ["public"]
        retriever = _make_retriever(populated_store, embeddings, roles)
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"public"}

    def test_unknown_user_defaults_to_public(self, populated_store, embeddings):
        # "dave" not in roles.yaml → defaults to public role
        roles = get_user_roles("dave")                          # ["public"]
        retriever = _make_retriever(populated_store, embeddings, roles)
        docs = retriever.invoke("document information")
        assert _roles_seen(docs) == {"public"}
