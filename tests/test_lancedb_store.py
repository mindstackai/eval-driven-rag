"""
Tests for src/vectorstore/lancedb_store.py

Uses real LanceDB with tiny dummy vectors (no embeddings API needed).
Each test gets its own tmp_path so there is no shared state.
"""
import pytest

from src.vectorstore.lancedb_store import LanceDBStore


# ── Shared dummy data ──────────────────────────────────────────────

# 4-dimensional unit vectors — small enough to be fast, real enough for
# LanceDB's ANN index to handle without errors.
VECTORS = {
    "public_1":  [1.0, 0.0, 0.0, 0.0],
    "public_2":  [0.9, 0.1, 0.0, 0.0],
    "analyst_1": [0.0, 1.0, 0.0, 0.0],
    "analyst_2": [0.0, 0.9, 0.1, 0.0],
    "admin_1":   [0.0, 0.0, 1.0, 0.0],
}

CHUNKS = {
    "public_1":  "Public document one.",
    "public_2":  "Public document two.",
    "analyst_1": "Analyst report one.",
    "analyst_2": "Analyst report two.",
    "admin_1":   "Admin-only document.",
}

ROLES = {
    "public_1":  "public",
    "public_2":  "public",
    "analyst_1": "analyst",
    "analyst_2": "analyst",
    "admin_1":   "admin",
}


@pytest.fixture
def store(tmp_path):
    """Fresh LanceDB store populated with all five test chunks."""
    s = LanceDBStore(db_path=str(tmp_path / "db"), table_name="test")
    keys = list(CHUNKS.keys())
    s.add_documents(
        chunks=[CHUNKS[k] for k in keys],
        embeddings=[VECTORS[k] for k in keys],
        metadatas=[{"allowed_roles": ROLES[k], "source": k} for k in keys],
    )
    return s


# ──────────────────────────────────────────────
# add_documents
# ──────────────────────────────────────────────

class TestAddDocuments:
    def test_creates_table(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        s.add_documents(["hello"], [[1.0, 0.0, 0.0, 0.0]], [{"allowed_roles": "public"}])
        s.load()  # should not raise

    def test_mismatched_lengths_raise(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        with pytest.raises(ValueError, match="same length"):
            s.add_documents(["a", "b"], [[1.0, 0.0]], [{}])

    def test_appends_to_existing_table(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        s.add_documents(["first"], [[1.0, 0.0, 0.0, 0.0]], [{"allowed_roles": "public"}])
        s.add_documents(["second"], [[0.0, 1.0, 0.0, 0.0]], [{"allowed_roles": "analyst"}])
        results = s.similarity_search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(results) == 2


# ──────────────────────────────────────────────
# similarity_search — no filter
# ──────────────────────────────────────────────

class TestSearchNoFilter:
    def test_returns_k_results(self, store):
        results = store.similarity_search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3

    def test_result_has_expected_keys(self, store):
        results = store.similarity_search([1.0, 0.0, 0.0, 0.0], k=1)
        assert "text" in results[0]
        assert "score" in results[0]
        assert "allowed_roles" in results[0]

    def test_closest_vector_ranked_first(self, store):
        # Query closest to public_1 vector
        results = store.similarity_search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["text"] == CHUNKS["public_1"]

    def test_score_is_positive(self, store):
        results = store.similarity_search([1.0, 0.0, 0.0, 0.0], k=5)
        for r in results:
            assert r["score"] > 0


# ──────────────────────────────────────────────
# similarity_search — scalar filter (equality)
# ──────────────────────────────────────────────

class TestSearchScalarFilter:
    def test_public_filter_returns_only_public(self, store):
        results = store.similarity_search(
            [1.0, 0.0, 0.0, 0.0], k=5,
            filters={"allowed_roles": "public"},
        )
        assert all(r["allowed_roles"] == "public" for r in results)
        assert len(results) == 2  # exactly 2 public chunks

    def test_analyst_filter_excludes_public_and_admin(self, store):
        results = store.similarity_search(
            [0.0, 1.0, 0.0, 0.0], k=5,
            filters={"allowed_roles": "analyst"},
        )
        assert all(r["allowed_roles"] == "analyst" for r in results)
        assert len(results) == 2

    def test_admin_filter_returns_only_admin(self, store):
        results = store.similarity_search(
            [0.0, 0.0, 1.0, 0.0], k=5,
            filters={"allowed_roles": "admin"},
        )
        assert len(results) == 1
        assert results[0]["allowed_roles"] == "admin"


# ──────────────────────────────────────────────
# similarity_search — list filter (IN clause)
# This is the role hierarchy case: admin queries with expanded role list
# ──────────────────────────────────────────────

class TestSearchListFilter:
    def test_admin_list_sees_all_chunks(self, store):
        """Admin role expands to [admin, analyst, public] — should see all 5 chunks."""
        results = store.similarity_search(
            [1.0, 0.0, 0.0, 0.0], k=10,
            filters={"allowed_roles": ["admin", "analyst", "public"]},
        )
        assert len(results) == 5

    def test_analyst_list_excludes_admin(self, store):
        """Analyst expands to [analyst, public] — should not see admin chunk."""
        results = store.similarity_search(
            [1.0, 0.0, 0.0, 0.0], k=10,
            filters={"allowed_roles": ["analyst", "public"]},
        )
        roles_returned = {r["allowed_roles"] for r in results}
        assert "admin" not in roles_returned
        assert len(results) == 4  # 2 analyst + 2 public

    def test_public_list_sees_only_public(self, store):
        results = store.similarity_search(
            [1.0, 0.0, 0.0, 0.0], k=10,
            filters={"allowed_roles": ["public"]},
        )
        assert all(r["allowed_roles"] == "public" for r in results)
        assert len(results) == 2

    def test_single_item_list_same_as_scalar(self, store):
        list_results = store.similarity_search(
            [0.0, 1.0, 0.0, 0.0], k=5,
            filters={"allowed_roles": ["analyst"]},
        )
        scalar_results = store.similarity_search(
            [0.0, 1.0, 0.0, 0.0], k=5,
            filters={"allowed_roles": "analyst"},
        )
        list_texts = {r["text"] for r in list_results}
        scalar_texts = {r["text"] for r in scalar_results}
        assert list_texts == scalar_texts


# ──────────────────────────────────────────────
# prefilter param
# ──────────────────────────────────────────────

class TestPrefilter:
    def test_prefilter_true_returns_correct_results(self, store):
        """prefilter=True should return the same filtered results as post-filter."""
        post = store.similarity_search(
            [0.0, 0.0, 1.0, 0.0], k=5,
            filters={"allowed_roles": "admin"},
            prefilter=False,
        )
        pre = store.similarity_search(
            [0.0, 0.0, 1.0, 0.0], k=5,
            filters={"allowed_roles": "admin"},
            prefilter=True,
        )
        assert {r["text"] for r in post} == {r["text"] for r in pre}

    def test_prefilter_with_list_filter(self, store):
        pre = store.similarity_search(
            [1.0, 0.0, 0.0, 0.0], k=10,
            filters={"allowed_roles": ["analyst", "public"]},
            prefilter=True,
        )
        assert len(pre) == 4
        assert all(r["allowed_roles"] in {"analyst", "public"} for r in pre)


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

class TestMetadataSanitization:
    def test_dot_in_metadata_key_is_stored(self, tmp_path):
        """PyPDF injects keys like 'ptex.fullbanner' — dots must be sanitized."""
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        s.add_documents(
            ["pdf chunk"],
            [[1.0, 0.0, 0.0, 0.0]],
            [{"ptex.fullbanner": "TeX", "allowed_roles": "public"}],
        )
        results = s.similarity_search([1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["ptex_fullbanner"] == "TeX"

    def test_multiple_dots_sanitized(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        s.add_documents(
            ["chunk"],
            [[1.0, 0.0, 0.0, 0.0]],
            [{"a.b.c": "val", "allowed_roles": "public"}],
        )
        results = s.similarity_search([1.0, 0.0, 0.0, 0.0], k=1)
        assert "a_b_c" in results[0]


class TestEdgeCases:
    def test_search_before_add_raises(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "tbl")
        with pytest.raises(RuntimeError, match="No table found"):
            s.similarity_search([1.0, 0.0, 0.0, 0.0], k=3)

    def test_load_missing_table_raises(self, tmp_path):
        s = LanceDBStore(str(tmp_path / "db"), "nonexistent")
        with pytest.raises(RuntimeError, match="not found"):
            s.load()

    def test_k_larger_than_filtered_set_returns_all_matches(self, store):
        """Requesting k=10 when only 1 admin chunk exists should return 1."""
        results = store.similarity_search(
            [0.0, 0.0, 1.0, 0.0], k=10,
            filters={"allowed_roles": "admin"},
        )
        assert len(results) == 1

    def test_save_is_noop(self, store):
        store.save()  # should not raise
