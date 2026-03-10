import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from src.splitters import resolve_chunking_config, make_text_splitter


# --- resolve_chunking_config ---

class TestResolvChunkingConfig:
    def test_defaults_when_empty(self):
        r = resolve_chunking_config({})
        assert r == {"chunk_size": 800, "chunk_overlap": 120, "chunking_strategy": "recursive"}

    def test_explicit_values_used(self):
        r = resolve_chunking_config({"chunk_size": 500, "chunk_overlap": 60, "chunking_strategy": "fixed"})
        assert r["chunk_size"] == 500
        assert r["chunk_overlap"] == 60
        assert r["chunking_strategy"] == "fixed"

    def test_profile_sets_all_values(self):
        r = resolve_chunking_config({"profile": "legal"})
        assert r["chunk_size"] == 1500
        assert r["chunk_overlap"] == 300
        assert r["chunking_strategy"] == "recursive"

    def test_profile_faq(self):
        r = resolve_chunking_config({"profile": "faq"})
        assert r["chunk_size"] == 400
        assert r["chunk_overlap"] == 50
        assert r["chunking_strategy"] == "sentence"

    def test_profile_medical(self):
        r = resolve_chunking_config({"profile": "medical"})
        assert r["chunk_size"] == 1000
        assert r["chunking_strategy"] == "semantic"

    def test_explicit_overrides_profile(self):
        r = resolve_chunking_config({"profile": "legal", "chunk_size": 2000})
        assert r["chunk_size"] == 2000
        assert r["chunk_overlap"] == 300  # from profile

    def test_explicit_strategy_overrides_profile(self):
        r = resolve_chunking_config({"profile": "faq", "chunking_strategy": "fixed"})
        assert r["chunking_strategy"] == "fixed"
        assert r["chunk_size"] == 400  # from profile

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_chunking_config({"profile": "nonexistent"})

    def test_custom_profile_from_config(self):
        cfg = {
            "profile": "custom",
            "profiles": {"custom": {"chunk_size": 999, "chunk_overlap": 99, "chunking_strategy": "fixed"}},
        }
        r = resolve_chunking_config(cfg)
        assert r["chunk_size"] == 999
        assert r["chunk_overlap"] == 99
        assert r["chunking_strategy"] == "fixed"

    def test_backward_compat_old_config(self):
        """Config with only old-style fields (no profile, no strategy) should work."""
        r = resolve_chunking_config({"chunk_size": 600, "chunk_overlap": 80})
        assert r["chunk_size"] == 600
        assert r["chunk_overlap"] == 80
        assert r["chunking_strategy"] == "recursive"


# --- make_text_splitter ---

class TestMakeTextSplitter:
    def test_recursive_strategy(self):
        s = make_text_splitter({"chunk_size": 500, "chunk_overlap": 50, "chunking_strategy": "recursive"})
        assert isinstance(s, RecursiveCharacterTextSplitter)

    def test_fixed_strategy(self):
        s = make_text_splitter({"chunk_size": 500, "chunk_overlap": 50, "chunking_strategy": "fixed"})
        assert isinstance(s, CharacterTextSplitter)

    def test_semantic_strategy(self):
        s = make_text_splitter({"chunk_size": 500, "chunk_overlap": 50, "chunking_strategy": "semantic"})
        assert isinstance(s, RecursiveCharacterTextSplitter)

    def test_sentence_strategy(self):
        s = make_text_splitter({"chunk_size": 500, "chunk_overlap": 50, "chunking_strategy": "sentence"})
        assert isinstance(s, RecursiveCharacterTextSplitter)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            make_text_splitter({"chunk_size": 500, "chunk_overlap": 50, "chunking_strategy": "magic"})

    def test_splitter_respects_chunk_size(self):
        # Use text with sentence boundaries so all strategies can split
        text = "This is a sentence. " * 200  # ~4000 chars with ". " separators
        for strategy in ["recursive", "fixed", "semantic", "sentence"]:
            s = make_text_splitter({"chunk_size": 200, "chunk_overlap": 20, "chunking_strategy": strategy})
            chunks = s.split_text(text)
            assert len(chunks) > 1, f"{strategy} should produce multiple chunks"
            for chunk in chunks:
                assert len(chunk) <= 220, f"{strategy} chunk too large: {len(chunk)}"

    def test_profile_produces_working_splitter(self):
        text = "Hello world. This is a test document. " * 100
        for profile in ["general", "legal", "medical", "faq"]:
            s = make_text_splitter({"profile": profile})
            chunks = s.split_text(text)
            assert len(chunks) > 0
