from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import yaml


# --- Profile defaults ---
PROFILES = {
    "general":  {"chunk_size": 800,  "chunk_overlap": 120, "chunking_strategy": "recursive"},
    "legal":    {"chunk_size": 1500, "chunk_overlap": 300, "chunking_strategy": "recursive"},
    "medical":  {"chunk_size": 1000, "chunk_overlap": 200, "chunking_strategy": "semantic"},
    "faq":      {"chunk_size": 400,  "chunk_overlap": 50,  "chunking_strategy": "sentence"},
}

# Keys that can be set explicitly to override a profile
_CHUNK_KEYS = {"chunk_size", "chunk_overlap", "chunking_strategy"}

# Defaults when nothing is specified
_DEFAULTS = {"chunk_size": 800, "chunk_overlap": 120, "chunking_strategy": "recursive"}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_chunking_config(cfg: dict) -> dict:
    """Resolve effective chunk_size, chunk_overlap, and chunking_strategy.

    Priority (highest → lowest):
      1. Explicit top-level values in config.yaml
      2. Profile defaults (if ``profile`` key is set)
      3. Built-in defaults
    """
    # Start with built-in defaults
    resolved = dict(_DEFAULTS)

    # Apply profile if set
    profile_name = cfg.get("profile")
    if profile_name:
        # Check built-in profiles first, then config-defined profiles
        profile = PROFILES.get(profile_name)
        if profile is None:
            profile = cfg.get("profiles", {}).get(profile_name)
        if profile is None:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {', '.join(sorted(set(PROFILES) | set(cfg.get('profiles', {}))))}"
            )
        resolved.update(profile)

    # Explicit top-level values override profile
    for key in _CHUNK_KEYS:
        if key in cfg:
            resolved[key] = cfg[key]

    return resolved


def make_text_splitter(cfg: dict = None):
    if cfg is None:
        cfg = load_config()

    resolved = resolve_chunking_config(cfg)
    strategy = resolved["chunking_strategy"]
    chunk_size = resolved["chunk_size"]
    chunk_overlap = resolved["chunk_overlap"]

    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
    elif strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="",
        )
    elif strategy == "semantic":
        # Semantic splitting: first split into sentences, then group by token budget.
        # Uses RecursiveCharacterTextSplitter with sentence-aware separators.
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
    elif strategy == "sentence":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! "],
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            "Available: recursive, fixed, semantic, sentence"
        )
