from src.embedders.base import BaseEmbedder
from src.embedders.huggingface import HuggingFaceEmbedder
from src.embedders.openai import OpenAIEmbedder


def load_embedder(model_name: str, config: dict) -> BaseEmbedder:
    """Factory: returns the right embedder based on model source in config."""
    embedding_cfg = config.get("embedding", {})

    # Search base models and finetuned models for this name
    all_model_cfgs = embedding_cfg.get("models", []) + embedding_cfg.get("finetuned_models", [])
    model_cfg = next((m for m in all_model_cfgs if m["name"] == model_name), None)

    if model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in config.yaml embedding section.")

    source = model_cfg.get("source", "local")
    dim = model_cfg.get("dim")

    if source == "openai":
        cost = model_cfg.get("cost_per_1k_tokens", 0.0)
        return OpenAIEmbedder(
            model_name=model_name,
            embedding_dim=dim,
            cost_per_1k_tokens=cost,
        )
    else:
        batch_size = embedding_cfg.get("batch_size", 32)
        normalize = embedding_cfg.get("normalize_embeddings", True)
        return HuggingFaceEmbedder(
            model_name=model_name,
            embedding_dim=dim,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )


__all__ = ["BaseEmbedder", "HuggingFaceEmbedder", "OpenAIEmbedder", "load_embedder"]
