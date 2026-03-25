import numpy as np

from src.embedders.base import BaseEmbedder


def _detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._device = _detect_device()

        from sentence_transformers import SentenceTransformer

        # NOTE: MPS does not support fp16/bf16 — keep both False
        self._model = SentenceTransformer(model_name, device=self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            # NOTE: If MPS op error, add PYTORCH_ENABLE_MPS_FALLBACK=1 to env
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device
