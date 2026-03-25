import os

import numpy as np

from src.embedders.base import BaseEmbedder

_OPENAI_BATCH_LIMIT = 100  # OpenAI embeddings API limit per request


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        cost_per_1k_tokens: float = 0.0,
    ):
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._cost = cost_per_1k_tokens

        from openai import OpenAI
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment.")
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[list[float]] = []

        # Batch at 100 texts max (OpenAI limit)
        for i in range(0, len(texts), _OPENAI_BATCH_LIMIT):
            batch = texts[i : i + _OPENAI_BATCH_LIMIT]
            response = self._client.embeddings.create(model=self._model_name, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalize to unit length
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def source(self) -> str:
        return "openai"

    @property
    def cost_per_1k_tokens(self) -> float:
        return self._cost
