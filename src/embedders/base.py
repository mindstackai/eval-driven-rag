from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns shape (n_texts, embedding_dim), float32, normalized."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    def source(self) -> str:
        return "openai" if "openai" in type(self).__name__.lower() else "local"

    @property
    def cost_per_1k_tokens(self) -> float:
        return 0.0  # override in OpenAIEmbedder
