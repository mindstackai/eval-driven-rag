import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, cfg: dict):
        self._config = cfg
        emb_cfg = cfg.get("embedding", {})
        mode = emb_cfg.get("mode", "auto")
        if mode == "auto":
            mode = "openai" if os.getenv("OPENAI_API_KEY") else "local"
        self.embedding_mode = mode

    def get(self, key, default=None):
        return self._config.get(key, default)

    def get_index_path(self) -> str:
        return self._config.get("index_path", "./vectorstore/faiss_index")

    def is_incremental_enabled(self) -> bool:
        return self._config.get("incremental", {}).get("enabled", False)

    def get_tracker_path(self) -> str:
        return self._config.get("incremental", {}).get(
            "tracker_path", "./vectorstore/document_tracker.json"
        )

    def get_embeddings(self):
        if self.embedding_mode == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        elif self.embedding_mode == "openai-legacy":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def print_config_summary(self):
        print("=== Configuration ===")
        print(f"  Embedding mode : {self.embedding_mode}")
        print(f"  Index path     : {self.get_index_path()}")
        print(f"  Chunk size     : {self._config.get('chunk_size', 800)}")
        print(f"  Chunk overlap  : {self._config.get('chunk_overlap', 120)}")
        print(f"  Top-k          : {self._config.get('top_k', 4)}")
        print(f"  Incremental    : {self.is_incremental_enabled()}")
        print()


_config_instance: Config | None = None


def get_config(path: str = "config.yaml") -> Config:
    global _config_instance
    if _config_instance is None:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        _config_instance = Config(cfg)
    return _config_instance
