from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_text_splitter(cfg: dict = None):
    if cfg is None:
        cfg = load_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 800),
        chunk_overlap=cfg.get("chunk_overlap", 120),
        separators=["\n\n", "\n", " ", ""]  # fallback to ensure chunks
    )
