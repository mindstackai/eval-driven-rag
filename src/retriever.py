from src.embed_store import load_faiss
from src.splitters import load_config

def get_retriever():
    cfg = load_config()
    vs = load_faiss(cfg["index_path"])
    return vs.as_retriever(search_kwargs={"k": cfg.get("top_k", 4)})
