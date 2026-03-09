import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from src.config_manager import get_config

load_dotenv()

def build_faiss(docs, index_path: str = None):
    """Build FAISS index from documents using configured embeddings"""
    config = get_config()
    if index_path is None:
        index_path = config.get_index_path()

    embeddings = config.get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_path)
    return vs

def update_faiss(docs, index_path: str = None):
    """Add new documents to existing FAISS index"""
    config = get_config()
    if index_path is None:
        index_path = config.get_index_path()

    embeddings = config.get_embeddings()

    # Try to load existing index
    if Path(index_path).exists():
        vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
        vs.save_local(index_path)
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(index_path)

    return vs

def load_faiss(index_path: str = None):
    """Load FAISS index using configured embeddings"""
    config = get_config()
    if index_path is None:
        index_path = config.get_index_path()

    embeddings = config.get_embeddings()
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
