from dotenv import load_dotenv
load_dotenv()

import hashlib
import lancedb
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.splitters import make_text_splitter
from src.config_manager import get_config
from src.vectorstore.lancedb_store import LanceDBStore

RAW_DIR = Path("data/raw")


def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign deterministic chunk_id to each document based on source, page, and content."""
    for i, doc in enumerate(chunks):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        doc.metadata["chunk_id"] = f"{source}:p{page}:c{i}:{content_hash}"
    return chunks


def load_docs() -> list[Document]:
    """Load all supported documents from data/raw/."""
    docs = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def ingest_file_to_lancedb(file_path: Path, allowed_role: str, config) -> int:
    """
    Ingest a single file into LanceDB with the given access role.

    Used by the admin UI for per-file role assignment.
    Appends new chunks to the existing table — skips chunks whose chunk_id
    (content hash) already exists.  Never deletes existing data, so re-uploading
    the same file is idempotent and safe against name-collision attacks.

    Args:
        file_path: Path to a PDF, .txt, or .md file.
        allowed_role: Role string tagged on every chunk (e.g. "analyst").
        config: Config instance from get_config().

    Returns:
        Number of new chunks stored (0 if all already existed).
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()

    splitter = make_text_splitter(config._config)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)

    for chunk in chunks:
        chunk.metadata["allowed_roles"] = allowed_role

    store = LanceDBStore(
        db_path=config.get_lancedb_path(),
        table_name=config.get_lancedb_table(),
    )

    # Deduplicate by chunk_id (content hash) — safe against re-upload attacks.
    # We never delete existing data; we only skip chunks already present.
    table = store._open_table()
    if table is not None:
        existing_ids = set(
            table.search().select(["chunk_id"]).limit(None).to_pandas()["chunk_id"].tolist()
        )
        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    else:
        new_chunks = chunks

    if not new_chunks:
        return 0  # all chunks already exist, nothing to do

    embeddings_model = config.get_embeddings()
    texts = [c.page_content for c in new_chunks]
    vectors = embeddings_model.embed_documents(texts)
    metadatas = [c.metadata for c in new_chunks]

    store.add_documents(texts, vectors, metadatas)

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  Skipped {skipped} duplicate chunk(s) already in the index.")

    return len(new_chunks)


def main():
    config = get_config()
    config.print_config_summary()

    docs = load_docs()
    if not docs:
        print("No documents found in data/raw/. Add PDFs or .txt files and re-run.")
        return

    splitter = make_text_splitter(config._config)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)

    # Bulk ingest tags everything as public — use the admin UI to assign other roles
    for chunk in chunks:
        chunk.metadata.setdefault("allowed_roles", "public")

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")
    print("Embedding chunks...")

    embeddings_model = config.get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)
    metadatas = [c.metadata for c in chunks]

    # Drop existing table so this is always a clean rebuild
    db = lancedb.connect(config.get_lancedb_path())
    if config.get_lancedb_table() in db.table_names():
        db.drop_table(config.get_lancedb_table())
        print(f"Dropped existing table '{config.get_lancedb_table()}'")

    store = LanceDBStore(
        db_path=config.get_lancedb_path(),
        table_name=config.get_lancedb_table(),
    )
    store.add_documents(texts, vectors, metadatas)

    print(f"Stored {len(chunks)} chunks in LanceDB → {config.get_lancedb_path()}")
    print()


if __name__ == "__main__":
    main()
