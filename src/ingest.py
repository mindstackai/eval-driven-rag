from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import lancedb
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.splitters import make_text_splitter
from src.config_manager import get_config
from src.vectorstore.lancedb_store import LanceDBStore

RAW_DIR = Path("data/raw")

IngestSource = Literal["bulk", "admin"]


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


def ingest_file_to_lancedb(
    file_path: Path,
    allowed_role: str,
    config,
    ingest_source: IngestSource = "admin",
) -> int:
    """
    Ingest a single file into LanceDB with the given access role.

    Single entry point for both bulk CLI (`ingest_source="bulk"`) and the
    admin UI (`ingest_source="admin"`).  The origin is stored in the
    ``ingest_source`` metadata column so you can query or audit it later:

        WHERE ingest_source = 'admin'

    Deduplication: chunks whose chunk_id (content hash) already exist in
    the table are skipped.  Existing data is never deleted — safe against
    name-collision attacks where an attacker re-uploads a file to overwrite
    existing chunks.

    Args:
        file_path:      Path to a PDF, .txt, or .md file.
        allowed_role:   Role string tagged on every chunk (e.g. "analyst").
        config:         Config instance from get_config().
        ingest_source:  "bulk" (CLI rebuild) or "admin" (UI upload).

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

    ingested_at = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata["allowed_roles"] = allowed_role
        chunk.metadata["ingest_source"] = ingest_source
        chunk.metadata["ingested_at"] = ingested_at

    store = LanceDBStore(
        db_path=config.get_lancedb_path(),
        table_name=config.get_lancedb_table(),
    )

    # Deduplicate by chunk_id — never delete, only skip existing.
    table = store._open_table()
    if table is not None:
        existing_ids = set(
            table.search().select(["chunk_id"]).limit(None).to_pandas()["chunk_id"].tolist()
        )
        new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_ids]
    else:
        new_chunks = chunks

    if not new_chunks:
        return 0

    embeddings_model = config.get_embeddings()
    texts = [c.page_content for c in new_chunks]
    vectors = embeddings_model.embed_documents(texts)
    metadatas = [c.metadata for c in new_chunks]

    store.add_documents(texts, vectors, metadatas)

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        print(f"  [{ingest_source}] Skipped {skipped} duplicate chunk(s) already in the index.")

    return len(new_chunks)


def main():
    """Ingest files from data/raw/ into LanceDB.

    Idempotent by default: chunks are skipped if their chunk_id (content hash)
    already exists. Re-running with unchanged files does nothing.

    Use --force to wipe all bulk-ingested chunks first (e.g. after removing
    files from data/raw/). Admin-uploaded chunks are never affected.
    """
    parser = argparse.ArgumentParser(description="Ingest documents from data/raw/ into LanceDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing bulk-ingested chunks before re-ingesting. "
             "Use when files have been removed from data/raw/. "
             "Admin-uploaded chunks are preserved.",
    )
    args = parser.parse_args()

    config = get_config()
    config.print_config_summary()

    raw_files = [
        p for p in RAW_DIR.glob("**/*")
        if not p.is_dir() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    ]
    if not raw_files:
        print("No documents found in data/raw/. Add PDFs or .txt files and re-run.")
        return

    if args.force:
        db = lancedb.connect(config.get_lancedb_path())
        if config.get_lancedb_table() in db.table_names():
            tbl = db.open_table(config.get_lancedb_table())
            tbl.delete("ingest_source = 'bulk'")
            print(f"--force: removed existing bulk chunks (admin chunks preserved)")

    total = 0
    for file_path in raw_files:
        print(f"  Ingesting {file_path.name} ...")
        n = ingest_file_to_lancedb(file_path, allowed_role="public", config=config, ingest_source="bulk")
        print(f"    → {n} chunks")
        total += n

    if total == 0:
        print("Nothing new to ingest — all chunks already in the index.")
    else:
        print(f"\nStored {total} new bulk chunks in LanceDB → {config.get_lancedb_path()}")


if __name__ == "__main__":
    main()
