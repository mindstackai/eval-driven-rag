from dotenv import load_dotenv
load_dotenv()
import hashlib
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.splitters import make_text_splitter
from src.embed_store import build_faiss, update_faiss, index_exists
from src.config_manager import get_config
from src.document_tracker import DocumentTracker

RAW_DIR = Path("data/raw")


def assign_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign deterministic chunk_id to each document based on source, page, and content.

    IDs are stable across re-indexing as long as the content doesn't change.
    Works with any vector store (FAISS, LanceDB, etc.).
    """
    for i, doc in enumerate(chunks):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
        doc.metadata["chunk_id"] = f"{source}:p{page}:c{i}:{content_hash}"
    return chunks

def load_docs():
    docs = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() in [".pdf"]:
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def load_docs_with_tracking(tracker: DocumentTracker):
    """Load only new or changed documents"""
    all_files = [p for p in RAW_DIR.glob("**/*")
                 if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]]

    # Clean up tracker for missing files
    num_removed = tracker.clean_missing_files(set(all_files))
    if num_removed > 0:
        print(f"Removed {num_removed} deleted files from tracker")

    docs_to_process = []
    files_to_process = []

    for p in all_files:
        if tracker.is_changed(p):
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                docs = loader.load()
                docs_to_process.extend(docs)
                files_to_process.append((p, len(docs)))
            elif p.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(p), encoding="utf-8")
                docs = loader.load()
                docs_to_process.extend(docs)
                files_to_process.append((p, len(docs)))

    return docs_to_process, files_to_process, len(all_files)

def main():
    config = get_config()

    # Print configuration summary
    config.print_config_summary()

    # Check if incremental indexing is enabled
    if config.is_incremental_enabled():
        tracker = DocumentTracker(config.get_tracker_path())
        old_stats = tracker.get_stats()

        # Check if embedding mode has changed
        if not tracker.check_embedding_mode_compatibility(config.embedding_mode):
            print(f"Embedding mode changed: {tracker.embedding_mode} -> {config.embedding_mode}")
            print("Different embedding models have different dimensions.")
            print("Rebuilding entire index from scratch...")
            print()

            # Delete old index
            index_path = Path(config.get_index_path())
            if index_path.exists():
                import shutil
                shutil.rmtree(index_path)
                print(f"Deleted old index at {index_path}")
                print()

            tracker.reset()
            old_stats = {"total_files": 0, "total_chunks": 0}

        print(f"Current index: {old_stats['total_files']} files, {old_stats['total_chunks']} chunks")
        print()

        # Load only new/changed documents
        docs, files_to_process, total_files = load_docs_with_tracking(tracker)

        if not docs:
            # Check if index actually exists - tracker may be out of sync
            if not index_exists(config.get_index_path()):
                print("Index is missing! Resetting tracker and rebuilding...")
                print()
                tracker.reset()
                old_stats = {"total_files": 0, "total_chunks": 0}
                docs, files_to_process, total_files = load_docs_with_tracking(tracker)
                if not docs:
                    print("No documents found in data/raw. Add PDFs or .txt files and re-run.")
                    return
            else:
                print("All documents are up to date. No changes to embed.")
                print()
                return

        print(f"Found {len(files_to_process)} new/changed files out of {total_files} total:")
        for file_path, num_docs in files_to_process[:5]:  # Show first 5
            print(f"   * {file_path.name} ({num_docs} pages)")
        if len(files_to_process) > 5:
            print(f"   ... and {len(files_to_process) - 5} more")
        print()

        # Split into chunks
        splitter = make_text_splitter(config._config)
        chunks = splitter.split_documents(docs)
        assign_chunk_ids(chunks)
        print(f"Split into {len(chunks)} chunks")
        print()

        # Embed chunks - use build_faiss if no existing index, otherwise update
        print("Embedding chunks...")
        if old_stats['total_chunks'] == 0:
            build_faiss(chunks, config.get_index_path())
        else:
            update_faiss(chunks, config.get_index_path())

        # Update tracker
        for file_path, _ in files_to_process:
            file_chunks = [c for c in chunks if c.metadata.get('source') == str(file_path)]
            tracker.update(file_path, len(file_chunks))

        # Save embedding mode
        tracker.set_embedding_mode(config.embedding_mode)
        tracker.save()

        new_stats = tracker.get_stats()
        print()
        print("Index updated successfully!")
        print(f"   Total: {new_stats['total_files']} files, {new_stats['total_chunks']} chunks")
        print(f"   Added: {len(chunks)} new chunks from {len(files_to_process)} files")
        print(f"   Index saved to: {config.get_index_path()}")
        print()

    else:
        # Non-incremental mode (rebuild from scratch)
        print("Incremental indexing is disabled. Rebuilding entire index...")
        print()

        docs = load_docs()
        if not docs:
            print("No documents found in data/raw. Add PDFs or .txt files and re-run.")
            return

        splitter = make_text_splitter(config._config)
        chunks = splitter.split_documents(docs)
        assign_chunk_ids(chunks)
        print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
        print()

        build_faiss(chunks, config.get_index_path())
        print(f"FAISS index saved to: {config.get_index_path()}")
        print()

if __name__ == "__main__":
    main()
