"""
Smoke test for all three local HF embedding models.

Loads models from config.yaml, encodes a small test corpus via LanceDB,
runs 3 test queries, and verifies similarity scores are in [0, 1].
Cleans up temp tables after each run.

Usage:
    cd eval-driven-rag
    python scripts/smoke_test_embedders.py
"""

import re
import sys
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedders import load_embedder

TEST_CORPUS = [
    "Retrieval-augmented generation combines dense retrieval with language model generation.",
    "LanceDB is a serverless vector database built on top of Apache Arrow and Lance format.",
    "Sentence transformers produce fixed-size embeddings for variable-length text.",
    "Hard negative mining improves contrastive learning by selecting challenging examples.",
    "The BGE family of embedding models is trained by BAAI with strong MTEB performance.",
    "MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon.",
    "Cosine similarity measures the angle between two vectors in embedding space.",
    "Chunking strategy affects both retrieval precision and the quality of generated answers.",
    "Fine-tuning on domain-specific pairs can significantly improve retrieval hit rate.",
    "The MultipleNegativesRankingLoss treats in-batch examples as implicit negatives.",
]

TEST_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does LanceDB store vector embeddings?",
    "How can you improve embedding model quality on a specific domain?",
]

LANCEDB_PATH = "./vectorstore/lancedb"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def run_smoke_test(embedder, db) -> bool:
    table_name = f"smoke_test_{slugify(embedder.model_name)}"

    print(f"\n{'='*60}")
    print(f"Model : {embedder.model_name}")
    print(f"Device: {getattr(embedder, 'device', 'n/a')}")
    print(f"Dim   : {embedder.embedding_dim}")

    # --- Encode corpus ---
    t0 = time.perf_counter()
    corpus_vecs = embedder.embed(TEST_CORPUS)
    encode_time = time.perf_counter() - t0

    assert corpus_vecs.shape == (len(TEST_CORPUS), embedder.embedding_dim), (
        f"Unexpected shape: {corpus_vecs.shape}"
    )
    print(f"Encode time: {encode_time:.3f}s for {len(TEST_CORPUS)} chunks")

    # --- Write to temp LanceDB table ---
    schema = pa.schema([
        pa.field("chunk_id", pa.int32()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), embedder.embedding_dim)),
    ])
    rows = [
        {"chunk_id": i, "text": text, "vector": corpus_vecs[i].tolist()}
        for i, text in enumerate(TEST_CORPUS)
    ]
    table = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")

    # --- Run queries ---
    all_ok = True
    for query in TEST_QUERIES:
        q_vec = embedder.embed([query])[0]
        results = table.search(q_vec).metric("cosine").limit(3).to_list()

        print(f"\n  Query: {query!r}")
        for r in results:
            sim = 1.0 - r["_distance"]  # cosine metric: sim = 1 - distance
            if not (0.0 <= sim <= 1.0):
                print(f"    [FAIL] similarity out of range: {sim:.4f}")
                all_ok = False
            else:
                print(f"    sim={sim:.4f}  {r['text'][:70]!r}")

    # --- Cleanup ---
    db.drop_table(table_name)
    print(f"\n  Temp table '{table_name}' dropped.")
    return all_ok


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    local_models = [m for m in config["embedding"]["models"] if m["source"] == "local"]
    print(f"Found {len(local_models)} local HF models to test.")

    db = lancedb.connect(LANCEDB_PATH)

    results = {}
    for model_cfg in local_models:
        embedder = load_embedder(model_cfg["name"], config)
        ok = run_smoke_test(embedder, db)
        results[model_cfg["name"]] = ok

    print(f"\n{'='*60}")
    print("SUMMARY")
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_passed = False

    if not all_passed:
        sys.exit(1)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
