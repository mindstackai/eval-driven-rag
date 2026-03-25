"""
Synthetic query generation for embedding fine-tuning.

For each selected chunk, calls the Claude API to generate questions that the
chunk directly and specifically answers. Saves incrementally so runs are
resume-safe — chunks already in the output file are skipped.

Usage:
    from src.training.generate_pairs import generate_pairs
    pairs = generate_pairs(chunks, config)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path("data/training_pairs.jsonl")


def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _load_existing_pair_ids(path: Path) -> set[str]:
    """Return set of pair_ids already saved (for idempotent resume)."""
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["pair_id"])
    return ids


def _select_diverse_chunks(chunks: list[str], embedder, coverage_ratio: float) -> list[int]:
    """
    Select a diverse subset of chunk indices using k-means on embeddings.
    Picks coverage_ratio * len(chunks) chunks, one representative per cluster.
    """
    from sklearn.cluster import KMeans

    n_total = len(chunks)
    n_select = max(1, int(n_total * coverage_ratio))

    # Encode all chunks
    vecs = embedder.embed(chunks)

    # K-means: one cluster per desired chunk
    n_clusters = min(n_select, n_total)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(vecs)

    # For each cluster, pick the chunk closest to the centroid
    selected = []
    for c in range(n_clusters):
        cluster_indices = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(vecs[cluster_indices] - centroid, axis=1)
        closest = cluster_indices[np.argmin(dists)]
        selected.append(int(closest))

    return selected


def _generate_queries_for_chunk(chunk_text: str, n_queries: int, client) -> list[str]:
    """Call Claude API to generate n_queries questions answered by chunk_text."""
    prompt = f"""You are creating training data for a retrieval model.

Given the passage below, generate exactly {n_queries} questions that this passage directly and specifically answers.

Rules:
- Each question must be answerable ONLY from this passage, not from general knowledge
- Questions should vary in phrasing and focus on different facts in the passage
- Do NOT generate general topic questions — be specific to the content
- Output one question per line, no numbering or bullets

Passage:
{chunk_text}

Questions:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    queries = [q.strip() for q in raw.splitlines() if q.strip()]
    return queries[:n_queries]


def generate_pairs(chunks: list[str], config: dict) -> list[dict]:
    """
    Generate synthetic (query, positive chunk) pairs from a corpus.

    Args:
        chunks: List of chunk texts.
        config: Parsed config.yaml dict.

    Returns:
        List of pair dicts (all pairs including previously saved ones).
    """
    from anthropic import Anthropic
    from dotenv import load_dotenv
    from src.embedders import load_embedder

    load_dotenv()

    training_cfg = config.get("training", {})
    n_queries = training_cfg.get("queries_per_chunk", 3)
    coverage_ratio = training_cfg.get("coverage_ratio", 0.6)

    # Use first local model for cluster-based selection
    local_models = [m for m in config["embedding"]["models"] if m["source"] == "local"]
    embedder = load_embedder(local_models[0]["name"], config)

    client = Anthropic()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_pair_ids(OUTPUT_PATH)
    print(f"Resuming: {len(existing_ids)} pairs already saved.")

    # Select diverse chunks
    print(f"Selecting {coverage_ratio*100:.0f}% of {len(chunks)} chunks via k-means...")
    selected_indices = _select_diverse_chunks(chunks, embedder, coverage_ratio)
    print(f"Selected {len(selected_indices)} chunks.")

    new_pairs = []
    with open(OUTPUT_PATH, "a") as out_f:
        for i, chunk_idx in enumerate(selected_indices):
            chunk_text = chunks[chunk_idx]
            pid = _chunk_id(chunk_text)

            if pid in existing_ids:
                continue  # already processed — skip

            queries = _generate_queries_for_chunk(chunk_text, n_queries, client)

            for query in queries:
                pair = {
                    "pair_id": pid,
                    "chunk_id": chunk_idx,
                    "query": query,
                    "positive": chunk_text,
                    "positive_id": chunk_idx,
                }
                out_f.write(json.dumps(pair) + "\n")
                new_pairs.append(pair)

            existing_ids.add(pid)
            total = len(existing_ids) * n_queries
            print(f"  [{i+1}/{len(selected_indices)}] chunk {chunk_idx} → {len(queries)} queries (total pairs: {total})")

    # Return all pairs (existing + new)
    all_pairs = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    print(f"\nDone. {len(all_pairs)} total pairs saved to {OUTPUT_PATH}")
    return all_pairs
