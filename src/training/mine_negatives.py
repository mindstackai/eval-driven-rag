"""
Hard negative mining for embedding fine-tuning.

For each (query, positive) pair, retrieves the top-30 most similar chunks
using LanceDB and assigns them to difficulty tiers based on cosine similarity.
Uses the SAME embedder that will be used for fine-tuning — never cross-mix.

Usage:
    from src.training.mine_negatives import mine_hard_negatives
    triplets = mine_hard_negatives(pairs, embedder, chunks, config)
"""

import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa

OUTPUT_PATH = Path("data/triplets.jsonl")
LANCEDB_PATH = "./vectorstore/lancedb"
TOP_RETRIEVE = 30


def _table_name(model_name: str) -> str:
    return "corpus_" + re.sub(r"[^a-z0-9_]", "_", model_name.lower())


def _get_or_create_corpus_table(db, table_name: str, embedder, chunks: list[str]):
    """Return existing corpus table or build one by encoding all chunks."""
    if table_name in db.table_names():
        print(f"  Reusing existing LanceDB table '{table_name}'.")
        return db.open_table(table_name)

    print(f"  Encoding {len(chunks)} chunks for table '{table_name}'...")
    vecs = embedder.embed(chunks)

    schema = pa.schema([
        pa.field("chunk_id", pa.int32()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), embedder.embedding_dim)),
    ])
    rows = [
        {"chunk_id": i, "text": text, "vector": vecs[i].tolist()}
        for i, text in enumerate(chunks)
    ]
    table = db.create_table(table_name, data=rows, schema=schema, mode="overwrite")
    print(f"  Table created with {len(rows)} rows.")
    return table


def _tier(sim: float, cfg: dict) -> str:
    very_hard = cfg.get("very_hard_threshold", 0.75)
    hard = cfg.get("hard_threshold", 0.55)
    medium = cfg.get("medium_threshold", 0.35)

    if sim > very_hard:
        return "very_hard"
    if sim > hard:
        return "hard"
    if sim > medium:
        return "medium"
    return "easy"


def _pick_negatives(candidates: list[dict], tier_cfg: dict) -> tuple[list[str], list[float], list[str]]:
    """
    Pick up to 3 negatives — one from each tier (very_hard, hard, medium).
    Falls back to top retrieved if a tier is empty.
    """
    by_tier: dict[str, list[dict]] = {"very_hard": [], "hard": [], "medium": [], "easy": []}
    for c in candidates:
        by_tier[c["tier"]].append(c)

    chosen = []
    for tier in ("very_hard", "hard", "medium"):
        if by_tier[tier]:
            chosen.append(by_tier[tier][0])
        elif candidates and len(chosen) < 3:
            # fallback: pick next best not already chosen
            for c in candidates:
                if c not in chosen:
                    chosen.append(c)
                    break

    texts = [c["text"] for c in chosen]
    scores = [round(c["sim"], 4) for c in chosen]
    tiers = [c["tier"] for c in chosen]
    return texts, scores, tiers


def mine_hard_negatives(
    pairs: list[dict],
    embedder,
    chunks: list[str],
    config: dict,
) -> list[dict]:
    """
    Mine hard negatives for each (query, positive) pair.

    Args:
        pairs: Output of generate_pairs — list of pair dicts.
        embedder: The BaseEmbedder to use (must match the model being fine-tuned).
        chunks: Full corpus chunk texts (same order as used for pair generation).
        config: Parsed config.yaml dict.

    Returns:
        List of triplet dicts written to data/triplets.jsonl.
    """
    import lancedb

    tier_cfg = config.get("training", {}).get("hard_negative_tiers", {})

    db = lancedb.connect(LANCEDB_PATH)
    tname = _table_name(embedder.model_name)
    table = _get_or_create_corpus_table(db, tname, embedder, chunks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    triplets = []

    with open(OUTPUT_PATH, "w") as out_f:
        for i, pair in enumerate(pairs):
            query = pair["query"]
            positive_id = pair["positive_id"]

            # Encode query
            q_vec = embedder.embed([query])[0]

            # Retrieve top-30 candidates
            results = table.search(q_vec).metric("cosine").limit(TOP_RETRIEVE).to_list()

            # Build candidates excluding the positive chunk (match by text, not ID)
            positive_text = pair["positive"]
            candidates = []
            for r in results:
                if r["text"] == positive_text:
                    continue
                sim = 1.0 - r["_distance"]
                candidates.append({
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "sim": sim,
                    "tier": _tier(sim, tier_cfg),
                })

            neg_texts, neg_scores, neg_tiers = _pick_negatives(candidates, tier_cfg)

            triplet = {
                **pair,
                "negatives": neg_texts,
                "negative_scores": neg_scores,
                "negative_tiers": neg_tiers,
                "embedder_used": embedder.model_name,
            }
            out_f.write(json.dumps(triplet) + "\n")
            triplets.append(triplet)

            if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
                tier_counts = {}
                for t in neg_tiers:
                    tier_counts[t] = tier_counts.get(t, 0) + 1
                print(f"  [{i+1}/{len(pairs)}] tiers: {tier_counts}")

    print(f"\nDone. {len(triplets)} triplets saved to {OUTPUT_PATH}")
    return triplets
