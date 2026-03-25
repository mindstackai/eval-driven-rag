"""
False negative detection and train/eval split for fine-tuning data.

A "false negative" is a mined hard negative that is actually semantically
equivalent to the positive chunk — including it as a negative would teach
the model the wrong thing.

Usage:
    from src.training.false_negative_check import flag_false_negatives, split_train_eval
    clean = flag_false_negatives(triplets, embedder)
    train, eval = split_train_eval(clean)
"""

import json
import random
from pathlib import Path

import numpy as np

FLAGGED_PATH = Path("data/triplets_flagged.jsonl")
CLEAN_PATH = Path("data/triplets_clean.jsonl")
TRAIN_PATH = Path("data/train.jsonl")
EVAL_PATH = Path("data/eval.jsonl")
EVAL_QUERIES_PATH = Path("data/eval/eval_queries.jsonl")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def flag_false_negatives(
    triplets: list[dict],
    embedder,
    threshold: float = 0.92,
) -> list[dict]:
    """
    Flag triplets where any negative is suspiciously similar to the positive.

    Args:
        triplets: Output of mine_hard_negatives.
        embedder: BaseEmbedder used to encode positives and negatives.
        threshold: Cosine similarity above which a negative is flagged (default 0.92).

    Returns:
        Clean triplets (false negatives removed). Also writes:
            data/triplets_flagged.jsonl — flagged for manual review
            data/triplets_clean.jsonl   — input to fine-tuning
    """
    for path in (FLAGGED_PATH, CLEAN_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    flagged = []
    clean = []

    all_texts = []
    for t in triplets:
        all_texts.append(t["positive"])
        all_texts.extend(t["negatives"])

    # Encode all texts in one batch for efficiency
    print(f"Encoding {len(all_texts)} texts to check for false negatives...")
    all_vecs = embedder.embed(all_texts)

    idx = 0
    for triplet in triplets:
        pos_vec = all_vecs[idx]
        idx += 1
        neg_vecs = all_vecs[idx : idx + len(triplet["negatives"])]
        idx += len(triplet["negatives"])

        has_risk = False
        for neg_vec in neg_vecs:
            sim = _cosine_sim(pos_vec.tolist(), neg_vec.tolist())
            if sim > threshold:
                has_risk = True
                break

        triplet = {**triplet, "has_false_negative_risk": has_risk}
        if has_risk:
            flagged.append(triplet)
        else:
            clean.append(triplet)

    with open(FLAGGED_PATH, "w") as f:
        for t in flagged:
            f.write(json.dumps(t) + "\n")

    with open(CLEAN_PATH, "w") as f:
        for t in clean:
            f.write(json.dumps(t) + "\n")

    print(f"Flagged: {len(flagged)} / {len(triplets)} triplets (threshold={threshold})")
    print(f"Clean:   {len(clean)} triplets → {CLEAN_PATH}")
    print(f"Flagged: {len(flagged)} triplets → {FLAGGED_PATH} (review manually)")

    return clean


def split_train_eval(
    triplets: list[dict],
    eval_split: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split triplets into train and eval sets, stratified by chunk_id so the
    same source chunk never appears in both splits.

    Args:
        triplets: Clean triplets (output of flag_false_negatives).
        eval_split: Fraction to reserve for eval (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        (train_triplets, eval_triplets)
    """
    for path in (TRAIN_PATH, EVAL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Group by chunk_id
    by_chunk: dict[int, list[dict]] = {}
    for t in triplets:
        cid = t["chunk_id"]
        by_chunk.setdefault(cid, []).append(t)

    chunk_ids = list(by_chunk.keys())
    rng = random.Random(seed)
    rng.shuffle(chunk_ids)

    n_eval_chunks = max(1, int(len(chunk_ids) * eval_split))
    eval_chunk_ids = set(chunk_ids[:n_eval_chunks])

    train, eval_ = [], []
    for t in triplets:
        if t["chunk_id"] in eval_chunk_ids:
            eval_.append(t)
        else:
            train.append(t)

    with open(TRAIN_PATH, "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")

    with open(EVAL_PATH, "w") as f:
        for t in eval_:
            f.write(json.dumps(t) + "\n")

    # Build eval_queries.jsonl for the benchmark runner.
    # The benchmark loads corpus chunks from triplets_clean.jsonl as a list of
    # unique positives in order of first appearance. relevant_chunk_ids must be
    # 0-based positions into that list — NOT the original chunk_id values.
    seen: dict[str, int] = {}
    with open(CLEAN_PATH) as f:
        for line in f:
            t = json.loads(line)
            if t["positive"] not in seen:
                seen[t["positive"]] = len(seen)
    # Map chunk_id → corpus position via the positive text
    chunk_id_to_pos: dict[int, int] = {}
    with open(CLEAN_PATH) as f:
        for line in f:
            t = json.loads(line)
            cid = t["chunk_id"]
            if cid not in chunk_id_to_pos and t["positive"] in seen:
                chunk_id_to_pos[cid] = seen[t["positive"]]

    EVAL_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(EVAL_QUERIES_PATH, "w") as f:
        for t in eval_:
            pos = chunk_id_to_pos.get(t["chunk_id"])
            if pos is None:
                continue
            record = {"question": t["query"], "relevant_chunk_ids": [pos]}
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Split: {len(train)} train / {len(eval_)} eval")
    print(f"  Eval chunks: {len(eval_chunk_ids)} / {len(chunk_ids)} unique chunk_ids held out")
    print(f"  → {TRAIN_PATH}")
    print(f"  → {EVAL_PATH}")
    print(f"  → {EVAL_QUERIES_PATH} ({written} benchmark eval queries)")

    return train, eval_
