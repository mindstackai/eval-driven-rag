"""
update_training_data.py — One command to refresh all training data after
new documents are added to data/raw/.

Runs Steps 1-3 of the training pipeline in order:
    Step 1  generate_pairs       — Claude API, idempotent (skips existing chunks)
    Step 2  mine_hard_negatives  — LanceDB, per HF embedder
    Step 3  false_negative_check — flag + split → data/train.jsonl + data/eval.jsonl

Step 4 (fine-tuning) is intentionally NOT run automatically — it takes 2-4 hours
per model on M2. Run it manually when you're happy with the training data:
    python -m src.training.finetune

Usage:
    python -m scripts.update_training_data
    python -m scripts.update_training_data --skip_mining    # Step 1 only
    python -m scripts.update_training_data --dry_run        # show chunk count, no API calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refresh training data after new documents are added to data/raw/."
    )
    p.add_argument(
        "--skip_mining",
        action="store_true",
        help="Run Step 1 (generate_pairs) only — skip hard negative mining.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Load and count chunks only — no Claude API calls.",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return p.parse_args()


def _load_chunks(config: dict) -> list[str]:
    """Load raw docs from data/raw/ and split into chunks."""
    from src.ingest import load_docs, assign_chunk_ids
    from src.splitters import make_text_splitter

    print("Loading documents from data/raw/ ...")
    docs = load_docs()
    if not docs:
        print("  [error] No documents found in data/raw/. Add PDFs or .txt files first.")
        sys.exit(1)
    print(f"  {len(docs)} pages loaded from {len(set(d.metadata.get('source','?') for d in docs))} file(s)")

    splitter = make_text_splitter(config)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)
    texts = [c.page_content for c in chunks]
    print(f"  {len(texts)} chunks after splitting")
    return texts


def _existing_pair_count() -> int:
    path = Path("data/training_pairs.jsonl")
    if not path.exists():
        return 0
    return sum(1 for l in path.open() if l.strip())


def _existing_triplet_count() -> int:
    path = Path("data/triplets.jsonl")
    if not path.exists():
        return 0
    return sum(1 for l in path.open() if l.strip())


def main() -> None:
    from dotenv import load_dotenv
    import yaml

    load_dotenv()

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    t_start = time.perf_counter()

    print()
    print("=" * 60)
    print("  update_training_data — refreshing pipeline")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    chunks = _load_chunks(config)
    print()

    if args.dry_run:
        existing = _existing_pair_count()
        print(f"[dry_run] {len(chunks)} chunks in corpus  |  {existing} pairs already saved")
        print("[dry_run] No API calls made. Remove --dry_run to run for real.")
        return

    # ------------------------------------------------------------------
    # Step 1 — generate_pairs (idempotent)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 1 / 3 — Generating synthetic Q&A pairs (Claude API)")
    print("-" * 60)
    before = _existing_pair_count()

    from src.training.generate_pairs import generate_pairs
    pairs = generate_pairs(chunks, config)

    after = _existing_pair_count()
    print(f"  New pairs generated: {after - before}  |  Total: {after}")
    print()

    if args.skip_mining:
        print("--skip_mining set — stopping after Step 1.")
        print()
        _print_summary(t_start, after, None, None)
        return

    # ------------------------------------------------------------------
    # Step 2 — mine_hard_negatives (per HF embedder)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 2 / 3 — Mining hard negatives")
    print("-" * 60)

    from src.embedders import load_embedder
    from src.training.mine_negatives import mine_hard_negatives

    local_models = [
        m["name"]
        for m in config["embedding"]["models"]
        if m["source"] == "local"
    ]

    triplets = None
    for model_name in local_models:
        print(f"  [{model_name}]")
        embedder = load_embedder(model_name, config)
        triplets = mine_hard_negatives(pairs, embedder, chunks, config)
        print()

    triplet_count = _existing_triplet_count()
    print(f"  Triplets saved: {triplet_count}")
    print()

    # ------------------------------------------------------------------
    # Step 3 — false_negative_check + split
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Step 3 / 3 — Flagging false negatives + splitting train/eval")
    print("-" * 60)

    from src.training.false_negative_check import flag_false_negatives, split_train_eval

    # Use first local model for encoding
    embedder = load_embedder(local_models[0], config)

    if triplets is None:
        # Shouldn't happen, but load from disk as fallback
        triplets = [json.loads(l) for l in open("data/triplets.jsonl") if l.strip()]

    clean = flag_false_negatives(triplets, embedder)
    train, eval_ = split_train_eval(clean)
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(t_start, after, len(train), len(eval_))


def _print_summary(
    t_start: float,
    n_pairs: int,
    n_train: int | None,
    n_eval: int | None,
) -> None:
    elapsed = time.perf_counter() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print("=" * 60)
    print("  Done!")
    print(f"  Total pairs:       {n_pairs}")
    if n_train is not None:
        print(f"  Train set:         {n_train}  → data/train.jsonl")
    if n_eval is not None:
        print(f"  Eval set:          {n_eval}  → data/eval.jsonl")
    print(f"  Time elapsed:      {mins}m {secs}s")
    print()
    print("  Next step — fine-tune all 3 models (~2-4h per model on M2):")
    print("    python -m src.training.finetune")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
