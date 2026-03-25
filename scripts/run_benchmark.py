"""
Benchmark runner — evaluates all embedding models from config.yaml.

For each model (base + fine-tuned) it:
  1. Instantiates the correct embedder (HF or OpenAI)
  2. Runs Phase 1 retrieval eval with the default chunking config
  3. Logs results to traces/experiment_log.jsonl

Usage:
    python -m scripts.run_benchmark
    python -m scripts.run_benchmark --config config.yaml --eval_queries data/eval/eval_queries.jsonl
    python -m scripts.run_benchmark --skip_finetuned   # base models only
    python -m scripts.run_benchmark --model BAAI/bge-small-en-v1.5  # single model

Output:
    Final comparison table sorted by MRR:
    model | source | finetuned | hit@K | mrr | recall@K | index_time | cost/1k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.embedders import load_embedder
from src.eval.eval_trace import run_phase1_eval, load_experiments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_queries(path: str) -> list[dict]:
    """
    Load eval queries from a JSONL file.

    Each line must be:
      {"question": "...", "relevant_chunk_ids": [0, 3, 7]}

    relevant_chunk_ids are 0-based indices into the corpus chunks list.
    """
    if not os.path.exists(path):
        print(f"  [warn] eval queries file not found: {path}")
        return []
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _load_corpus_chunks(config: dict) -> list[str]:
    """
    Load plain-text chunks from data/training_pairs.jsonl (the 'positive' field),
    falling back to data/triplets_clean.jsonl or an empty list with a warning.

    In a real pipeline these chunks come from your document splitter.
    For the benchmark runner we reuse the positives already on disk so we
    don't have to re-ingest and re-split documents.
    """
    candidates = [
        "data/triplets_clean.jsonl",
        "data/training_pairs.jsonl",
    ]
    for path in candidates:
        if os.path.exists(path):
            texts: list[str] = []
            seen: set[str] = set()
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("positive", "")
                    if text and text not in seen:
                        seen.add(text)
                        texts.append(text)
            if texts:
                print(f"  Loaded {len(texts)} unique corpus chunks from {path}")
                return texts

    print("  [warn] No corpus chunks found. Benchmark will use a tiny dummy corpus.")
    return [
        "This is a placeholder corpus chunk for testing purposes.",
        "The benchmark runner needs real corpus chunks to produce meaningful scores.",
        "Populate data/training_pairs.jsonl or data/triplets_clean.jsonl first.",
    ]


def _is_finetuned(model_cfg: dict) -> bool:
    return "finetuned" in model_cfg.get("name", "").lower() or model_cfg.get("base_model") is not None


def _extract_family(model_name: str) -> str:
    name = model_name.lower()
    if "bge-small" in name:
        return "bge-small"
    if "bge-base" in name:
        return "bge-base"
    if "minilm" in name or "all-minilm" in name:
        return "minilm"
    if "text-embedding-3-small" in name:
        return "openai-small"
    if "text-embedding-3-large" in name:
        return "openai-large"
    return name.split("/")[-1][:16]


def _print_progress_table(results: list[dict], top_k: int) -> None:
    """Print a live progress table after each model completes."""
    hit_col = f"hit_rate@{top_k}"
    recall_col = f"recall@{top_k}"

    header = (
        f"  {'Model':<40} {'Src':<7} {'FT':<4} "
        f"{'Hit@' + str(top_k):<9} {'MRR':<9} {'Rec@' + str(top_k):<9} "
        f"{'Idx(s)':<8} {'Cost/1k':<10}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print()
    print(header)
    print(sep)
    for r in results:
        model = r["model"][-38:] if len(r["model"]) > 38 else r["model"]
        ft = "✓" if r["finetuned"] else " "
        cost = f"${r['cost_per_1k']:.5f}" if r["cost_per_1k"] > 0 else "$0"
        print(
            f"  {model:<40} {r['source']:<7} {ft:<4} "
            f"{r.get(hit_col, 0):<9.4f} {r['mrr']:<9.4f} {r.get(recall_col, 0):<9.4f} "
            f"{r['index_time_s']:<8.2f} {cost:<10}"
        )


def _print_final_table(results: list[dict], top_k: int) -> None:
    """Print final comparison table sorted by MRR."""
    results_sorted = sorted(results, key=lambda r: r["mrr"], reverse=True)
    hit_col = f"hit_rate@{top_k}"
    recall_col = f"recall@{top_k}"

    width = 100
    print()
    print("=" * width)
    print(f" Benchmark Results — sorted by MRR  (top_k={top_k})")
    print("=" * width)
    header = (
        f"  {'Rank':<5} {'Model':<40} {'Src':<7} {'FT':<4} "
        f"{'Hit@' + str(top_k):<9} {'MRR':<9} {'Rec@' + str(top_k):<9} "
        f"{'Idx(s)':<8} {'Cost/1k'}"
    )
    print(header)
    print("  " + "-" * (width - 2))

    for rank, r in enumerate(results_sorted, 1):
        model = r["model"][-38:] if len(r["model"]) > 38 else r["model"]
        ft = "✓" if r["finetuned"] else " "
        cost = f"${r['cost_per_1k']:.5f}" if r["cost_per_1k"] > 0 else "$0.00000"
        print(
            f"  {rank:<5} {model:<40} {r['source']:<7} {ft:<4} "
            f"{r.get(hit_col, 0):<9.4f} {r['mrr']:<9.4f} {r.get(recall_col, 0):<9.4f} "
            f"{r['index_time_s']:<8.2f} {cost}"
        )

    print("=" * width)

    # Highlight winner
    winner = results_sorted[0]
    print(f"\n  🏆  Best MRR: {winner['model']}  ({winner['mrr']:.4f})")
    if winner["finetuned"]:
        # Find base counterpart
        family = _extract_family(winner["model"])
        base = next(
            (r for r in results_sorted if not r["finetuned"] and _extract_family(r["model"]) == family),
            None,
        )
        if base:
            delta = winner["mrr"] - base["mrr"]
            print(f"      Fine-tuned lift over base: {delta:+.4f} MRR")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run retrieval benchmark across all embedding models.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--eval_queries",
        default="data/eval/eval_queries.jsonl",
        help="Path to eval queries JSONL (question + relevant_chunk_ids)",
    )
    p.add_argument("--top_k", type=int, default=None, help="Override top-K (default from config)")
    p.add_argument(
        "--skip_finetuned",
        action="store_true",
        help="Evaluate base models only (skip finetuned_models in config)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluate a single model by name (must match config.yaml exactly)",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional note appended to every experiment log entry in this run",
    )
    return p.parse_args()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    config = _load_config(args.config)

    embedding_cfg = config.get("embedding", {})
    eval_cfg = config.get("eval", {})
    chunking_cfg = config.get("chunking", {})

    top_k: int = args.top_k or eval_cfg.get("top_k", 5)
    chunking_strategy: str = chunking_cfg.get("default_strategy", "recursive")
    chunk_size: int = chunking_cfg.get("default_chunk_size", 512)

    # Collect all models to benchmark
    base_models: list[dict] = embedding_cfg.get("models", [])
    finetuned_models: list[dict] = [] if args.skip_finetuned else embedding_cfg.get("finetuned_models", [])
    all_model_cfgs: list[dict] = base_models + finetuned_models

    # Filter to single model if requested
    if args.model:
        all_model_cfgs = [m for m in all_model_cfgs if m["name"] == args.model]
        if not all_model_cfgs:
            print(f"Error: model '{args.model}' not found in config.yaml", file=sys.stderr)
            sys.exit(1)

    print(f"Benchmark: {len(all_model_cfgs)} model(s)  top_k={top_k}  strategy={chunking_strategy}  chunk_size={chunk_size}")
    print()

    # Load shared data (same corpus + eval queries for all models)
    print("Loading corpus chunks ...")
    corpus_chunks = _load_corpus_chunks(config)

    print("Loading eval queries ...")
    eval_queries = _load_eval_queries(args.eval_queries)
    if not eval_queries:
        print("  [warn] No eval queries found — all metrics will be 0.0")
        print(f"  Create {args.eval_queries} with format:")
        print('  {"question": "...", "relevant_chunk_ids": [0, 3, 7]}')

    print()

    results: list[dict] = []
    total = len(all_model_cfgs)

    for i, model_cfg in enumerate(all_model_cfgs, 1):
        model_name: str = model_cfg["name"]
        is_ft = _is_finetuned(model_cfg)

        # Skip fine-tuned models whose checkpoint directory doesn't exist
        if is_ft and not os.path.exists(model_name):
            print(f"[{i}/{total}] SKIP  {model_name}  (checkpoint not found at '{model_name}')")
            continue

        print(f"[{i}/{total}] {model_name}")
        print(f"  source={model_cfg.get('source', 'local')}  dim={model_cfg.get('dim')}  finetuned={is_ft}")

        try:
            embedder = load_embedder(model_name, config)
            metrics = run_phase1_eval(
                embedder=embedder,
                chunks=corpus_chunks,
                eval_queries=eval_queries,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                top_k=top_k,
                notes=args.notes,
            )
        except Exception as exc:
            print(f"  [error] {exc}")
            continue

        hit_col = f"hit_rate@{top_k}"
        recall_col = f"recall@{top_k}"

        result_row = {
            "model": model_name,
            "source": model_cfg.get("source", "local"),
            "finetuned": is_ft,
            hit_col: metrics.get(hit_col, 0.0),
            "mrr": metrics.get("mrr", 0.0),
            recall_col: metrics.get(recall_col, 0.0),
            "index_time_s": metrics.get("index_time_s", 0.0),
            "cost_per_1k": model_cfg.get("cost_per_1k_tokens", 0.0),
        }
        results.append(result_row)

        print(
            f"  hit@{top_k}={result_row[hit_col]:.4f}  "
            f"mrr={result_row['mrr']:.4f}  "
            f"recall@{top_k}={result_row[recall_col]:.4f}  "
            f"index_time={result_row['index_time_s']:.2f}s"
        )

        # Live progress table after each model
        _print_progress_table(results, top_k)

    # Final sorted comparison
    if results:
        _print_final_table(results, top_k)
        print(f"All runs logged to traces/experiment_log.jsonl")
        print(f"Run the dashboard to explore: bash run_dashboard.sh")
    else:
        print("No models were evaluated.")


if __name__ == "__main__":
    main()
