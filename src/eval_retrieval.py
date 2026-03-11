"""
Phase 1: Retrieval Evaluation — no LLM calls, local only.

Tests multiple chunk sizes, re-ingests docs for each, runs retrieval,
and calculates Hit Rate and MRR per chunk size.

Since chunk IDs change with different chunk sizes, this module uses
content-based matching: a retrieved chunk is considered a "hit" if it
shares significant text overlap with any expected chunk.

Usage:
    python -m src.eval_retrieval
    python -m src.eval_retrieval --ground_truth eval/ground_truth.json
"""
import argparse
import json
import os
import sys
from datetime import datetime

import yaml

from metrics.overlap import text_overlap_ratio, is_content_match, content_reciprocal_rank

from src.ingest import load_docs, assign_chunk_ids
from src.splitters import make_text_splitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Retrieval eval across multiple chunk sizes."
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="eval/ground_truth.json",
        help="Path to the ground truth JSON file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config.yaml file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Short description/goal for this eval run (e.g. 'baseline recursive strategy').",
    )
    return parser.parse_args()


def _build_reference_content_map(docs, config):
    """Build chunks at reference config and return {chunk_id: page_content}."""
    splitter = make_text_splitter(config)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)
    return {c.metadata["chunk_id"]: c.page_content for c in chunks}


def _build_temp_index(docs, chunk_size, chunk_overlap, chunking_strategy, embeddings):
    """Build a temporary FAISS index with the given chunking config."""
    from langchain_community.vectorstores import FAISS

    cfg = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
    }
    splitter = make_text_splitter(cfg)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)

    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks


def _run_retrieval_for_config(vs, ground_truth, k, ref_content_map):
    """Run retrieval eval using content-based matching."""
    retriever = vs.as_retriever(search_kwargs={"k": k})

    per_question = []
    hit_rates = []
    mrr_scores = []
    recall_scores = []

    for item in ground_truth:
        question = item["question"]
        expected_ids = item.get("expected_chunk_ids", [])

        # Look up expected chunk content from reference map
        expected_texts = [
            ref_content_map[cid] for cid in expected_ids if cid in ref_content_map
        ]

        docs = retriever.invoke(question)
        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]

        retrieved_texts = [d.page_content for d in docs]

        # Content-based hit: any retrieved chunk overlaps with any expected chunk
        hit = 1.0 if any(
            is_content_match(d.page_content, expected_texts) for d in docs
        ) else 0.0

        mrr = content_reciprocal_rank(retrieved_texts, expected_texts)

        # Content-based recall: fraction of expected chunks "covered" by retrieved
        if expected_texts:
            covered = sum(
                1 for exp_text in expected_texts
                if any(text_overlap_ratio(d.page_content, exp_text) >= 0.5 for d in docs)
            )
            r_at_k = covered / len(expected_texts)
        else:
            r_at_k = 0.0

        hit_rates.append(hit)
        mrr_scores.append(mrr)
        recall_scores.append(r_at_k)

        per_question.append({
            "question": question,
            "expected_chunk_ids": expected_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "hit": hit,
            "mrr": mrr,
            "recall_at_k": r_at_k,
        })

    n = len(ground_truth)
    return {
        "hit_rate": sum(hit_rates) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
        "recall_at_k": sum(recall_scores) / n if n else 0.0,
        "per_question": per_question,
    }


def main() -> None:
    args = parse_args()

    # Load ground truth
    try:
        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(ground_truth)} questions from {args.ground_truth}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    chunk_sizes = eval_cfg.get("chunk_sizes_to_test", [128, 256, 512, 1024])
    chunk_overlap = config.get("chunk_overlap", 120)
    chunking_strategy = config.get("chunking_strategy", "recursive")
    top_k = config.get("top_k", 4)
    results_dir = eval_cfg.get("results_dir", "eval/results")

    os.makedirs(results_dir, exist_ok=True)

    # Load embeddings once
    from dotenv import load_dotenv
    load_dotenv()
    from src.config_manager import get_config
    cfg_obj = get_config(args.config)
    embeddings = cfg_obj.get_embeddings()

    # Load raw docs once
    print("Loading documents from data/raw/ ...")
    docs = load_docs()
    if not docs:
        print("Error: No documents found in data/raw/", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(docs)} document pages")

    # Build reference content map from default config (to resolve expected_chunk_ids)
    print("Building reference chunks for content matching ...")
    ref_content_map = _build_reference_content_map(docs, config)
    print(f"Reference index: {len(ref_content_map)} chunks at chunk_size={config.get('chunk_size', 800)}")
    print()

    # Run retrieval eval for each chunk size
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine run name
    run_name = args.name
    if not run_name:
        sizes_str = "_".join(str(s) for s in chunk_sizes)
        run_name = f"{chunking_strategy}_{sizes_str}_overlap{chunk_overlap}"

    # Build display name with formatted timestamp
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"
    print(f"Run: {display_name}")
    print()

    for chunk_size in chunk_sizes:
        print(f"--- Chunk size: {chunk_size} ---")
        print(f"  Building index (strategy={chunking_strategy}, overlap={chunk_overlap}) ...")

        vs, chunks = _build_temp_index(
            docs, chunk_size, chunk_overlap, chunking_strategy, embeddings
        )
        print(f"  {len(chunks)} chunks indexed")

        metrics = _run_retrieval_for_config(vs, ground_truth, top_k, ref_content_map)
        metrics["chunk_size"] = chunk_size
        metrics["chunk_overlap"] = chunk_overlap
        metrics["chunking_strategy"] = chunking_strategy
        metrics["num_chunks"] = len(chunks)
        metrics["top_k"] = top_k
        metrics["timestamp"] = timestamp
        metrics["run_name"] = run_name
        metrics["display_name"] = display_name

        all_results.append(metrics)

        # Save per-config results
        result_path = os.path.join(results_dir, f"retrieval_{chunk_size}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Hit Rate: {metrics['hit_rate']:.4f}  |  MRR: {metrics['mrr']:.4f}  |  Recall@{top_k}: {metrics['recall_at_k']:.4f}")
        print(f"  Saved: {result_path}")
        print()

    # Print summary table ranked by hit rate
    all_results.sort(key=lambda r: r["hit_rate"], reverse=True)

    width = 72
    print("=" * width)
    print(" Phase 1: Retrieval Eval Summary (ranked by Hit Rate)")
    print("=" * width)
    print(f"  {'Rank':<6} {'Chunk Size':<12} {'Chunks':<8} {'Hit Rate':<10} {'MRR':<10} {'Recall@' + str(top_k):<10}")
    print("-" * width)
    for rank, r in enumerate(all_results, 1):
        print(
            f"  {rank:<6} {r['chunk_size']:<12} {r['num_chunks']:<8} "
            f"{r['hit_rate']:<10.4f} {r['mrr']:<10.4f} {r['recall_at_k']:<10.4f}"
        )
    print("=" * width)
    print()
    print(f"Results saved to {results_dir}/")
    print("Run Phase 2 with: python -m src.eval_generation")


if __name__ == "__main__":
    main()
