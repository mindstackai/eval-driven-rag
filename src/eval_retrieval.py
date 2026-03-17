"""
Phase 1: Retrieval Evaluation — no LLM calls, local only.

Tests multiple chunk sizes, re-ingests docs for each, runs retrieval,
and calculates Hit Rate and MRR per chunk size.

Usage:
    python -m src.eval_retrieval
    python -m src.eval_retrieval --ground_truth eval/ground_truth.json
"""
import argparse
import json
import lancedb
import os
import sys
from datetime import datetime

import yaml

from metrics.overlap import text_overlap_ratio, is_content_match, content_reciprocal_rank

from src.ingest import load_docs, assign_chunk_ids
from src.splitters import make_text_splitter
from src.vectorstore.lancedb_store import LanceDBStore
from src.retriever import LanceDBRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Retrieval eval across multiple chunk sizes."
    )
    parser.add_argument("--ground_truth", type=str, default="eval/ground_truth.json")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--apply-winner",
        action="store_true",
        help="Automatically update config.yaml with the winning chunk size.",
    )
    return parser.parse_args()


def _build_reference_content_map(docs, config):
    splitter = make_text_splitter(config)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)
    return {c.metadata["chunk_id"]: c.page_content for c in chunks}


def _build_and_save_index(docs, chunk_size, chunk_overlap, chunking_strategy, embeddings, base_dir="vectorstore"):
    """Build a LanceDB index for one chunk-size config and save it to disk."""
    cfg = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunking_strategy": chunking_strategy,
    }
    splitter = make_text_splitter(cfg)
    chunks = splitter.split_documents(docs)
    assign_chunk_ids(chunks)

    for chunk in chunks:
        chunk.metadata.setdefault("allowed_roles", "public")

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)
    metadatas = [c.metadata for c in chunks]

    db_path = os.path.join(base_dir, f"lancedb_eval_{chunk_size}")
    table_name = "eval_chunks"

    # Drop and recreate for a clean eval run
    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        db.drop_table(table_name)

    store = LanceDBStore(db_path=db_path, table_name=table_name)
    store.add_documents(texts, vectors, metadatas)

    return store, chunks, db_path, table_name


def _run_retrieval_for_config(retriever, ground_truth, k, ref_content_map):
    per_question = []
    hit_rates, mrr_scores, recall_scores = [], [], []

    for item in ground_truth:
        question = item["question"]
        expected_ids = item.get("expected_chunk_ids", [])

        expected_texts = [
            ref_content_map[cid] for cid in expected_ids if cid in ref_content_map
        ]

        results_with_scores = retriever.similarity_search_with_relevance_scores(question, k=k)
        docs = [doc for doc, _ in results_with_scores]
        scores = [round(float(score), 4) for _, score in results_with_scores]
        retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]
        retrieved_texts = [d.page_content for d in docs]

        hit = 1.0 if any(is_content_match(d.page_content, expected_texts) for d in docs) else 0.0
        mrr = content_reciprocal_rank(retrieved_texts, expected_texts)

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
            "retrieval_scores": scores,
            "mean_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        })

    n = len(ground_truth)
    return {
        "hit_rate": sum(hit_rates) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
        "recall_at_k": sum(recall_scores) / n if n else 0.0,
        "avg_retrieval_confidence": sum(q["mean_retrieval_score"] for q in per_question) / n if n else 0.0,
        "per_question": per_question,
    }


def main() -> None:
    args = parse_args()

    try:
        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(ground_truth)} questions from {args.ground_truth}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    chunk_sizes = eval_cfg.get("chunk_sizes_to_test", [128, 256, 512, 1024])
    chunk_overlap = config.get("chunk_overlap", 120)
    chunking_strategy = config.get("chunking_strategy", "recursive")
    top_k = config.get("top_k", 4)
    results_dir = eval_cfg.get("results_dir", "eval/results")

    os.makedirs(results_dir, exist_ok=True)

    from dotenv import load_dotenv
    load_dotenv()
    from src.config_manager import get_config
    cfg_obj = get_config(args.config)
    embeddings = cfg_obj.get_embeddings()

    print("Loading documents from data/raw/ ...")
    docs = load_docs()
    if not docs:
        print("Error: No documents found in data/raw/", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(docs)} document pages")

    print("Building reference chunks for content matching ...")
    ref_content_map = _build_reference_content_map(docs, config)
    print(f"Reference index: {len(ref_content_map)} chunks at chunk_size={config.get('chunk_size', 800)}")
    print()

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = args.name
    if not run_name:
        sizes_str = "_".join(str(s) for s in chunk_sizes)
        run_name = f"{chunking_strategy}_{sizes_str}_overlap{chunk_overlap}"

    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"
    print(f"Run: {display_name}")
    print()

    for chunk_size in chunk_sizes:
        print(f"--- Chunk size: {chunk_size} ---")
        print(f"  Building index (strategy={chunking_strategy}, overlap={chunk_overlap}) ...")

        store, chunks, lancedb_path, table_name = _build_and_save_index(
            docs, chunk_size, chunk_overlap, chunking_strategy, embeddings
        )
        retriever = LanceDBRetriever(store, embeddings, top_k)

        print(f"  {len(chunks)} chunks indexed")
        print(f"  Index saved to: {lancedb_path}")

        metrics = _run_retrieval_for_config(retriever, ground_truth, top_k, ref_content_map)
        metrics.update({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "num_chunks": len(chunks),
            "top_k": top_k,
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
            "lancedb_path": lancedb_path,
            "table_name": table_name,
        })

        all_results.append(metrics)

        result_path = os.path.join(results_dir, f"retrieval_{chunk_size}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Hit Rate: {metrics['hit_rate']:.4f}  |  MRR: {metrics['mrr']:.4f}  |  Recall@{top_k}: {metrics['recall_at_k']:.4f}")
        print(f"  Saved: {result_path}")
        print()

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

    winner = all_results[0]
    print(f"Winner: chunk_size={winner['chunk_size']} (Hit Rate: {winner['hit_rate']:.4f})")
    print(f"Index saved at: {winner['lancedb_path']}")
    print()

    if args.apply_winner:
        config["chunk_size"] = winner["chunk_size"]
        with open(args.config, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {args.config}: chunk_size → {winner['chunk_size']}")
    else:
        print("To use this config, update config.yaml:")
        print(f"  chunk_size: {winner['chunk_size']}")
        print("Or re-run with --apply-winner")
    print()
    print(f"Results saved to {results_dir}/")
    print("Run Phase 2 with: python -m src.eval_generation")


if __name__ == "__main__":
    main()
