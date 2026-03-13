"""
Phase 2: Generation Evaluation — runs LLM on top chunk configs from Phase 1.

Reads Phase 1 retrieval results, takes the top N configs by hit rate,
runs full RAG with LLM for each, caches responses, and evaluates
answer correctness.

Usage:
    python -m src.eval_generation
    python -m src.eval_generation --top_n 3
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml

from judge.adapters import LangChainJudgeClient
from judge.scorers import correctness_score, faithfulness_score, relevance_score

from src.ingest import load_docs, assign_chunk_ids
from src.splitters import make_text_splitter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Generation eval on top chunk configs from Phase 1."
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
        "--top_n",
        type=int,
        default=None,
        help="Override number of top configs to evaluate (default from config).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Short description/goal for this eval run (e.g. 'testing insurance docs').",
    )
    parser.add_argument(
        "--apply-winner",
        action="store_true",
        help="Automatically update config.yaml with the winning chunk size and index path.",
    )
    return parser.parse_args()


def _load_phase1_results(results_dir, ranking_metric="hit_rate"):
    """Load all Phase 1 retrieval results and return best per chunk_size.

    Args:
        results_dir: Directory containing Phase 1 result JSON files.
        ranking_metric: Metric to rank by (hit_rate, mrr, or recall_at_k).
    """
    pattern = os.path.join(results_dir, "retrieval_*.json")
    files = glob(pattern)

    if not files:
        return []

    # Group by chunk_size, keep latest timestamp for each
    by_chunk_size = {}
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
        cs = data.get("chunk_size")
        ts = data.get("timestamp", "")
        if cs not in by_chunk_size or ts > by_chunk_size[cs]["timestamp"]:
            by_chunk_size[cs] = data

    # Sort by ranking_metric descending
    results = sorted(by_chunk_size.values(), key=lambda r: r.get(ranking_metric, 0), reverse=True)
    return results


def _cache_key(question, retrieved_chunk_ids):
    """Generate deterministic cache key from question + retrieved chunks."""
    payload = json.dumps({"q": question, "chunks": sorted(retrieved_chunk_ids)}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _load_cache(cache_dir, key):
    """Load cached LLM response if it exists."""
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_cache(cache_dir, key, data):
    """Save LLM response to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_rag_answer(question, retriever, llm):
    """Run strict-RAG: retrieve context, generate answer with LLM.

    Returns (answer, retrieved_chunk_ids, context_chunks).
    """
    docs = retriever.invoke(question)
    retrieved_ids = [d.metadata.get("chunk_id", "") for d in docs]
    context_chunks = [d.page_content for d in docs]

    context_text = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""Answer the question using ONLY the provided context. If the context
does not contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content.strip()
    return answer, retrieved_ids, context_chunks


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    results_dir = eval_cfg.get("results_dir", "eval/results")
    cache_dir = eval_cfg.get("cache_dir", "eval/cache")
    cache_enabled = eval_cfg.get("cache_responses", True)
    top_n = args.top_n or eval_cfg.get("top_k_configs_for_generation", 2)
    ranking_metric = eval_cfg.get("phase1_ranking_metric", "hit_rate")

    # Load Phase 1 results
    phase1_results = _load_phase1_results(results_dir, ranking_metric)
    if not phase1_results:
        print("Error: No Phase 1 results found.", file=sys.stderr)
        print("Run Phase 1 first: python -m src.eval_retrieval", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(phase1_results)} chunk configs from Phase 1")
    top_configs = phase1_results[:top_n]
    print(f"Evaluating top {len(top_configs)} configs from Phase 1 (by {ranking_metric}):")
    for cfg in top_configs:
        print(f"  chunk_size={cfg['chunk_size']}  hit_rate={cfg['hit_rate']:.4f}  mrr={cfg['mrr']:.4f}  recall_at_k={cfg.get('recall_at_k', 0):.4f}")
    print()

    # Load ground truth
    try:
        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(ground_truth)} questions from {args.ground_truth}")

    # Initialize LLM and embeddings
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_openai import ChatOpenAI
    from src.config_manager import get_config

    cfg_obj = get_config(args.config)
    embeddings = cfg_obj.get_embeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_client = LangChainJudgeClient(llm)

    # Load raw docs once
    print("Loading documents ...")
    docs = load_docs()
    print(f"Loaded {len(docs)} document pages")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_config_results = []

    # Determine run name
    run_name = args.name
    if not run_name:
        sizes_str = "_".join(str(c["chunk_size"]) for c in top_configs)
        strategy = top_configs[0].get("chunking_strategy", "unknown")
        run_name = f"gen_{strategy}_{sizes_str}"

    # Build display name with formatted timestamp
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"
    print(f"Run: {display_name}")
    print()

    for cfg_result in top_configs:
        chunk_size = cfg_result["chunk_size"]
        chunk_overlap = cfg_result.get("chunk_overlap", 120)
        chunking_strategy = cfg_result.get("chunking_strategy", "recursive")
        top_k = cfg_result.get("top_k", 4)
        index_path = cfg_result.get("index_path")

        print(f"=== Chunk size: {chunk_size} ===")

        from langchain_community.vectorstores import FAISS

        # Load saved index from Phase 1 if available, otherwise rebuild
        if index_path and os.path.exists(os.path.join(index_path, "index.faiss")):
            print(f"  Loading saved index from: {index_path}")
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            num_chunks = cfg_result.get("num_chunks", "?")
            print(f"  Loaded index: {num_chunks} chunks")
        else:
            print(f"  No saved index found, rebuilding...")
            splitter_cfg = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_strategy": chunking_strategy,
            }
            splitter = make_text_splitter(splitter_cfg)
            chunks = splitter.split_documents(docs)
            assign_chunk_ids(chunks)
            vs = FAISS.from_documents(chunks, embeddings)
            num_chunks = len(chunks)
            print(f"  Built index: {num_chunks} chunks")

        per_question = []
        cache_hits = 0

        for i, item in enumerate(ground_truth, 1):
            question = item["question"]
            expected_answer = item.get("expected_answer", "")
            print(f"  [{i}/{len(ground_truth)}] {question[:55]}...")

            # Check cache first (retrieve to get chunk IDs for cache key)
            results_with_scores = vs.similarity_search_with_relevance_scores(question, k=top_k)
            ret_docs = [doc for doc, _ in results_with_scores]
            scores = [round(float(score), 4) for _, score in results_with_scores]
            retrieved_ids = [d.metadata.get("chunk_id", "") for d in ret_docs]
            context_chunks = [d.page_content for d in ret_docs]
            key = _cache_key(question, retrieved_ids)

            cached = _load_cache(cache_dir, key) if cache_enabled else None

            if cached:
                answer = cached["answer"]
                cache_hits += 1
                print(f"    [cache hit]")
            else:
                # Generate answer with LLM
                context_text = "\n\n".join(
                    f"[{j+1}] {chunk}" for j, chunk in enumerate(context_chunks)
                )
                prompt = f"""Answer the question using ONLY the provided context. If the context
does not contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {question}

Answer:"""
                response = llm.invoke(prompt)
                answer = response.content.strip()

                if cache_enabled:
                    _save_cache(cache_dir, key, {
                        "question": question,
                        "answer": answer,
                        "retrieved_chunk_ids": retrieved_ids,
                    })

            # Evaluate answer quality
            correctness = correctness_score(answer, expected_answer, judge_client)
            faithfulness = faithfulness_score(answer, context_chunks, judge_client)
            relevance = relevance_score(question, answer, judge_client)

            per_question.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieval_scores": scores,
                "mean_retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "cache_hit": cached is not None,
            })

        # Aggregate metrics
        n = len(per_question)
        avg_correctness = sum(q["correctness"] for q in per_question) / n if n else 0.0
        avg_faithfulness = sum(q["faithfulness"] for q in per_question) / n if n else 0.0
        avg_relevance = sum(q["relevance"] for q in per_question) / n if n else 0.0

        config_result = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "num_chunks": num_chunks,
            "top_k": top_k,
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
            "cache_hits": cache_hits,
            "avg_correctness": avg_correctness,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_retrieval_confidence": sum(q["mean_retrieval_score"] for q in per_question) / n if n else 0.0,
            "per_question": per_question,
            # Carry over Phase 1 metrics
            "phase1_hit_rate": cfg_result.get("hit_rate", 0.0),
            "phase1_mrr": cfg_result.get("mrr", 0.0),
            "index_path": index_path,
        }
        all_config_results.append(config_result)

        # Save results
        result_path = os.path.join(results_dir, f"generation_{chunk_size}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(config_result, f, indent=2)

        print(f"  Correctness: {avg_correctness:.4f}  |  Faithfulness: {avg_faithfulness:.4f}  |  Relevance: {avg_relevance:.4f}")
        print(f"  Cache hits: {cache_hits}/{n}")
        print(f"  Saved: {result_path}")
        print()

    # Print summary table
    all_config_results.sort(key=lambda r: r["avg_correctness"], reverse=True)

    width = 90
    print("=" * width)
    print(" Phase 2: Generation Eval Summary (ranked by Correctness)")
    print("=" * width)
    print(
        f"  {'Rank':<6} {'Chunk':<8} {'Chunks':<8} "
        f"{'Correct':<10} {'Faithful':<10} {'Relevant':<10} "
        f"{'HitRate':<10} {'MRR':<10}"
    )
    print("-" * width)
    for rank, r in enumerate(all_config_results, 1):
        print(
            f"  {rank:<6} {r['chunk_size']:<8} {r['num_chunks']:<8} "
            f"{r['avg_correctness']:<10.4f} {r['avg_faithfulness']:<10.4f} {r['avg_relevance']:<10.4f} "
            f"{r['phase1_hit_rate']:<10.4f} {r['phase1_mrr']:<10.4f}"
        )
    print("=" * width)
    print()

    # Show winner and how to use it
    winner = all_config_results[0]
    print(f"Winner: chunk_size={winner['chunk_size']} (Correctness: {winner['avg_correctness']:.4f})")
    if winner.get("index_path"):
        print(f"Index saved at: {winner['index_path']}")
        print()

        if args.apply_winner:
            # Update config.yaml with winner
            config["chunk_size"] = winner["chunk_size"]
            config["index_path"] = f"./{winner['index_path']}"
            with open(args.config, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Updated {args.config} with winning config:")
            print(f"  chunk_size: {winner['chunk_size']}")
            print(f"  index_path: ./{winner['index_path']}")
        else:
            print("To use this config in production, update config.yaml:")
            print(f"  chunk_size: {winner['chunk_size']}")
            print(f"  index_path: ./{winner['index_path']}")
            print()
            print("Or re-run with --apply-winner to auto-update config.yaml")
    print()
    print(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
