"""
CLI entrypoint for retrieval evaluation.

Usage:
    python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 5
"""
import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval eval over a QA pairs file."
    )
    parser.add_argument(
        "--qa_pairs",
        type=str,
        default="data/eval/qa_pairs.json",
        help="Path to the QA pairs JSON file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval cut-off for recall@k (default: 5).",
    )
    return parser.parse_args()


def print_results_table(results: dict, k: int) -> None:
    """Print a formatted summary table of eval results."""
    width = 40
    print()
    print("=" * width)
    print(" Retrieval Eval Results")
    print("=" * width)
    print(f"  {'Metric':<25} {'Score':>10}")
    print("-" * width)
    print(f"  {'Recall@' + str(k):<25} {results['recall_at_k']:>10.4f}")
    print(f"  {'MRR':<25} {results['mrr']:>10.4f}")
    avg_rel = results.get("avg_chunk_relevance")
    if avg_rel is not None:
        print(f"  {'Avg Chunk Relevance':<25} {avg_rel:>10.4f}")
    else:
        print(f"  {'Avg Chunk Relevance':<25} {'N/A':>10}")
    print("=" * width)
    print()


def main() -> None:
    args = parse_args()

    # Load QA pairs
    try:
        with open(args.qa_pairs, "r") as f:
            qa_pairs: list[dict] = json.load(f)
    except FileNotFoundError:
        print(f"Error: QA pairs file not found: {args.qa_pairs}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: Failed to parse QA pairs file: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(qa_pairs)} QA pairs from {args.qa_pairs}")
    print(f"Running retrieval eval with k={args.k} ...")

    # Import here so the module is importable without all deps present
    from dotenv import load_dotenv
    load_dotenv()

    from src.retriever import get_retriever
    from src.eval.retrieval_eval import run_retrieval_eval

    retriever = get_retriever()
    results = run_retrieval_eval(qa_pairs, retriever, k=args.k)

    print_results_table(results, k=args.k)


if __name__ == "__main__":
    main()
