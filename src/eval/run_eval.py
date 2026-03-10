"""
CLI entrypoint for retrieval evaluation.

Usage:
    python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 5
    python -m src.eval.run_eval --qa_pairs data/eval/qa_pairs.json --k 4 --evaltrace
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
    parser.add_argument(
        "--evaltrace",
        action="store_true",
        help="Run full EvalTrace evaluation (judge, latency, cost, SLO).",
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


def run_evaltrace_eval(qa_pairs: list, retriever, llm, config: dict) -> None:
    """Run full EvalTrace evaluation: judge + latency + cost + SLO."""
    import yaml
    from src.tracing import recorder, traced_strict_rag_answer
    from judge.adapters import LangChainJudgeClient
    from judge.rubrics.rag_answer_quality import RagAnswerQualityRubric
    from latency.slo import LatencySLO
    from report.full_report import run_full_eval, format_report
    from storage.trace_store import JsonlTraceStore
    from storage.result_store import JsonResultStore

    judge_client = LangChainJudgeClient(llm)
    rubric = RagAnswerQualityRubric()

    # Build SLO configs from config.yaml
    slo_section = config.get("slo", {})
    slo_configs = []
    if slo_section.get("e2e_p95_ms"):
        slo_configs.append(LatencySLO(name="e2e_p95", p95_ms_max=slo_section["e2e_p95_ms"]))
    if slo_section.get("retriever_p95_ms"):
        slo_configs.append(LatencySLO(name="retriever_p95", p95_ms_max=slo_section["retriever_p95_ms"]))
    if slo_section.get("llm_p95_ms"):
        slo_configs.append(LatencySLO(name="llm_p95", p95_ms_max=slo_section["llm_p95_ms"]))

    # Run traced RAG for each QA pair and collect spans per trace
    traces = []
    print(f"\nRunning EvalTrace eval on {len(qa_pairs)} QA pairs...")
    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        print(f"  [{i}/{len(qa_pairs)}] {question[:60]}...")
        recorder.reset()
        traced_strict_rag_answer(question, retriever, llm)
        traces.append(recorder.get_spans())

    # Run full evaluation
    report = run_full_eval(traces, judge_client, rubric, slo_configs or None)

    # Print report
    print()
    print(format_report(report))

    # Persist traces and results
    trace_store = JsonlTraceStore(path="data/eval/traces/traces.jsonl")
    result_store = JsonResultStore(directory="data/eval/results")

    for span_list in traces:
        trace_store.write(span_list)

    if span_list:
        trace_id = span_list[0].trace_id
        result_store.write(trace_id, "full_eval", report.to_dict())

    print("Traces saved to: data/eval/traces/traces.jsonl")
    print("Results saved to: data/eval/results/")


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

    # Import here so the module is importable without all deps present
    from dotenv import load_dotenv
    load_dotenv()

    from src.retriever import get_retriever
    from src.eval.retrieval_eval import run_retrieval_eval

    retriever = get_retriever()

    # Retrieval eval (always runs)
    print(f"Running retrieval eval with k={args.k} ...")
    results = run_retrieval_eval(qa_pairs, retriever, k=args.k)
    print_results_table(results, k=args.k)

    # EvalTrace eval (optional)
    if args.evaltrace:
        import yaml
        from langchain_openai import ChatOpenAI

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        run_evaltrace_eval(qa_pairs, retriever, llm, config)


if __name__ == "__main__":
    main()
