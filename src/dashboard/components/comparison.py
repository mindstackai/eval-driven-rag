"""Config comparison tables and strategy comparison section."""
import streamlit as st

from src.dashboard.utils.metrics import (
    compute_overall_score,
    grouped_bar_chart,
)

_RETRIEVAL_METRICS = ["hit_rate", "mrr", "recall_at_k"]
_GENERATION_METRICS = ["avg_correctness", "avg_faithfulness", "avg_relevance"]


def _render_comparison_table(df, metric_cols, extra_cols=None):
    """Render a styled comparison table with avg_overall, grouping, and recommended badge."""
    display_cols = ["display_name", "chunk_size", "chunk_overlap", "chunking_strategy",
                    "num_chunks"] + metric_cols
    if extra_cols:
        display_cols += extra_cols

    table_df = compute_overall_score(df[display_cols].copy(), metric_cols)
    table_df = table_df.sort_values(
        ["chunking_strategy", "avg_overall"], ascending=[True, False]
    ).reset_index(drop=True)

    # Strategy winner summary
    strategy_best = table_df.groupby("chunking_strategy")["avg_overall"].max()
    winner = strategy_best.idxmax()
    st.markdown(f"**Strategy winner: {winner}** (avg overall: {strategy_best[winner]:.4f})")

    # Mark best row
    best_idx = table_df["avg_overall"].idxmax()
    table_df.insert(0, "recommended", "")
    table_df.loc[best_idx, "recommended"] = "* Best"

    format_dict = {col: "{:.4f}" for col in metric_cols}
    format_dict["avg_overall"] = "{:.4f}"

    styled = table_df.style.apply(
        lambda row: ["background-color: #d4edda" if row.name == best_idx else "" for _ in row],
        axis=1,
    ).format(format_dict)
    st.dataframe(styled, use_container_width=True)


def render_retrieval_comparison(ret_df):
    """Render Phase 1 config comparison table."""
    if ret_df.empty:
        return
    st.subheader("Config Comparison")
    _render_comparison_table(ret_df, _RETRIEVAL_METRICS)


def render_generation_comparison(gen_df):
    """Render Phase 2 config comparison table."""
    if gen_df.empty:
        return
    st.subheader("Config Comparison")
    _render_comparison_table(gen_df, _GENERATION_METRICS, extra_cols=["phase1_hit_rate", "cache_hits"])


def render_strategy_comparison(ret_df):
    """Render strategy comparison section with side-by-side charts."""
    if ret_df.empty or len(ret_df["chunking_strategy"].unique()) <= 1:
        return

    st.header("Strategy Comparison")

    # Find common chunk sizes across strategies
    strategy_chunks = ret_df.groupby("chunking_strategy")["chunk_size"].apply(set)
    common_sizes = set.intersection(*strategy_chunks.values) if len(strategy_chunks) > 0 else set()

    if not common_sizes:
        st.info("No common chunk sizes across strategies to compare. "
                "Run evals with the same chunk sizes for both strategies.")
        return

    compare_df = ret_df[ret_df["chunk_size"].isin(common_sizes)].copy()
    compare_df["chunk_size"] = compare_df["chunk_size"].astype(str)

    for metric in _RETRIEVAL_METRICS:
        chart = grouped_bar_chart(compare_df, "chunk_size", metric, "chunking_strategy",
                                  metric.replace("_", " ").title())
        st.altair_chart(chart, use_container_width=True)

    # Side-by-side summary
    st.subheader("Strategy Summary")
    summary = compare_df.groupby("chunking_strategy").agg(
        avg_hit_rate=("hit_rate", "mean"),
        avg_mrr=("mrr", "mean"),
        avg_recall=("recall_at_k", "mean"),
        configs=("chunk_size", "count"),
    ).reset_index()
    summary["avg_overall"] = (summary["avg_hit_rate"] + summary["avg_mrr"] + summary["avg_recall"]) / 3
    st.dataframe(
        summary.sort_values("avg_overall", ascending=False).reset_index(drop=True).style.format({
            "avg_hit_rate": "{:.4f}",
            "avg_mrr": "{:.4f}",
            "avg_recall": "{:.4f}",
            "avg_overall": "{:.4f}",
        }),
        use_container_width=True,
    )
