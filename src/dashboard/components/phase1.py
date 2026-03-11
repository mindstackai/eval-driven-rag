"""Phase 1: Retrieval Eval dashboard section."""
import pandas as pd
import streamlit as st

from src.dashboard.utils.metrics import metric_grouped_chart


def render_phase1(ret_df):
    """Render the Phase 1 retrieval eval section.

    Args:
        ret_df: Filtered retrieval DataFrame.
    """
    st.header("Phase 1: Retrieval Eval")

    if ret_df.empty:
        st.info("No retrieval results match the current filters.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    best = ret_df.loc[ret_df["hit_rate"].idxmax()]
    col1.metric("Best Hit Rate", f"{best['hit_rate']:.2%}", f"chunk_size={best['chunk_size']}")
    col2.metric("Best MRR", f"{ret_df['mrr'].max():.4f}")
    col3.metric("Configs Tested", len(ret_df))
    col4.metric("Best Recall@k", f"{ret_df['recall_at_k'].max():.2%}")

    # Grouped bar charts: retrieval metrics by chunk size, colored by strategy
    st.subheader("Retrieval Metrics by Chunk Size")
    chart_df = ret_df[["chunk_size", "chunking_strategy", "hit_rate", "mrr", "recall_at_k"]].copy()
    chart_df["chunk_size"] = chart_df["chunk_size"].astype(str)
    st.altair_chart(
        metric_grouped_chart(chart_df, "chunk_size", ["hit_rate", "mrr", "recall_at_k"], "chunking_strategy"),
        use_container_width=True,
    )

    # Per-question score distribution
    st.subheader("Per-Question Score Distribution")
    raw_ret_rows = []
    for _, row in ret_df.iterrows():
        for q in row["per_question"]:
            raw_ret_rows.append({
                "chunk_size": str(row["chunk_size"]),
                "question": q["question"][:60] + "...",
                "hit": q.get("hit", 0),
                "mrr": q.get("mrr", 0),
                "recall_at_k": q.get("recall_at_k", 0),
            })
    if raw_ret_rows:
        pq_df = pd.DataFrame(raw_ret_rows)
        tab1, tab2 = st.tabs(["Hit Rate by Question", "Data Table"])
        with tab1:
            pivot = pq_df.pivot_table(index="question", columns="chunk_size", values="hit", aggfunc="mean")
            st.bar_chart(pivot)
        with tab2:
            st.dataframe(pq_df, use_container_width=True)
