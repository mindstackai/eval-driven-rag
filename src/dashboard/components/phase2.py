"""Phase 2: Generation Eval dashboard section."""
import pandas as pd
import streamlit as st

from src.dashboard.utils.metrics import composite_melt_chart, metric_grouped_chart


def render_phase2(gen_df, ret_df):
    """Render the Phase 2 generation eval section.

    Args:
        gen_df: Filtered generation DataFrame.
        ret_df: Filtered retrieval DataFrame (used to detect missing strategies).
    """
    st.header("Phase 2: Generation Eval")

    # Check which strategies have Phase 2 results
    gen_strategies = set(gen_df["chunking_strategy"].unique()) if not gen_df.empty else set()
    ret_strategies = set(ret_df["chunking_strategy"].unique()) if not ret_df.empty else set()
    missing_gen_strategies = ret_strategies - gen_strategies

    if missing_gen_strategies:
        missing_str = ", ".join(sorted(missing_gen_strategies))
        st.warning(
            f"Phase 2 only ran for **{', '.join(sorted(gen_strategies)) or 'no'}** strategy. "
            f"No generation results for: **{missing_str}**. "
            f"Run Phase 2 for {missing_str} strategy separately:\n\n"
            f"`python -m src.eval_generation --name '{missing_str} generation test'`"
        )

    if gen_df.empty:
        st.info("No Phase 2 results found for selected filters. Run Phase 2 eval first.")
        st.code("python -m src.eval_generation", language="bash")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    best_gen = gen_df.loc[gen_df["avg_correctness"].idxmax()]
    col1.metric("Best Correctness", f"{best_gen['avg_correctness']:.2%}", f"chunk_size={best_gen['chunk_size']}")
    col2.metric("Best Faithfulness", f"{gen_df['avg_faithfulness'].max():.2%}")
    col3.metric("Best Relevance", f"{gen_df['avg_relevance'].max():.2%}")
    col4.metric("Total Cache Hits", int(gen_df["cache_hits"].sum()))

    # Grouped bar chart: generation quality by chunk size
    st.subheader("Generation Quality by Chunk Size")
    gen_chart_df = gen_df[["chunk_size", "chunking_strategy",
                           "avg_correctness", "avg_faithfulness", "avg_relevance"]].copy()
    gen_chart_df["chunk_size"] = gen_chart_df["chunk_size"].astype(str)
    st.altair_chart(
        metric_grouped_chart(gen_chart_df, "chunk_size",
                             ["avg_correctness", "avg_faithfulness", "avg_relevance"],
                             "chunking_strategy"),
        use_container_width=True,
    )

    # Split Retrieval vs Generation into two separate charts (different scales)
    st.subheader("Retrieval vs Generation Metrics")
    st.caption("Split into separate charts because retrieval and generation metrics have different scales.")

    combined_df = gen_df[["chunk_size", "chunking_strategy",
                          "phase1_hit_rate", "phase1_mrr",
                          "avg_correctness", "avg_faithfulness", "avg_relevance"]].copy()
    combined_df["chunk_size"] = combined_df["chunk_size"].astype(str)

    col_ret, col_gen = st.columns(2)

    with col_ret:
        st.altair_chart(
            composite_melt_chart(
                combined_df,
                id_vars=["chunk_size", "chunking_strategy"],
                value_vars=["phase1_hit_rate", "phase1_mrr"],
                title="Retrieval Metrics (Phase 1)",
            ),
            use_container_width=True,
        )

    with col_gen:
        st.altair_chart(
            composite_melt_chart(
                combined_df,
                id_vars=["chunk_size", "chunking_strategy"],
                value_vars=["avg_correctness", "avg_faithfulness", "avg_relevance"],
                title="Generation Metrics (Phase 2)",
            ),
            use_container_width=True,
        )

    # Per-question drill-down
    st.subheader("Per-Question Scores")
    raw_gen_rows = []
    for _, row in gen_df.iterrows():
        for q in row["per_question"]:
            raw_gen_rows.append({
                "chunk_size": str(row["chunk_size"]),
                "question": q["question"][:60] + "...",
                "correctness": q.get("correctness", 0),
                "faithfulness": q.get("faithfulness", 0),
                "relevance": q.get("relevance", 0),
                "answer_preview": q.get("generated_answer", "")[:100] + "...",
            })
    if raw_gen_rows:
        gen_pq_df = pd.DataFrame(raw_gen_rows)
        tab1, tab2 = st.tabs(["Correctness by Question", "Data Table"])
        with tab1:
            pivot = gen_pq_df.pivot_table(index="question", columns="chunk_size",
                                          values="correctness", aggfunc="mean")
            st.bar_chart(pivot)
        with tab2:
            st.dataframe(gen_pq_df, use_container_width=True)
