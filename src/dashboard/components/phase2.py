"""Phase 2: Generation Eval dashboard section."""
import altair as alt
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
                "chunk_size": int(row["chunk_size"]),
                "chunking_strategy": row["chunking_strategy"],
                "question": q["question"],
                "expected_answer": q.get("expected_answer", ""),
                "generated_answer": q.get("generated_answer", ""),
                "correctness": q.get("correctness", 0),
                "faithfulness": q.get("faithfulness", 0),
                "relevance": q.get("relevance", 0),
                "retrieved_chunk_ids": q.get("retrieved_chunk_ids", []),
                "retrieval_scores": q.get("retrieval_scores", []),
                "mean_retrieval_score": q.get("mean_retrieval_score", 0),
                "cache_hit": q.get("cache_hit", False),
            })
    if raw_gen_rows:
        gen_pq_df = pd.DataFrame(raw_gen_rows)

        tab_heatmap, tab_detail, tab_table = st.tabs(
            ["Score Heatmap", "Question Details", "Data Table"]
        )

        with tab_heatmap:
            _render_score_heatmap(gen_pq_df)

        with tab_detail:
            _render_question_details(gen_pq_df)

        with tab_table:
            table_df = gen_pq_df.copy()
            table_df["question"] = table_df["question"].str[:60] + "..."
            table_df["answer_preview"] = table_df["generated_answer"].str[:100] + "..."
            st.dataframe(
                table_df[["chunk_size", "chunking_strategy", "question",
                          "correctness", "faithfulness", "relevance", "answer_preview"]],
                use_container_width=True,
            )


def _render_score_heatmap(pq_df):
    """Render a heatmap of all 3 scores across questions × chunk sizes."""
    melted = pq_df.melt(
        id_vars=["chunk_size", "question"],
        value_vars=["correctness", "faithfulness", "relevance"],
        var_name="metric",
        value_name="score",
    )
    melted["question_short"] = melted["question"].str[:50] + "..."
    melted["chunk_size"] = melted["chunk_size"].astype(str)

    heatmap = (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X("chunk_size:N", title="Chunk Size"),
            y=alt.Y("question_short:N", title="Question", sort=None),
            color=alt.Color(
                "score:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                title="Score",
            ),
            facet=alt.Facet("metric:N", columns=3, title="Metric"),
            tooltip=[
                alt.Tooltip("question:N", title="Full Question"),
                "chunk_size:N",
                "metric:N",
                alt.Tooltip("score:Q", format=".2f"),
            ],
        )
        .properties(width=250, height=40 * min(len(pq_df["question"].unique()), 20))
    )
    st.altair_chart(heatmap)


def _render_question_details(pq_df):
    """Render per-question detail cards with full Q&A and scores."""
    questions = pq_df["question"].unique().tolist()
    selected_q = st.selectbox(
        "Select a question",
        questions,
        format_func=lambda q: q[:80] + "..." if len(q) > 80 else q,
    )

    q_rows = pq_df[pq_df["question"] == selected_q].sort_values("chunk_size")

    for _, row in q_rows.iterrows():
        with st.expander(
            f"chunk_size={row['chunk_size']} · {row['chunking_strategy']} — "
            f"C={row['correctness']:.2f}  F={row['faithfulness']:.2f}  R={row['relevance']:.2f}",
            expanded=len(q_rows) == 1,
        ):
            score_cols = st.columns(3)
            score_cols[0].metric("Correctness", f"{row['correctness']:.2f}")
            score_cols[1].metric("Faithfulness", f"{row['faithfulness']:.2f}")
            score_cols[2].metric("Relevance", f"{row['relevance']:.2f}")

            st.markdown("**Expected Answer:**")
            st.info(row["expected_answer"] or "_not provided_")

            st.markdown("**Generated Answer:**")
            st.success(row["generated_answer"] or "_empty_")

            chunk_ids = row["retrieved_chunk_ids"]
            if chunk_ids:
                scores = row.get("retrieval_scores", [])
                mean_score = row.get("mean_retrieval_score", None)
                if scores:
                    min_s = min(scores)
                    max_s = max(scores)
                    summary = f"mean={mean_score:.4f}, min={min_s:.4f}, max={max_s:.4f}" if mean_score is not None else f"min={min_s:.4f}, max={max_s:.4f}"
                    st.markdown(f"**Retrieved Chunks** ({len(chunk_ids)}) — {summary}:")
                else:
                    st.markdown(f"**Retrieved Chunks** ({len(chunk_ids)}):")
                for i, cid in enumerate(chunk_ids):
                    score_str = f" — score: {scores[i]:.4f}" if i < len(scores) else ""
                    st.code(f"{cid}{score_str}", language=None)

            if row["cache_hit"]:
                st.caption("(served from cache)")
