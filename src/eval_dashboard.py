"""
Streamlit Evaluation Dashboard — visualize eval results across chunk configs.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/eval_dashboard.py
"""
import json
import os
from datetime import datetime
from glob import glob

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_DIR = "eval/results"

STRATEGY_COLORS = {
    "recursive": "#4285F4",  # blue
    "semantic": "#F4A42B",   # orange
    "fixed": "#34A853",      # green
    "sentence": "#EA4335",   # red
}

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def _format_timestamp(ts):
    """Format raw timestamp string (e.g. '20260310_162357') for display."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ts


def _make_display_name(data):
    """Build a display_name from result data, with fallback for old runs."""
    if data.get("display_name"):
        return data["display_name"]
    if data.get("run_name"):
        ts_fmt = _format_timestamp(data.get("timestamp", ""))
        return f"{data['run_name']} ({ts_fmt})"
    # Fallback for old results without run_name
    ts_fmt = _format_timestamp(data.get("timestamp", ""))
    return ts_fmt or data.get("timestamp", "unknown")


def load_all_results():
    """Load all retrieval and generation result files into DataFrames."""
    retrieval_rows = []
    generation_rows = []

    for fpath in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
        fname = os.path.basename(fpath)
        with open(fpath, "r") as f:
            data = json.load(f)

        display_name = _make_display_name(data)

        if fname.startswith("retrieval_"):
            retrieval_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "run_name": data.get("run_name", ""),
                "display_name": display_name,
                "hit_rate": data.get("hit_rate", 0),
                "mrr": data.get("mrr", 0),
                "recall_at_k": data.get("recall_at_k", 0),
                "per_question": data.get("per_question", []),
            })

        elif fname.startswith("generation_"):
            generation_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "run_name": data.get("run_name", ""),
                "display_name": display_name,
                "avg_correctness": data.get("avg_correctness", 0),
                "avg_faithfulness": data.get("avg_faithfulness", 0),
                "avg_relevance": data.get("avg_relevance", 0),
                "cache_hits": data.get("cache_hits", 0),
                "phase1_hit_rate": data.get("phase1_hit_rate", 0),
                "phase1_mrr": data.get("phase1_mrr", 0),
                "per_question": data.get("per_question", []),
            })

    ret_df = pd.DataFrame(retrieval_rows) if retrieval_rows else pd.DataFrame()
    gen_df = pd.DataFrame(generation_rows) if generation_rows else pd.DataFrame()
    return ret_df, gen_df


# ---------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------

def _grouped_bar_chart(df, x_col, y_col, color_col, y_title, color_scale=None):
    """Build a grouped bar chart with data labels using Altair."""
    if color_scale is None:
        color_scale = alt.Scale(
            domain=list(STRATEGY_COLORS.keys()),
            range=list(STRATEGY_COLORS.values()),
        )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", title=x_col.replace("_", " ").title()),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color(f"{color_col}:N", scale=color_scale, title=color_col.replace("_", " ").title()),
            xOffset=alt.XOffset(f"{color_col}:N"),
            tooltip=list(df.columns),
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(dy=-8, fontSize=11)
        .encode(
            x=alt.X(f"{x_col}:N"),
            y=alt.Y(f"{y_col}:Q"),
            text=alt.Text(f"{y_col}:Q", format=".3f"),
            xOffset=alt.XOffset(f"{color_col}:N"),
        )
    )
    return (bars + labels).properties(height=350)


def _metric_grouped_chart(df, x_col, metric_cols, color_col, color_scale=None):
    """Build grouped bar charts — one per metric — stacked vertically."""
    if color_scale is None:
        color_scale = alt.Scale(
            domain=list(STRATEGY_COLORS.keys()),
            range=list(STRATEGY_COLORS.values()),
        )

    charts = []
    for metric in metric_cols:
        chart = _grouped_bar_chart(df, x_col, metric, color_col, metric.replace("_", " ").title(), color_scale)
        charts.append(chart)
    return alt.vconcat(*charts).resolve_scale(color="shared")


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Eval Dashboard", layout="wide")
st.title("Evaluation Dashboard")

ret_df_raw, gen_df_raw = load_all_results()

if ret_df_raw.empty and gen_df_raw.empty:
    st.warning("No eval results found in `eval/results/`. Run Phase 1 or Phase 2 first.")
    st.code("python -m src.eval_retrieval\npython -m src.eval_generation", language="bash")
    st.stop()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.markdown("### Filters")

# Strategy filter
all_strategies = sorted(
    set(ret_df_raw["chunking_strategy"].dropna().unique().tolist()
        + (gen_df_raw["chunking_strategy"].dropna().unique().tolist() if not gen_df_raw.empty else []))
)
selected_strategies = st.sidebar.multiselect(
    "Chunking Strategy",
    all_strategies,
    default=all_strategies,
)

# Run filter (display_name as primary label, with tooltip showing full name)
all_display_names = sorted(
    set(ret_df_raw["display_name"].dropna().unique().tolist()
        + (gen_df_raw["display_name"].dropna().unique().tolist() if not gen_df_raw.empty else [])),
    reverse=True,
)
selected_runs = st.sidebar.multiselect(
    "Eval Run",
    all_display_names,
    default=all_display_names,
    help="Select runs to include. Each label shows the run name (or formatted timestamp for older runs).",
)

# Clear all / Select all buttons
sb_col1, sb_col2 = st.sidebar.columns(2)
if sb_col1.button("Select all", use_container_width=True):
    st.session_state["selected_runs_override"] = all_display_names
    st.rerun()
if sb_col2.button("Clear all", use_container_width=True):
    st.session_state["selected_runs_override"] = []
    st.rerun()

# Handle override from buttons
if "selected_runs_override" in st.session_state:
    selected_runs = st.session_state.pop("selected_runs_override")

# Apply filters
ret_df = ret_df_raw.copy()
gen_df = gen_df_raw.copy()

if not ret_df.empty:
    ret_df = ret_df[
        ret_df["chunking_strategy"].isin(selected_strategies)
        & ret_df["display_name"].isin(selected_runs)
    ]
    # Keep only latest run per (chunk_size, chunking_strategy) for cleaner charts
    if not ret_df.empty:
        ret_df = ret_df.sort_values("timestamp", ascending=False).drop_duplicates(
            subset=["chunk_size", "chunking_strategy"], keep="first"
        )
if not gen_df.empty:
    gen_df = gen_df[
        gen_df["chunking_strategy"].isin(selected_strategies)
        & gen_df["display_name"].isin(selected_runs)
    ]
    if not gen_df.empty:
        gen_df = gen_df.sort_values("timestamp", ascending=False).drop_duplicates(
            subset=["chunk_size", "chunking_strategy"], keep="first"
        )

# ---------------------------------------------------------
# Phase 1: Retrieval metrics
# ---------------------------------------------------------
st.header("Phase 1: Retrieval Eval")

if ret_df.empty:
    st.info("No retrieval results match the current filters.")
else:
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
        _metric_grouped_chart(chart_df, "chunk_size", ["hit_rate", "mrr", "recall_at_k"], "chunking_strategy"),
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

    # Config comparison table with avg_overall and recommended badge
    st.subheader("Config Comparison")
    table_df = ret_df[["display_name", "chunk_size", "chunk_overlap", "chunking_strategy",
                        "num_chunks", "hit_rate", "mrr", "recall_at_k"]].copy()
    table_df["avg_overall"] = (table_df["hit_rate"] + table_df["mrr"] + table_df["recall_at_k"]) / 3
    table_df = table_df.sort_values(["chunking_strategy", "avg_overall"], ascending=[True, False]).reset_index(drop=True)

    # Mark best row
    best_idx = table_df["avg_overall"].idxmax()
    table_df["recommended"] = ""
    table_df.loc[best_idx, "recommended"] = "* Best"

    cols_order = ["recommended", "display_name", "chunk_size", "chunk_overlap", "chunking_strategy",
                  "num_chunks", "hit_rate", "mrr", "recall_at_k", "avg_overall"]
    styled = table_df[cols_order].style.apply(
        lambda row: ["background-color: #d4edda" if row.name == best_idx else "" for _ in row],
        axis=1,
    ).format({
        "hit_rate": "{:.4f}",
        "mrr": "{:.4f}",
        "recall_at_k": "{:.4f}",
        "avg_overall": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True)

# ---------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------
if not ret_df.empty and len(ret_df["chunking_strategy"].unique()) > 1:
    st.header("Strategy Comparison")

    # Find common chunk sizes across strategies
    strategy_chunks = ret_df.groupby("chunking_strategy")["chunk_size"].apply(set)
    common_sizes = set.intersection(*strategy_chunks.values) if len(strategy_chunks) > 0 else set()

    if common_sizes:
        compare_df = ret_df[ret_df["chunk_size"].isin(common_sizes)].copy()
        compare_df["chunk_size"] = compare_df["chunk_size"].astype(str)

        for metric in ["hit_rate", "mrr", "recall_at_k"]:
            chart = _grouped_bar_chart(compare_df, "chunk_size", metric, "chunking_strategy",
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
    else:
        st.info("No common chunk sizes across strategies to compare. "
                "Run evals with the same chunk sizes for both strategies.")

# ---------------------------------------------------------
# Phase 2: Generation metrics
# ---------------------------------------------------------
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
    st.info("No generation results available. Run Phase 2 to see generation metrics.")
else:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    best_gen = gen_df.loc[gen_df["avg_correctness"].idxmax()]
    col1.metric("Best Correctness", f"{best_gen['avg_correctness']:.2%}", f"chunk_size={best_gen['chunk_size']}")
    col2.metric("Best Faithfulness", f"{gen_df['avg_faithfulness'].max():.2%}")
    col3.metric("Best Relevance", f"{gen_df['avg_relevance'].max():.2%}")
    col4.metric("Total Cache Hits", int(gen_df["cache_hits"].sum()))

    # Grouped bar chart: generation quality by chunk size
    st.subheader("Generation Quality by Chunk Size")
    gen_chart_df = gen_df[["chunk_size", "chunking_strategy", "avg_correctness", "avg_faithfulness", "avg_relevance"]].copy()
    gen_chart_df["chunk_size"] = gen_chart_df["chunk_size"].astype(str)
    st.altair_chart(
        _metric_grouped_chart(gen_chart_df, "chunk_size",
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
        st.markdown("**Retrieval Metrics (Phase 1)**")
        ret_melt = combined_df.melt(
            id_vars=["chunk_size", "chunking_strategy"],
            value_vars=["phase1_hit_rate", "phase1_mrr"],
            var_name="metric", value_name="value",
        )
        # Composite key so bars for different strategies don't overlap
        ret_melt["group"] = ret_melt["chunking_strategy"] + " / " + ret_melt["metric"]
        ret_chart = (
            alt.Chart(ret_melt)
            .mark_bar()
            .encode(
                x=alt.X("chunk_size:N", title="Chunk Size"),
                y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", title="Strategy / Metric"),
                xOffset="group:N",
                tooltip=["chunk_size", "chunking_strategy", "metric", alt.Tooltip("value:Q", format=".4f")],
            )
        )
        ret_labels = (
            alt.Chart(ret_melt)
            .mark_text(dy=-8, fontSize=10)
            .encode(
                x=alt.X("chunk_size:N"),
                y=alt.Y("value:Q"),
                text=alt.Text("value:Q", format=".3f"),
                xOffset="group:N",
            )
        )
        st.altair_chart((ret_chart + ret_labels).properties(height=300), use_container_width=True)

    with col_gen:
        st.markdown("**Generation Metrics (Phase 2)**")
        gen_melt = combined_df.melt(
            id_vars=["chunk_size", "chunking_strategy"],
            value_vars=["avg_correctness", "avg_faithfulness", "avg_relevance"],
            var_name="metric", value_name="value",
        )
        # Composite key so bars for different strategies don't overlap
        gen_melt["group"] = gen_melt["chunking_strategy"] + " / " + gen_melt["metric"]
        gen_chart = (
            alt.Chart(gen_melt)
            .mark_bar()
            .encode(
                x=alt.X("chunk_size:N", title="Chunk Size"),
                y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", title="Strategy / Metric"),
                xOffset="group:N",
                tooltip=["chunk_size", "chunking_strategy", "metric", alt.Tooltip("value:Q", format=".4f")],
            )
        )
        gen_labels = (
            alt.Chart(gen_melt)
            .mark_text(dy=-8, fontSize=10)
            .encode(
                x=alt.X("chunk_size:N"),
                y=alt.Y("value:Q"),
                text=alt.Text("value:Q", format=".3f"),
                xOffset="group:N",
            )
        )
        st.altair_chart((gen_chart + gen_labels).properties(height=300), use_container_width=True)

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
            pivot = gen_pq_df.pivot_table(index="question", columns="chunk_size", values="correctness", aggfunc="mean")
            st.bar_chart(pivot)
        with tab2:
            st.dataframe(gen_pq_df, use_container_width=True)

    # Config comparison with avg_overall and recommended badge
    st.subheader("Config Comparison")
    gen_table = gen_df[["display_name", "chunk_size", "chunk_overlap", "chunking_strategy", "num_chunks",
                        "avg_correctness", "avg_faithfulness", "avg_relevance",
                        "phase1_hit_rate", "cache_hits"]].copy()
    gen_table["avg_overall"] = (gen_table["avg_correctness"] + gen_table["avg_faithfulness"] + gen_table["avg_relevance"]) / 3
    gen_table = gen_table.sort_values(["chunking_strategy", "avg_overall"], ascending=[True, False]).reset_index(drop=True)

    best_gen_idx = gen_table["avg_overall"].idxmax()
    gen_table["recommended"] = ""
    gen_table.loc[best_gen_idx, "recommended"] = "* Best"

    gen_cols_order = ["recommended", "display_name", "chunk_size", "chunk_overlap", "chunking_strategy", "num_chunks",
                      "avg_correctness", "avg_faithfulness", "avg_relevance", "avg_overall",
                      "phase1_hit_rate", "cache_hits"]
    gen_styled = gen_table[gen_cols_order].style.apply(
        lambda row: ["background-color: #d4edda" if row.name == best_gen_idx else "" for _ in row],
        axis=1,
    ).format({
        "avg_correctness": "{:.4f}",
        "avg_faithfulness": "{:.4f}",
        "avg_relevance": "{:.4f}",
        "avg_overall": "{:.4f}",
        "phase1_hit_rate": "{:.4f}",
    })
    st.dataframe(gen_styled, use_container_width=True)

# ---------------------------------------------------------
# Run new eval
# ---------------------------------------------------------
st.header("Run New Eval")

# --- Phase 1 ---
with st.form("run_eval_form"):
    st.subheader("Run Phase 1: Retrieval Eval")
    st.markdown("Build indexes with different chunk sizes and evaluate retrieval quality.")

    run_name_input = st.text_input(
        "Run name (short description/goal)",
        placeholder="e.g. baseline recursive strategy",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_sizes_input = st.text_input("Chunk sizes (comma-separated)", "128, 256, 512, 1024")
    with col2:
        overlap = st.number_input("Chunk overlap", min_value=0, value=120)
    with col3:
        strategy = st.selectbox("Chunking strategy", ["recursive", "fixed", "semantic", "sentence"])

    submitted = st.form_submit_button("Run Phase 1 Eval")

if submitted:
    try:
        chunk_sizes = [int(x.strip()) for x in chunk_sizes_input.split(",")]
    except ValueError:
        st.error("Invalid chunk sizes. Enter comma-separated integers.")
        st.stop()

    import yaml
    from src.ingest import load_docs, assign_chunk_ids
    from src.splitters import make_text_splitter
    from src.eval_retrieval import _build_temp_index, _run_retrieval_for_config, _build_reference_content_map

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    from dotenv import load_dotenv
    load_dotenv()
    from src.config_manager import get_config
    cfg_obj = get_config()
    embeddings = cfg_obj.get_embeddings()

    progress = st.progress(0, text="Loading documents...")
    docs = load_docs()
    if not docs:
        st.error("No documents found in data/raw/")
        st.stop()

    ref_content_map = _build_reference_content_map(docs, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine run name
    run_name = run_name_input.strip() if run_name_input.strip() else f"{strategy}_{'_'.join(str(s) for s in chunk_sizes)}_overlap{overlap}"
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"

    results = []

    for i, cs in enumerate(chunk_sizes):
        progress.progress((i) / len(chunk_sizes), text=f"Evaluating chunk_size={cs}...")
        vs, chunks = _build_temp_index(docs, cs, overlap, strategy, embeddings)
        metrics = _run_retrieval_for_config(vs, ground_truth, config.get("top_k", 4), ref_content_map)
        metrics.update({
            "chunk_size": cs,
            "chunk_overlap": overlap,
            "chunking_strategy": strategy,
            "num_chunks": len(chunks),
            "top_k": config.get("top_k", 4),
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
        })
        results.append(metrics)

        result_path = os.path.join(RESULTS_DIR, f"retrieval_{cs}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

    progress.progress(1.0, text="Done!")
    st.success(f"Saved {len(results)} results — **{display_name}**")
    st.info("Refresh the page to see new results in the charts above.")

# --- Phase 2 ---
st.divider()
with st.form("run_phase2_form"):
    st.subheader("Run Phase 2: Generation Eval")
    st.markdown("Run LLM generation on top configs from Phase 1 and evaluate answer quality.")

    p2_run_name = st.text_input(
        "Run name",
        placeholder="e.g. testing semantic generation",
        key="p2_run_name",
    )

    p2_col1, p2_col2 = st.columns(2)
    with p2_col1:
        p2_strategy_filter = st.selectbox(
            "Filter Phase 1 configs by strategy",
            ["all"] + all_strategies,
            key="p2_strategy",
        )
    with p2_col2:
        p2_top_n = st.number_input("Top N configs to evaluate", min_value=1, value=4, key="p2_top_n")

    # Show which Phase 1 configs will be used
    if not ret_df_raw.empty:
        preview_df = ret_df_raw.copy()
        if p2_strategy_filter != "all":
            preview_df = preview_df[preview_df["chunking_strategy"] == p2_strategy_filter]
        if not preview_df.empty:
            preview_df = preview_df.sort_values("timestamp", ascending=False).drop_duplicates(
                subset=["chunk_size", "chunking_strategy"], keep="first"
            )
            preview_df = preview_df.sort_values("hit_rate", ascending=False).head(p2_top_n)
            st.markdown("**Phase 1 configs that will be used:**")
            st.dataframe(
                preview_df[["chunking_strategy", "chunk_size", "chunk_overlap", "hit_rate", "mrr", "display_name"]]
                .reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.warning("No Phase 1 results found for the selected strategy.")

    p2_submitted = st.form_submit_button("Run Phase 2 Eval")

if p2_submitted:
    import yaml
    from src.ingest import load_docs, assign_chunk_ids
    from src.splitters import make_text_splitter
    from src.eval_generation import _load_phase1_results, _cache_key, _load_cache, _save_cache, _answer_correctness
    from src.eval.answer_eval import faithfulness_score, answer_relevance_score

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("eval", {})
    cache_dir = eval_cfg.get("cache_dir", "eval/cache")
    cache_enabled = eval_cfg.get("cache_responses", True)

    with open("eval/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    from dotenv import load_dotenv
    load_dotenv()
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from src.config_manager import get_config
    cfg_obj = get_config()
    embeddings = cfg_obj.get_embeddings()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Load Phase 1 results and filter
    phase1_results = _load_phase1_results(RESULTS_DIR)
    if p2_strategy_filter != "all":
        phase1_results = [r for r in phase1_results if r.get("chunking_strategy") == p2_strategy_filter]

    if not phase1_results:
        st.error("No Phase 1 results found for the selected strategy.")
        st.stop()

    top_configs = phase1_results[:p2_top_n]

    progress = st.progress(0, text="Loading documents...")
    docs = load_docs()
    if not docs:
        st.error("No documents found in data/raw/")
        st.stop()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = p2_run_name.strip() if p2_run_name.strip() else f"gen_{'_'.join(str(c['chunk_size']) for c in top_configs)}"
    display_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
    display_name = f"{run_name} ({display_ts})"

    total_steps = len(top_configs) * len(ground_truth)
    step = 0

    for cfg_result in top_configs:
        chunk_size = cfg_result["chunk_size"]
        chunk_overlap = cfg_result.get("chunk_overlap", 120)
        chunking_strategy = cfg_result.get("chunking_strategy", "recursive")
        top_k = cfg_result.get("top_k", 4)

        splitter_cfg = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
        }
        splitter = make_text_splitter(splitter_cfg)
        chunks = splitter.split_documents(docs)
        assign_chunk_ids(chunks)
        vs = FAISS.from_documents(chunks, embeddings)
        retriever = vs.as_retriever(search_kwargs={"k": top_k})

        per_question = []
        cache_hits = 0

        for item in ground_truth:
            step += 1
            progress.progress(step / total_steps, text=f"chunk_size={chunk_size}: {item['question'][:40]}...")
            question = item["question"]
            expected_answer = item.get("expected_answer", "")

            ret_docs = retriever.invoke(question)
            retrieved_ids = [d.metadata.get("chunk_id", "") for d in ret_docs]
            context_chunks = [d.page_content for d in ret_docs]
            key = _cache_key(question, retrieved_ids)

            cached = _load_cache(cache_dir, key) if cache_enabled else None

            if cached:
                answer = cached["answer"]
                cache_hits += 1
            else:
                context_text = "\n\n".join(f"[{j+1}] {chunk}" for j, chunk in enumerate(context_chunks))
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

            correctness = _answer_correctness(answer, expected_answer, llm)
            faithfulness = faithfulness_score(answer, context_chunks, llm)
            relevance = answer_relevance_score(question, answer, llm)

            per_question.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "retrieved_chunk_ids": retrieved_ids,
                "correctness": correctness,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "cache_hit": cached is not None,
            })

        n = len(per_question)
        config_result = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "num_chunks": len(chunks),
            "top_k": top_k,
            "timestamp": timestamp,
            "run_name": run_name,
            "display_name": display_name,
            "cache_hits": cache_hits,
            "avg_correctness": sum(q["correctness"] for q in per_question) / n if n else 0.0,
            "avg_faithfulness": sum(q["faithfulness"] for q in per_question) / n if n else 0.0,
            "avg_relevance": sum(q["relevance"] for q in per_question) / n if n else 0.0,
            "per_question": per_question,
            "phase1_hit_rate": cfg_result.get("hit_rate", 0.0),
            "phase1_mrr": cfg_result.get("mrr", 0.0),
        }

        result_path = os.path.join(RESULTS_DIR, f"generation_{chunk_size}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(config_result, f, indent=2)

    progress.progress(1.0, text="Done!")
    st.success(f"Phase 2 complete — **{display_name}**")
    st.info("Refresh the page to see new results in the charts above.")
