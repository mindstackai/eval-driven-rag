"""
Streamlit Evaluation Dashboard — visualize eval results across chunk configs.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/eval_dashboard.py
"""
import json
import os
from datetime import datetime
from glob import glob

import pandas as pd
import streamlit as st

RESULTS_DIR = "eval/results"

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_all_results():
    """Load all retrieval and generation result files into DataFrames."""
    retrieval_rows = []
    generation_rows = []

    for fpath in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
        fname = os.path.basename(fpath)
        with open(fpath, "r") as f:
            data = json.load(f)

        if fname.startswith("retrieval_"):
            retrieval_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "hit_rate": data.get("hit_rate", 0),
                "mrr": data.get("mrr", 0),
                "recall_at_k": data.get("recall_at_k", 0),
                "per_question": data.get("per_question", []),
            })

        elif fname.startswith("generation_"):
            retrieval_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
                "hit_rate": data.get("phase1_hit_rate", 0),
                "mrr": data.get("phase1_mrr", 0),
                "recall_at_k": 0,
                "per_question": [],
            })
            generation_rows.append({
                "file": fname,
                "chunk_size": data.get("chunk_size"),
                "chunk_overlap": data.get("chunk_overlap"),
                "chunking_strategy": data.get("chunking_strategy"),
                "num_chunks": data.get("num_chunks"),
                "top_k": data.get("top_k"),
                "timestamp": data.get("timestamp", ""),
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


def _per_question_df(rows_with_pq, phase):
    """Flatten per_question lists into a single DataFrame for drill-down."""
    flat = []
    for row in rows_with_pq:
        cs = row.get("chunk_size")
        ts = row.get("timestamp", "")
        strategy = row.get("chunking_strategy", "")
        for q in row.get("per_question", []):
            entry = {
                "chunk_size": cs,
                "timestamp": ts,
                "chunking_strategy": strategy,
                "question": q.get("question", ""),
            }
            if phase == "retrieval":
                entry["hit"] = q.get("hit", 0)
                entry["mrr"] = q.get("mrr", 0)
                entry["recall_at_k"] = q.get("recall_at_k", 0)
            else:
                entry["correctness"] = q.get("correctness", 0)
                entry["faithfulness"] = q.get("faithfulness", 0)
                entry["relevance"] = q.get("relevance", 0)
            flat.append(entry)
    return pd.DataFrame(flat) if flat else pd.DataFrame()


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Eval Dashboard", layout="wide")
st.title("Evaluation Dashboard")

ret_df, gen_df = load_all_results()

if ret_df.empty and gen_df.empty:
    st.warning("No eval results found in `eval/results/`. Run Phase 1 or Phase 2 first.")
    st.code("python -m src.eval_retrieval\npython -m src.eval_generation", language="bash")
    st.stop()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.markdown("### Filters")

# Strategy filter
all_strategies = sorted(
    set(ret_df["chunking_strategy"].dropna().unique().tolist()
        + (gen_df["chunking_strategy"].dropna().unique().tolist() if not gen_df.empty else []))
)
selected_strategies = st.sidebar.multiselect(
    "Chunking Strategy",
    all_strategies,
    default=all_strategies,
)

# Timestamp filter (run selector)
all_timestamps = sorted(
    set(ret_df["timestamp"].dropna().unique().tolist()
        + (gen_df["timestamp"].dropna().unique().tolist() if not gen_df.empty else [])),
    reverse=True,
)
selected_runs = st.sidebar.multiselect(
    "Eval Run (timestamp)",
    all_timestamps,
    default=[all_timestamps[0]] if all_timestamps else [],
)

# Apply filters
if not ret_df.empty:
    ret_df = ret_df[
        ret_df["chunking_strategy"].isin(selected_strategies)
        & ret_df["timestamp"].isin(selected_runs)
    ]
if not gen_df.empty:
    gen_df = gen_df[
        gen_df["chunking_strategy"].isin(selected_strategies)
        & gen_df["timestamp"].isin(selected_runs)
    ]

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

    # Bar chart: retrieval metrics by chunk size
    st.subheader("Retrieval Metrics by Chunk Size")
    chart_df = ret_df[["chunk_size", "hit_rate", "mrr", "recall_at_k"]].copy()
    chart_df["chunk_size"] = chart_df["chunk_size"].astype(str)
    chart_df = chart_df.set_index("chunk_size")
    st.bar_chart(chart_df)

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
            st.dataframe(pq_df, width="stretch")

    # Config comparison table
    st.subheader("Config Comparison")
    display_cols = ["chunk_size", "chunk_overlap", "chunking_strategy", "num_chunks", "hit_rate", "mrr", "recall_at_k", "timestamp"]
    st.dataframe(
        ret_df[display_cols].sort_values("hit_rate", ascending=False).reset_index(drop=True),
        width="stretch",
    )

# ---------------------------------------------------------
# Phase 2: Generation metrics
# ---------------------------------------------------------
st.header("Phase 2: Generation Eval")

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

    # Bar chart: generation quality by chunk size
    st.subheader("Generation Quality by Chunk Size")
    gen_chart = gen_df[["chunk_size", "avg_correctness", "avg_faithfulness", "avg_relevance"]].copy()
    gen_chart.columns = ["chunk_size", "Correctness", "Faithfulness", "Relevance"]
    gen_chart["chunk_size"] = gen_chart["chunk_size"].astype(str)
    gen_chart = gen_chart.set_index("chunk_size")
    st.bar_chart(gen_chart)

    # Combined retrieval + generation view
    st.subheader("Retrieval vs Generation (Combined)")
    combined = gen_df[["chunk_size", "phase1_hit_rate", "phase1_mrr", "avg_correctness", "avg_faithfulness", "avg_relevance"]].copy()
    combined.columns = ["chunk_size", "Hit Rate", "MRR", "Correctness", "Faithfulness", "Relevance"]
    combined["chunk_size"] = combined["chunk_size"].astype(str)
    combined = combined.set_index("chunk_size")
    st.bar_chart(combined)

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
            st.dataframe(gen_pq_df, width="stretch")

    # Config comparison
    st.subheader("Config Comparison")
    gen_display = ["chunk_size", "chunk_overlap", "chunking_strategy", "num_chunks",
                   "avg_correctness", "avg_faithfulness", "avg_relevance",
                   "phase1_hit_rate", "cache_hits", "timestamp"]
    st.dataframe(
        gen_df[gen_display].sort_values("avg_correctness", ascending=False).reset_index(drop=True),
        width="stretch",
    )

# ---------------------------------------------------------
# Run new eval
# ---------------------------------------------------------
st.header("Run New Eval")

with st.form("run_eval_form"):
    st.markdown("Launch a new Phase 1 retrieval eval with custom chunk sizes.")

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
        })
        results.append(metrics)

        result_path = os.path.join(RESULTS_DIR, f"retrieval_{cs}_{timestamp}.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

    progress.progress(1.0, text="Done!")
    st.success(f"Saved {len(results)} results to eval/results/ (timestamp: {timestamp})")
    st.info("Refresh the page to see new results in the charts above.")
