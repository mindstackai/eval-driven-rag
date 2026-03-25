"""
Phase 5 — Model Comparison Dashboard sections.

Reads from traces/experiment_log.jsonl via load_experiments() and renders:
  Section 1: Leaderboard Table
  Section 2: MRR Comparison Bar Chart (base vs fine-tuned per family)
  Section 3: Quality vs Cost Scatter
  Section 4: Per-Query Drill-Down  (shown when per_query data is available)
  Section 5: Experiment Timeline
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Source colours used consistently across all charts
# ---------------------------------------------------------------------------
_SOURCE_COLORS = {"local": "#0D9488", "openai": "#F59E0B"}   # teal / amber
_SOURCE_SCALE = alt.Scale(
    domain=list(_SOURCE_COLORS.keys()),
    range=list(_SOURCE_COLORS.values()),
)

_AVG_TOKENS_PER_QUERY = 30  # rough estimate for cost_per_query calculation


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _extract_family(model_name: str) -> str:
    name = model_name.lower()
    if "bge-small" in name:
        return "bge-small"
    if "bge-base" in name:
        return "bge-base"
    if "minilm" in name or "all-minilm" in name:
        return "minilm"
    if "text-embedding-3-small" in name:
        return "openai-small"
    if "text-embedding-3-large" in name:
        return "openai-large"
    return model_name.split("/")[-1][:20]


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns needed by all sections."""
    df = df.copy()

    # Rename cfg_ prefixed columns to friendlier names for display
    rename = {c: c.replace("cfg_", "") for c in df.columns if c.startswith("cfg_")}
    df = df.rename(columns=rename)

    # Derived
    df["finetuned"] = (
        df["embedding_model"].str.contains("finetuned", case=False, na=False)
        | df["base_model"].notna()
    )
    df["model_family"] = df["embedding_model"].apply(_extract_family)

    # cost_per_query: local = $0, openai = tokens × rate
    df["cost_per_1k_tokens"] = df.get("cost_per_1k_tokens", 0.0).fillna(0.0)
    df["cost_per_query"] = df["cost_per_1k_tokens"] * _AVG_TOKENS_PER_QUERY / 1000

    # Normalise metric column names (handle different top_k values)
    hit_cols = [c for c in df.columns if c.startswith("hit_rate@")]
    recall_cols = [c for c in df.columns if c.startswith("recall@")]
    if hit_cols:
        df["hit_rate"] = df[hit_cols[0]]
    if recall_cols:
        df["recall_at_k"] = df[recall_cols[0]]

    # Short label for charts
    df["model_label"] = df["embedding_model"].apply(
        lambda m: m.split("/")[-1][:30] if "/" in m else m[-30:]
    )

    return df


def _latest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent run per embedding_model."""
    return (
        df.sort_values("timestamp", ascending=False)
        .drop_duplicates(subset=["embedding_model"], keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Sidebar filters for Model Comparison tab
# ---------------------------------------------------------------------------

def render_model_comparison_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar controls and return filtered DataFrame."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Comparison Filters")

    # Source filter
    sources = sorted(df["source"].dropna().unique().tolist())
    sel_sources = st.sidebar.multiselect(
        "Source", sources, default=sources, key="mc_source"
    )

    # Fine-tuned filter
    ft_options = ["base only", "finetuned only", "all"]
    ft_filter = st.sidebar.radio("Fine-tuned", ft_options, index=2, key="mc_finetuned")

    # Chunking strategy filter
    strategies = sorted(df["chunking_strategy"].dropna().unique().tolist())
    sel_strategies = st.sidebar.multiselect(
        "Chunking Strategy", strategies, default=strategies, key="mc_strategy"
    )

    # Top-K selector
    available_k = [3, 5, 10]
    top_k = st.sidebar.selectbox("Top-K", available_k, index=1, key="mc_top_k")

    # Apply filters
    filtered = df[df["source"].isin(sel_sources)]
    filtered = filtered[filtered["chunking_strategy"].isin(sel_strategies)]
    if ft_filter == "base only":
        filtered = filtered[~filtered["finetuned"]]
    elif ft_filter == "finetuned only":
        filtered = filtered[filtered["finetuned"]]

    return filtered, top_k


# ---------------------------------------------------------------------------
# Section 1: Leaderboard Table
# ---------------------------------------------------------------------------

def render_leaderboard(df: pd.DataFrame, top_k: int) -> None:
    st.subheader("Leaderboard")

    latest = _latest_per_model(df)

    if latest.empty:
        st.info("No experiment data yet. Run `python -m scripts.run_benchmark` first.")
        return

    hit_col = "hit_rate" if "hit_rate" in latest.columns else f"hit_rate@{top_k}"
    recall_col = "recall_at_k" if "recall_at_k" in latest.columns else f"recall@{top_k}"

    # Compute delta vs base for fine-tuned models
    base_mrr: dict[str, float] = {}
    for _, row in latest[~latest["finetuned"]].iterrows():
        base_mrr[row["model_family"]] = row["mrr"]

    rows = []
    for rank, (_, row) in enumerate(
        latest.sort_values("mrr", ascending=False).iterrows(), 1
    ):
        delta = ""
        if row["finetuned"]:
            base = base_mrr.get(row["model_family"])
            if base is not None:
                d = row["mrr"] - base
                delta = f"{d:+.4f}"

        rows.append(
            {
                "Rank": rank,
                "Model": row["embedding_model"],
                "Source": row["source"],
                "Fine-tuned": "✓" if row["finetuned"] else "",
                "MRR Δ vs base": delta,
                f"Hit@{top_k}": round(row.get(hit_col, 0.0), 4),
                "MRR": round(row["mrr"], 4),
                f"Recall@{top_k}": round(row.get(recall_col, 0.0), 4),
                "Index time (s)": round(row.get("index_time_s", 0.0), 2),
                "Cost / 1k tokens": f"${row['cost_per_1k_tokens']:.5f}"
                if row["cost_per_1k_tokens"] > 0
                else "$0",
            }
        )

    table_df = pd.DataFrame(rows)

    # Highlight best MRR row in green
    best_mrr_rank = table_df["MRR"].idxmax()

    def _highlight_best(row):
        return [
            "background-color: #d4edda" if row.name == best_mrr_rank else ""
            for _ in row
        ]

    st.dataframe(
        table_df.style.apply(_highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Section 2: MRR Comparison Bar Chart
# ---------------------------------------------------------------------------

def render_mrr_bar_chart(df: pd.DataFrame) -> None:
    st.subheader("MRR — Base vs Fine-tuned by Model Family")

    latest = _latest_per_model(df)
    if latest.empty:
        st.info("No data available.")
        return

    chart_df = latest[["model_family", "model_label", "finetuned", "mrr", "source"]].copy()
    chart_df["variant"] = chart_df["finetuned"].map({True: "fine-tuned", False: "base"})

    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("model_family:N", title="Model Family"),
            y=alt.Y("mrr:Q", title="MRR", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("source:N", scale=_SOURCE_SCALE, title="Source"),
            xOffset=alt.XOffset("variant:N"),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("variant:N", title="Variant"),
                alt.Tooltip("mrr:Q", title="MRR", format=".4f"),
            ],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(dy=-8, fontSize=11)
        .encode(
            x=alt.X("model_family:N"),
            y=alt.Y("mrr:Q"),
            xOffset=alt.XOffset("variant:N"),
            text=alt.Text("mrr:Q", format=".3f"),
        )
    )

    st.altair_chart(
        (bars + labels).properties(height=350),
        use_container_width=True,
    )

    st.caption("🟢 teal = local (free)  |  🟡 amber = OpenAI API")


# ---------------------------------------------------------------------------
# Section 3: Quality vs Cost Scatter
# ---------------------------------------------------------------------------

def render_quality_cost_scatter(df: pd.DataFrame) -> None:
    st.subheader("Quality vs Cost")

    latest = _latest_per_model(df)
    if latest.empty:
        st.info("No data available.")
        return

    scatter_df = latest[
        ["model_label", "embedding_model", "source", "finetuned",
         "mrr", "cost_per_query", "embedding_dim", "model_family"]
    ].copy()

    # Add a tiny jitter to local models (cost=0) so they don't stack
    scatter_df["cost_display"] = scatter_df["cost_per_query"].clip(lower=1e-7)
    scatter_df["variant"] = scatter_df["finetuned"].map({True: "fine-tuned", False: "base"})
    scatter_df["embedding_dim"] = scatter_df["embedding_dim"].fillna(384).astype(int)

    points = (
        alt.Chart(scatter_df)
        .mark_point(filled=True, opacity=0.85)
        .encode(
            x=alt.X(
                "cost_display:Q",
                title="Cost per query (USD, log scale)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("mrr:Q", title="MRR", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("source:N", scale=_SOURCE_SCALE, title="Source"),
            size=alt.Size(
                "embedding_dim:Q",
                title="Embedding dim",
                scale=alt.Scale(range=[60, 400]),
            ),
            shape=alt.Shape(
                "variant:N",
                scale=alt.Scale(
                    domain=["base", "fine-tuned"],
                    range=["circle", "diamond"],
                ),
                title="Variant",
            ),
            tooltip=[
                alt.Tooltip("embedding_model:N", title="Model"),
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("variant:N", title="Variant"),
                alt.Tooltip("mrr:Q", title="MRR", format=".4f"),
                alt.Tooltip("cost_per_query:Q", title="Cost/query ($)", format=".6f"),
                alt.Tooltip("embedding_dim:Q", title="Dim"),
            ],
        )
    )

    labels = (
        alt.Chart(scatter_df)
        .mark_text(dy=-14, fontSize=10)
        .encode(
            x=alt.X("cost_display:Q", scale=alt.Scale(type="log")),
            y=alt.Y("mrr:Q"),
            text=alt.Text("model_label:N"),
        )
    )

    st.altair_chart(
        (points + labels).properties(height=420),
        use_container_width=True,
    )

    # Key insight annotation
    local_ft = scatter_df[scatter_df["finetuned"] & (scatter_df["source"] == "local")]
    openai_base = scatter_df[~scatter_df["finetuned"] & (scatter_df["source"] == "openai")]
    if not local_ft.empty and not openai_base.empty:
        best_local_ft_mrr = local_ft["mrr"].max()
        best_openai_mrr = openai_base["mrr"].max()
        if best_local_ft_mrr > 0:
            insight = (
                f"💡 Best fine-tuned local model: MRR={best_local_ft_mrr:.4f} at $0 cost  "
                f"vs  best OpenAI base: MRR={best_openai_mrr:.4f}"
            )
            st.caption(insight)

    st.caption(
        "Point size = embedding dim  |  ◆ fine-tuned  |  ● base  |  "
        "local models shown at minimum cost (free)"
    )


# ---------------------------------------------------------------------------
# Section 4: Per-Query Drill-Down
# ---------------------------------------------------------------------------

def render_per_query_drilldown(df_raw: pd.DataFrame) -> None:
    """
    Show per-query retrieval results side-by-side across models.

    Requires per_query data to be stored in experiment_log.jsonl.
    If not available, shows a clear upgrade path.
    """
    st.subheader("Per-Query Drill-Down")

    # Check if any record has per_query data
    has_per_query = "per_query" in df_raw.columns and df_raw["per_query"].notna().any()

    if not has_per_query:
        st.info(
            "Per-query data is not stored in the experiment log yet. "
            "This section will activate automatically once the next benchmark run "
            "is logged (per-query results are captured from `run_phase1_eval()`)."
        )
        return

    # Collect all questions across all models
    all_questions: set[str] = set()
    model_query_map: dict[str, dict[str, dict]] = {}  # model → question → per_query row

    for _, row in df_raw.iterrows():
        pq_list = row.get("per_query", [])
        if not isinstance(pq_list, list):
            continue
        model = row.get("cfg_embedding_model", row.get("embedding_model", "unknown"))
        model_query_map[model] = {}
        for pq in pq_list:
            q = pq.get("question", "")
            all_questions.add(q)
            model_query_map[model][q] = pq

    if not all_questions:
        st.info("No questions found in per-query data.")
        return

    selected_q = st.selectbox(
        "Select a question",
        sorted(all_questions),
        format_func=lambda q: q[:90] + "..." if len(q) > 90 else q,
    )

    cols = st.columns(min(len(model_query_map), 3))
    for i, (model, q_map) in enumerate(sorted(model_query_map.items())):
        col = cols[i % len(cols)]
        pq = q_map.get(selected_q)
        with col:
            short_model = model.split("/")[-1][:25]
            st.markdown(f"**{short_model}**")
            if pq is None:
                st.caption("_no data_")
                continue

            rel_ids = set(pq.get("relevant_chunk_ids", []))
            retrieved = pq.get("retrieved_chunk_ids", [])
            sims = pq.get("similarities", [])

            st.metric("MRR", f"{pq.get('mrr', 0):.4f}")
            for rank, chunk_id in enumerate(retrieved, 1):
                sim = sims[rank - 1] if rank - 1 < len(sims) else None
                hit = chunk_id in rel_ids
                icon = "✅" if hit else "  "
                sim_str = f"  sim={sim:.3f}" if sim is not None else ""
                st.code(f"{icon} [{rank}] chunk {chunk_id}{sim_str}", language=None)


# ---------------------------------------------------------------------------
# Section 5: Experiment Timeline
# ---------------------------------------------------------------------------

def render_experiment_timeline(df: pd.DataFrame) -> None:
    st.subheader("Experiment Timeline")

    if df.empty or "timestamp" not in df.columns:
        st.info("No timeline data yet.")
        return

    timeline_df = df[["timestamp", "mrr", "model_label", "source", "finetuned"]].copy()
    timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"], utc=True, errors="coerce")
    timeline_df = timeline_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if timeline_df.empty:
        st.info("No valid timestamps found.")
        return

    timeline_df["variant"] = timeline_df["finetuned"].map({True: "fine-tuned", False: "base"})

    line = (
        alt.Chart(timeline_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("mrr:Q", title="MRR", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", title="Model"),
            strokeDash=alt.StrokeDash(
                "variant:N",
                scale=alt.Scale(
                    domain=["base", "fine-tuned"],
                    range=[[1, 0], [4, 2]],
                ),
                title="Variant",
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("mrr:Q", title="MRR", format=".4f"),
                alt.Tooltip("variant:N", title="Variant"),
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(line, use_container_width=True)
    st.caption("Solid lines = base models  |  Dashed lines = fine-tuned variants")


# ---------------------------------------------------------------------------
# Top-level renderer — called from dashboard/__init__.py
# ---------------------------------------------------------------------------

def render_model_comparison(df_raw: pd.DataFrame) -> None:
    """
    Entry point: render all 5 Model Comparison sections.

    Args:
        df_raw: Raw DataFrame from load_experiments() — not yet enriched.
    """
    if df_raw.empty:
        st.info(
            "No model comparison data yet.  "
            "Run the benchmark to populate `traces/experiment_log.jsonl`:"
        )
        st.code("python -m scripts.run_benchmark", language="bash")
        return

    df = _enrich(df_raw)

    # Sidebar filters (returns filtered df + selected top_k)
    df_filtered, top_k = render_model_comparison_sidebar(df)

    if df_filtered.empty:
        st.warning("No experiments match the current filters.")
        return

    # Section 1 — Leaderboard
    render_leaderboard(df_filtered, top_k)
    st.divider()

    # Section 2 — MRR Bar Chart
    render_mrr_bar_chart(df_filtered)
    st.divider()

    # Section 3 — Quality vs Cost
    render_quality_cost_scatter(df_filtered)
    st.divider()

    # Section 4 — Per-Query Drill-Down (uses raw df for per_query field)
    render_per_query_drilldown(df_raw)
    st.divider()

    # Section 5 — Experiment Timeline (uses full unfiltered df for history)
    render_experiment_timeline(df)
