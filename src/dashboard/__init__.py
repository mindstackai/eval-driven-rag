"""Eval Dashboard — modular Streamlit dashboard for visualizing eval results.

Standalone usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. streamlit run src/dashboard/app.py

Imported by another app:
    from src.dashboard import render_dashboard
    render_dashboard(results_dir="eval/results", config_path="config.yaml")
"""
import logging

import streamlit as st

from src.dashboard.components.comparison import (
    render_generation_comparison,
    render_retrieval_comparison,
    render_strategy_comparison,
)
from src.dashboard.components.model_comparison import render_model_comparison
from src.dashboard.components.phase1 import render_phase1
from src.dashboard.components.phase2 import render_phase2
from src.dashboard.components.run_eval import render_run_eval
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.utils.data_loader import load_all_results

logger = logging.getLogger(__name__)


@st.cache_data(ttl=30)
def _load_experiments():
    """Load experiment log — refreshes every 30s so new benchmark runs appear."""
    from src.eval.eval_trace import load_experiments
    return load_experiments()


def render_dashboard(results_dir="eval/results", config_path="config.yaml"):
    """Render the full evaluation dashboard.

    Args:
        results_dir: Path to directory containing eval result JSON files.
        config_path: Path to config.yaml.
    """
    st.set_page_config(page_title="EvalTrace Dashboard", layout="wide")
    st.title("EvalTrace Dashboard")

    # Top-level tabs
    tab_p1p2, tab_models = st.tabs(["Phase 1 & 2 — Chunking Eval", "Model Comparison"])

    # ------------------------------------------------------------------ #
    #  Tab 1: existing Phase 1 / Phase 2 / Run Eval sections              #
    # ------------------------------------------------------------------ #
    with tab_p1p2:
        ret_df_raw, gen_df_raw = load_all_results(results_dir)

        if ret_df_raw.empty and gen_df_raw.empty:
            st.warning(
                f"No eval results found in `{results_dir}/`. "
                "Run Phase 1 or Phase 2 first."
            )
            st.code(
                "python -m src.eval_retrieval\npython -m src.eval_generation",
                language="bash",
            )
        else:
            # Sidebar filters (only active when this tab is open)
            all_strategies, selected_strategies, selected_runs, ret_df, gen_df = (
                render_sidebar(ret_df_raw, gen_df_raw)
            )

            render_phase1(ret_df)
            render_retrieval_comparison(ret_df)
            render_strategy_comparison(ret_df)
            render_phase2(gen_df, ret_df)
            render_generation_comparison(gen_df)
            render_run_eval(ret_df_raw, all_strategies, results_dir, config_path)

    # ------------------------------------------------------------------ #
    #  Tab 2: Model Comparison (reads from traces/experiment_log.jsonl)   #
    # ------------------------------------------------------------------ #
    with tab_models:
        exp_df = _load_experiments()
        render_model_comparison(exp_df)
