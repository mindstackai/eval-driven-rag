"""Sidebar filters for the eval dashboard."""
import streamlit as st


def render_sidebar(ret_df_raw, gen_df_raw):
    """Render sidebar filters and return (selected_strategies, selected_runs, ret_df, gen_df).

    Args:
        ret_df_raw: Unfiltered retrieval DataFrame.
        gen_df_raw: Unfiltered generation DataFrame.

    Returns:
        Tuple of (all_strategies, selected_strategies, selected_runs, ret_df, gen_df)
        where ret_df/gen_df are filtered and deduplicated.
    """
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

    # Run filter (display_name as primary label)
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

    return all_strategies, selected_strategies, selected_runs, ret_df, gen_df
