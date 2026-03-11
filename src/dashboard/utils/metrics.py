"""Metric calculations and chart helpers for the eval dashboard."""
import altair as alt

STRATEGY_COLORS = {
    "recursive": "#4285F4",  # blue
    "semantic": "#F4A42B",   # orange
    "fixed": "#34A853",      # green
    "sentence": "#EA4335",   # red
}


def strategy_color_scale():
    """Return an Altair color scale for chunking strategies."""
    return alt.Scale(
        domain=list(STRATEGY_COLORS.keys()),
        range=list(STRATEGY_COLORS.values()),
    )


def grouped_bar_chart(df, x_col, y_col, color_col, y_title, color_scale=None):
    """Build a grouped bar chart with data labels using Altair."""
    if color_scale is None:
        color_scale = strategy_color_scale()

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


def metric_grouped_chart(df, x_col, metric_cols, color_col, color_scale=None):
    """Build grouped bar charts — one per metric — stacked vertically."""
    if color_scale is None:
        color_scale = strategy_color_scale()

    charts = []
    for metric in metric_cols:
        chart = grouped_bar_chart(df, x_col, metric, color_col,
                                  metric.replace("_", " ").title(), color_scale)
        charts.append(chart)
    return alt.vconcat(*charts).resolve_scale(color="shared")


def composite_melt_chart(df, id_vars, value_vars, title, height=300):
    """Build a grouped bar chart from melted data with composite strategy/metric grouping.

    This avoids the overlapping-bar bug when multiple strategies share the same
    chunk_size + metric position.
    """
    melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="metric",
        value_name="value",
    )
    # Composite key ensures bars for different strategies don't overlap
    if "chunking_strategy" in melted.columns:
        melted["group"] = melted["chunking_strategy"] + " / " + melted["metric"]
    else:
        melted["group"] = melted["metric"]

    bars = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("chunk_size:N", title="Chunk Size"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("group:N", title="Strategy / Metric"),
            xOffset="group:N",
            tooltip=["chunk_size", "metric", alt.Tooltip("value:Q", format=".4f")]
                    + (["chunking_strategy"] if "chunking_strategy" in melted.columns else []),
        )
    )
    labels = (
        alt.Chart(melted)
        .mark_text(dy=-8, fontSize=10)
        .encode(
            x=alt.X("chunk_size:N"),
            y=alt.Y("value:Q"),
            text=alt.Text("value:Q", format=".3f"),
            xOffset="group:N",
        )
    )
    return (bars + labels).properties(height=height, title=title)


def compute_overall_score(df, metric_cols):
    """Add avg_overall column as mean of the given metric columns."""
    df = df.copy()
    df["avg_overall"] = df[metric_cols].mean(axis=1)
    return df
