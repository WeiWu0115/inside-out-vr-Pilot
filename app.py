"""
Inside Out: Multi-Agent Cognitive State Visualization
=====================================================
Visualizes how multiple interpretive agents negotiate user cognitive states
during VR escape room gameplay, and when/how the system should intervene.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

DATA_PATH = os.path.join(os.path.dirname(__file__), "outputs", "agent_outputs.csv")
COMPARISON_PATH = os.path.join(os.path.dirname(__file__), "outputs", "comparison_correct.csv")
THREE_WAY_PATH = os.path.join(os.path.dirname(__file__), "outputs", "three_way_comparison.csv")
TOLERANCE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "tolerance_results.csv")
PROMPT_DETAIL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "prompt_detection_detail.csv")

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

AGENT_LABEL_COLORS = {
    "focused": "#2196F3", "searching": "#FF9800", "locked": "#F44336", "ambiguous": "#BDBDBD",
    "active": "#4CAF50", "hesitant": "#FFC107", "inactive": "#E91E63", "unknown": "#BDBDBD",
    "progressing": "#4CAF50", "stalled": "#FF5722", "failing": "#B71C1C",
    "transient": "#90CAF9", "persistent": "#FF9800", "looping": "#F44336",
    # Population agent labels
    "transitioning": "#9E9E9E", "disoriented": "#E91E63", "actively_solving": "#4CAF50",
    "exploring": "#FF9800", "cognitively_stuck": "#F44336",
}

CATEGORY_COLORS = {
    "consensus_intervene": "#F44336",
    "probe": "#FF9800",
    "watch": "#4CAF50",
}

CATEGORY_ICONS = {
    "consensus_intervene": "🚨",
    "probe": "🔍",
    "watch": "👁",
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time_min"] = df["window_start"] / 60.0

    # Handle both old (attention_state) and new (attention_label) column names
    renames = {}
    for agent in ["attention", "action", "performance", "temporal", "population"]:
        old_state = f"{agent}_state"
        new_label = f"{agent}_label"
        if old_state in df.columns and new_label not in df.columns:
            renames[old_state] = new_label
        # Ensure confidence column exists
        conf_col = f"{agent}_confidence"
        if conf_col not in df.columns:
            df[conf_col] = 0.5
        # Ensure reasoning column exists
        reason_col = f"{agent}_reasoning"
        if reason_col not in df.columns:
            df[reason_col] = ""
    if renames:
        df = df.rename(columns=renames)

    # Ensure negotiation columns exist
    for col, default in [
        ("disagreement_type", "unstructured"),
        ("disagreement_intensity", 0.5),
        ("dominant_tension", "none"),
        ("support_category", "watch"),
        ("support_confidence", 0.5),
        ("support_rationale", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


@st.cache_data
def load_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None
    df = pd.read_csv(COMPARISON_PATH)
    df["time_min"] = df["window_start"] / 60.0

    # Build categories
    df["io_cat"] = df.get("support_category", pd.Series(dtype=str)).map({
        "consensus_intervene": "intervene", "probe": "probe", "watch": "watch"
    }).fillna("watch")

    def _expert_cat(row):
        if row.get("expert_action") == "PROMPT":
            pt = str(row.get("expert_prompt_type", ""))
            if pt == "E":
                return "intervene"
            return "probe"  # R, V, SpecialA/B/C
        return "watch"
    df["expert_cat"] = df.apply(_expert_cat, axis=1)

    return df


@st.cache_data
def load_three_way():
    if not os.path.exists(THREE_WAY_PATH):
        return None, None, None
    tw = pd.read_csv(THREE_WAY_PATH)
    tw["time_min"] = tw["window_start"] / 60.0
    tol = pd.read_csv(TOLERANCE_PATH) if os.path.exists(TOLERANCE_PATH) else None
    detail = pd.read_csv(PROMPT_DETAIL_PATH) if os.path.exists(PROMPT_DETAIL_PATH) else None
    return tw, tol, detail


def make_three_way_timeline(tdf):
    """Three-row timeline: Facilitator vs Expert vs IO decisions."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_titles=["Facilitator (Ground Truth)", "Rule-Based System", "Inside Out"],
    )
    cat_colors = {"intervene": "#F44336", "probe": "#FF9800", "watch": "#4CAF50"}

    # Facilitator
    for cat in ["intervene", "probe", "watch"]:
        mask = tdf["facilitator_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=tdf["time_min"][mask], y=[cat] * mask.sum(),
            mode="markers", marker=dict(size=5, color=cat_colors[cat], opacity=0.6),
            name=f"Fac: {cat}", legendgroup=cat, showlegend=True,
        ), row=1, col=1)

    # Expert
    for cat in ["intervene", "probe", "watch"]:
        mask = tdf["expert_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=tdf["time_min"][mask], y=[cat] * mask.sum(),
            mode="markers", marker=dict(size=5, color=cat_colors[cat], opacity=0.6),
            name=f"Exp: {cat}", legendgroup=cat, showlegend=False,
        ), row=2, col=1)

    # IO
    for cat in ["intervene", "probe", "watch"]:
        mask = tdf["io_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=tdf["time_min"][mask], y=[cat] * mask.sum(),
            mode="markers", marker=dict(size=5, color=cat_colors[cat], opacity=0.6),
            name=f"IO: {cat}", legendgroup=cat, showlegend=False,
        ), row=3, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_layout(
        height=450, template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_tolerance_chart(tol_df):
    """Line chart: F1 scores at different temporal tolerances."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tol_df["tolerance_sec"], y=tol_df["io_f1"],
        mode="lines+markers", name="Inside Out",
        line=dict(color="#FF9800", width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=tol_df["tolerance_sec"], y=tol_df["ex_f1"],
        mode="lines+markers", name="Rule-Based",
        line=dict(color="#2196F3", width=3), marker=dict(size=8),
    ))
    # Add recall as dashed lines
    fig.add_trace(go.Scatter(
        x=tol_df["tolerance_sec"], y=tol_df["io_recall"],
        mode="lines", name="IO Recall",
        line=dict(color="#FF9800", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=tol_df["tolerance_sec"], y=tol_df["ex_recall"],
        mode="lines", name="Rule-Based Recall",
        line=dict(color="#2196F3", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        height=350, template="plotly_white",
        xaxis_title="Temporal Tolerance (seconds)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_three_way_heatmap(tdf):
    """3x3 heatmap: Facilitator vs IO decisions."""
    ct = pd.crosstab(tdf["facilitator_cat"], tdf["io_cat"])
    order = ["intervene", "probe", "watch"]
    ct = ct.reindex(index=[o for o in order if o in ct.index],
                    columns=[o for o in order if o in ct.columns], fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=ct.values, x=ct.columns, y=ct.index,
        colorscale="YlOrRd",
        text=[[str(v) for v in row] for row in ct.values],
        texttemplate="%{text}", textfont=dict(size=16),
    ))
    fig.update_layout(
        height=300,
        xaxis_title="Inside Out Decision",
        yaxis_title="Facilitator Decision",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_puzzle_detection_chart(detail_df):
    """Grouped bar chart: IO vs Expert detection rate per puzzle."""
    puzzles = sorted(detail_df["puzzle"].dropna().unique())
    io_rates = []
    ex_rates = []
    for p in puzzles:
        sub = detail_df[detail_df["puzzle"] == p]
        io_rates.append(sub["io_hit"].mean() * 100)
        ex_rates.append(sub["expert_hit"].mean() * 100)

    # Shorten puzzle names
    short = [p.replace("Spoke Puzzle: ", "").replace("Hub Puzzle: ", "") for p in puzzles]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Inside Out", x=short, y=io_rates,
                         marker_color="#FF9800"))
    fig.add_trace(go.Bar(name="Rule-Based", x=short, y=ex_rates,
                         marker_color="#2196F3"))
    fig.update_layout(
        barmode="group", height=350, template="plotly_white",
        yaxis_title="Detection Rate (%)",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_comparison_timeline(cdf):
    """Side-by-side timeline: expert vs IO decisions."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_titles=["Rule-Based System", "Inside Out"],
    )

    cat_colors = {
        "intervene": "#F44336",
        "probe": "#FF9800",
        "watch": "#4CAF50",
    }

    # Expert row
    for cat in ["intervene", "probe", "watch"]:
        mask = cdf["expert_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=cdf["time_min"][mask], y=[cat] * mask.sum(),
            mode="markers",
            marker=dict(size=6, color=cat_colors.get(cat, "#9E9E9E"), opacity=0.6),
            name=f"Rule: {cat}", showlegend=True,
        ), row=1, col=1)

    # IO row
    for cat in ["intervene", "probe", "watch"]:
        mask = cdf["io_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=cdf["time_min"][mask], y=[cat] * mask.sum(),
            mode="markers",
            marker=dict(size=6, color=cat_colors.get(cat, "#9E9E9E"), opacity=0.6),
            name=f"IO: {cat}", showlegend=True,
        ), row=2, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_layout(
        height=350, template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_agreement_heatmap(cdf):
    """Cross-tabulation heatmap of expert vs IO decisions."""
    ct = pd.crosstab(cdf["expert_cat"], cdf["io_cat"])
    # Ensure order
    order = ["intervene", "probe", "watch"]
    ct = ct.reindex(index=[o for o in order if o in ct.index],
                    columns=[o for o in order if o in ct.columns], fill_value=0)

    fig = go.Figure(go.Heatmap(
        z=ct.values, x=ct.columns, y=ct.index,
        colorscale="YlOrRd",
        text=[[str(v) for v in row] for row in ct.values],
        texttemplate="%{text}",
        textfont=dict(size=16),
    ))
    fig.update_layout(
        height=300,
        xaxis_title="Inside Out Decision",
        yaxis_title="Rule-Based Decision",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_disagreement_scatter(cdf):
    """Show where the two systems disagree, colored by type of disagreement."""
    # Categorize disagreements
    def _disagree_type(row):
        e, i = row["expert_cat"], row["io_cat"]
        if e == i:
            return "agree"
        if e != "watch" and i == "watch":
            return "rb_only"
        if i != "watch" and e == "watch":
            return "io_only"
        # Both active but different level (e.g. probe vs intervene)
        return "different_level"

    cdf = cdf.copy()
    cdf["disagree_type"] = cdf.apply(_disagree_type, axis=1)

    colors = {
        "agree": "#E0E0E0",
        "rb_only": "#2196F3",
        "io_only": "#F44336",
        "different_level": "#9C27B0",
    }
    labels = {
        "agree": "Agree (same category)",
        "rb_only": "Rule-based acts, IO watches",
        "io_only": "IO acts, rule-based watches",
        "different_level": "Both act, different level",
    }

    fig = go.Figure()
    for dtype in ["agree", "rb_only", "io_only", "different_level"]:
        mask = cdf["disagree_type"] == dtype
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=cdf["time_min"][mask],
            y=cdf.get("disagreement_intensity", pd.Series(0.5, index=cdf.index))[mask],
            mode="markers",
            marker=dict(
                size=4 if dtype == "agree" else 8,
                color=colors[dtype],
                opacity=0.3 if dtype == "agree" else 0.7,
            ),
            name=labels.get(dtype, dtype),
        ))

    fig.update_layout(
        height=300, template="plotly_white",
        xaxis_title="Time (minutes)",
        yaxis_title="IO Disagreement Intensity",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_agent_confidence_timeline(pdf):
    """Show agent confidence over time with color = label."""
    agents = [
        ("attention", "Perceptual Agent"),
        ("action", "Behavioral Agent"),
        ("performance", "Progress Agent"),
        ("temporal", "Temporal Agent"),
        ("population", "Population Agent"),
    ]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_titles=[title for _, title in agents],
    )

    for i, (agent, title) in enumerate(agents, 1):
        label_col = f"{agent}_label"
        conf_col = f"{agent}_confidence"

        for label in pdf[label_col].unique():
            mask = pdf[label_col] == label
            color = AGENT_LABEL_COLORS.get(label, "#BDBDBD")
            fig.add_trace(
                go.Scatter(
                    x=pdf["time_min"][mask],
                    y=pdf[conf_col][mask],
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.7),
                    name=f"{label}",
                    showlegend=(i == 1),
                    hovertemplate=(
                        f"<b>{title}</b>: {label}<br>"
                        "Confidence: %{y:.0%}<br>"
                        "Time: %{x:.1f} min<br>"
                        "<extra></extra>"
                    ),
                ),
                row=i, col=1,
            )
        fig.update_yaxes(range=[0, 1], dtick=0.25, row=i, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=5, col=1)
    fig.update_layout(
        height=750, template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_disagreement_timeline(pdf):
    """Disagreement intensity over time, colored by type."""
    fig = go.Figure()

    type_colors = {
        "contradictory": "#F44336",
        "constructive": "#4CAF50",
        "unstructured": "#9E9E9E",
    }

    for dtype in pdf["disagreement_type"].unique():
        mask = pdf["disagreement_type"] == dtype
        color = type_colors.get(dtype, "#9E9E9E")
        fig.add_trace(go.Scatter(
            x=pdf["time_min"][mask],
            y=pdf["disagreement_intensity"][mask],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.7),
            name=dtype,
            hovertemplate=(
                f"Type: {dtype}<br>"
                "Intensity: %{y:.0%}<br>"
                "Tension: %{customdata}<br>"
                "Time: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
            customdata=pdf["dominant_tension"][mask],
        ))

    # Mark intervention points
    interventions = pdf[pdf["support_category"] == "consensus_intervene"]
    if len(interventions) > 0:
        fig.add_trace(go.Scatter(
            x=interventions["time_min"],
            y=interventions["disagreement_intensity"],
            mode="markers",
            marker=dict(size=12, color="#F44336", symbol="star",
                        line=dict(width=1, color="#B71C1C")),
            name="Intervention",
            hovertemplate=(
                "<b>INTERVENE</b><br>"
                "Action: %{customdata}<br>"
                "Time: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
            customdata=interventions["suggested_support"],
        ))

    # Mark probes
    probes = pdf[pdf["support_category"] == "probe"]
    if len(probes) > 0:
        fig.add_trace(go.Scatter(
            x=probes["time_min"],
            y=probes["disagreement_intensity"],
            mode="markers",
            marker=dict(size=9, color="#FF9800", symbol="diamond",
                        line=dict(width=1, color="#E65100")),
            name="Probe",
            hovertemplate=(
                "<b>PROBE</b><br>"
                "Action: %{customdata}<br>"
                "Time: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
            customdata=probes["suggested_support"],
        ))

    fig.update_layout(
        height=300, template="plotly_white",
        xaxis_title="Time (minutes)",
        yaxis_title="Disagreement Intensity",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_cluster_vs_tension_heatmap(df):
    """Cross-tab: K-means cluster vs dominant tension."""
    if "cluster_id" not in df.columns:
        return None

    ct = pd.crosstab(df["cluster_id"], df["dominant_tension"], normalize="index").round(3)

    cluster_labels = {
        0: "C0: Transition", 1: "C1: Waiting", 2: "C2: Active Solving",
        3: "C3: Exploration", 4: "C4: Stuck-on-Clue",
    }
    y_labels = [cluster_labels.get(c, f"C{c}") for c in ct.index]

    fig = go.Figure(go.Heatmap(
        z=ct.values, x=ct.columns, y=y_labels,
        colorscale="YlOrRd",
        text=[[f"{v:.0%}" for v in row] for row in ct.values],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        height=350,
        xaxis_title="Dominant Tension (Agent Negotiation)",
        yaxis_title="K-Means Cluster (FDG)",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickangle=30),
    )
    return fig


def render_negotiation_panel(current):
    """Render the negotiation transcript for a single time step."""
    agents = [
        ("attention", "🔵 Perceptual"),
        ("action", "🟢 Behavioral"),
        ("performance", "🟠 Progress"),
        ("temporal", "⏱ Temporal"),
        ("population", "👥 Population"),
    ]

    # Agent cards
    cols = st.columns(5)
    for col, (agent, icon) in zip(cols, agents):
        label = str(current.get(f"{agent}_label", "unknown"))
        conf = float(current.get(f"{agent}_confidence", 0) or 0)
        reasoning = str(current.get(f"{agent}_reasoning", "") or "")
        color = AGENT_LABEL_COLORS.get(label, "#BDBDBD")

        col.markdown(
            f"**{icon}**<br>"
            f"<span style='color:{color}; font-size:1.3em; font-weight:bold'>{label}</span><br>"
            f"Confidence: **{conf:.0%}**<br>"
            f"<small>{reasoning}</small>",
            unsafe_allow_html=True,
        )

    # Negotiation result
    st.markdown("---")
    d_type = str(current.get("disagreement_type", "unstructured") or "unstructured")
    d_intensity = float(current.get("disagreement_intensity", 0) or 0)
    tension = str(current.get("dominant_tension", "none") or "none")
    support = str(current.get("suggested_support", "none") or "none")
    s_conf = float(current.get("support_confidence", 0) or 0)
    rationale = str(current.get("support_rationale", "") or "")
    category = str(current.get("support_category", "watch") or "watch")

    icon = CATEGORY_ICONS.get(category, "")
    cat_color = CATEGORY_COLORS.get(category, "#9E9E9E")

    if category == "consensus_intervene":
        st.error(
            f"**{icon} INTERVENE: {support}** (confidence: {s_conf:.0%})\n\n"
            f"Disagreement: {d_type} — *{tension}* (intensity: {d_intensity:.0%})\n\n"
            f"{rationale}"
        )
    elif category == "probe":
        st.warning(
            f"**{icon} PROBE: {support}** (confidence: {s_conf:.0%})\n\n"
            f"Disagreement: {d_type} — *{tension}* (intensity: {d_intensity:.0%})\n\n"
            f"{rationale}"
        )
    else:
        st.success(
            f"**{icon} {support.upper()}** (confidence: {s_conf:.0%})\n\n"
            f"Disagreement: {d_type} — *{tension}* (intensity: {d_intensity:.0%})\n\n"
            f"{rationale}"
        )


def make_dominance_chart(pdf):
    """
    Stacked area chart showing which agent dominates at each time step.
    Each agent's confidence is normalized so they sum to 1.0 — the agent
    with the largest area is "winning the debate" at that moment.
    """
    agents = [
        ("attention", "Perceptual", "#2196F3"),
        ("action", "Behavioral", "#4CAF50"),
        ("performance", "Progress", "#FF9800"),
        ("temporal", "Temporal", "#9C27B0"),
        ("population", "Population", "#795548"),
    ]

    time = pdf["time_min"].values

    # Collect raw confidences
    raw = {}
    for agent, _, _ in agents:
        col = f"{agent}_confidence"
        raw[agent] = pdf[col].fillna(0).values if col in pdf.columns else np.zeros(len(pdf))

    # Normalize so they sum to 1 at each time step
    total = sum(raw.values())
    total = [max(t, 0.01) for t in total]  # avoid division by zero
    normed = {}
    for agent in raw:
        normed[agent] = [r / t for r, t in zip(raw[agent], total)]

    fig = go.Figure()

    fill_colors = {
        "attention": "rgba(33,150,243,0.6)",
        "action": "rgba(76,175,80,0.6)",
        "performance": "rgba(255,152,0,0.6)",
        "temporal": "rgba(156,39,176,0.6)",
        "population": "rgba(121,85,72,0.6)",
    }

    for agent, label, color in agents:
        fig.add_trace(go.Scatter(
            x=time, y=normed[agent],
            mode="lines",
            name=label,
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=fill_colors[agent],
            hovertemplate=(
                f"<b>{label} Agent</b><br>"
                "Dominance: %{y:.0%}<br>"
                "Time: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
        ))

    # Add intervention markers at the top
    if "support_category" in pdf.columns:
        interventions = pdf[pdf["support_category"] == "consensus_intervene"]
        if len(interventions) > 0:
            fig.add_trace(go.Scatter(
                x=interventions["time_min"],
                y=[1.02] * len(interventions),
                mode="markers",
                marker=dict(size=10, color="#F44336", symbol="triangle-down"),
                name="🚨 Intervene",
                hovertemplate="<b>INTERVENTION</b><br>Time: %{x:.1f} min<extra></extra>",
            ))

        probes = pdf[pdf["support_category"] == "probe"]
        if len(probes) > 0:
            fig.add_trace(go.Scatter(
                x=probes["time_min"],
                y=[1.02] * len(probes),
                mode="markers",
                marker=dict(size=8, color="#FF9800", symbol="diamond"),
                name="🔍 Probe",
                hovertemplate="<b>PROBE</b><br>Time: %{x:.1f} min<extra></extra>",
            ))

    fig.update_layout(
        height=350,
        template="plotly_white",
        xaxis_title="Time (minutes)",
        yaxis_title="Agent Dominance",
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def make_dominance_line_chart(pdf):
    """
    Line chart showing each agent's confidence over time.
    The highest line at any point = the dominant agent.
    Crossover points = where negotiation shifts.
    """
    agents = [
        ("attention", "Perceptual", "#2196F3"),
        ("action", "Behavioral", "#4CAF50"),
        ("performance", "Progress", "#FF9800"),
        ("temporal", "Temporal", "#9C27B0"),
        ("population", "Population", "#795548"),
    ]

    fig = go.Figure()

    for agent, label, color in agents:
        conf_col = f"{agent}_confidence"
        # Smooth with rolling average for readability
        smoothed = pdf[conf_col].rolling(window=3, min_periods=1, center=True).mean()

        fig.add_trace(go.Scatter(
            x=pdf["time_min"],
            y=smoothed,
            mode="lines",
            name=label,
            line=dict(width=2.5, color=color),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Confidence: %{y:.0%}<br>"
                "Time: %{x:.1f} min<br>"
                "<extra></extra>"
            ),
        ))

    # Mark intervention points
    if "support_category" in pdf.columns:
        interventions = pdf[pdf["support_category"] == "consensus_intervene"]
        if len(interventions) > 0:
            fig.add_trace(go.Scatter(
                x=interventions["time_min"],
                y=[0.95] * len(interventions),
                mode="markers",
                marker=dict(size=12, color="#F44336", symbol="star",
                            line=dict(width=1, color="#B71C1C")),
                name="🚨 Intervene",
            ))

    fig.update_layout(
        height=350,
        template="plotly_white",
        xaxis_title="Time (minutes)",
        yaxis_title="Agent Confidence",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def check_password():
    """Simple password gate using st.secrets and session_state."""
    if st.session_state.get("authenticated"):
        return True

    st.title("🔒 Inside Out")
    st.caption("Multi-Agent Negotiation of Cognitive States in VR Escape Room")
    st.markdown("---")

    password = st.text_input("Enter password to access the app", type="password")
    if st.button("Login", type="primary"):
        correct = st.secrets.get("app_password", "insideout2026")
        if password == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Inside Out: Multi-Agent VR", layout="wide")

    if not check_password():
        return

    st.title("🧠 Inside Out")
    st.caption("Multi-Agent Negotiation of Cognitive States in VR Escape Room")

    df = load_data()

    # Sidebar
    st.sidebar.header("Controls")
    participants = sorted(df["participant_id"].unique())
    selected_pid = st.sidebar.selectbox("Player", participants, format_func=lambda x: f"Player {x}")
    pdf = df[df["participant_id"] == selected_pid].sort_values("window_start").reset_index(drop=True)

    puzzles = ["All"] + list(pdf["puzzle_id"].unique())
    selected_puzzle = st.sidebar.selectbox("Puzzle", puzzles)
    if selected_puzzle != "All":
        pdf = pdf[pdf["puzzle_id"] == selected_puzzle].reset_index(drop=True)

    # Stats
    st.sidebar.markdown("---")
    n_intervene = (pdf["support_category"] == "consensus_intervene").sum() if "support_category" in pdf.columns else 0
    n_probe = (pdf["support_category"] == "probe").sum() if "support_category" in pdf.columns else 0
    st.sidebar.markdown(f"**Windows:** {len(pdf)}")
    st.sidebar.markdown(f"**Duration:** {pdf['time_min'].max() - pdf['time_min'].min():.1f} min")
    st.sidebar.markdown(f"**🚨 Interventions:** {n_intervene}")
    st.sidebar.markdown(f"**🔍 Probes:** {n_probe}")

    # Load comparison data
    comp_df = load_comparison()
    tw_df, tol_df, detail_df = load_three_way()

    # Top-level tabs
    tab_study, tab_io, tab5, tab6, tab_fw = st.tabs([
        "🏠 Study Overview",
        "🧠 IO Agent",
        "🆚 Rule-Based vs IO",
        "🎯 Facilitator Benchmark",
        "🔮 Future Work",
    ])

    with tab_study:
        st.subheader("Study Overview: VR Escape Room Pilot")
        st.markdown(
            "**18 participants** played a VR escape room (Meta Quest Pro) with eye tracking, "
            "interaction logging, and a human facilitator observing in real time. "
            "The game consists of **4 spoke puzzles** and **1 hub puzzle** that integrates them all."
        )

        # Room layout + heatmap + movement paths in one row
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### VR Escape Room Layout")
            st.image("assets/EscapeRoom_HeattMapBG_20250813.png", use_container_width=True)
        with col2:
            st.markdown("#### Player Position Heatmap")
            st.image("assets/aggregate_heatmap_overlaid.png", use_container_width=True)
        with col3:
            st.markdown("#### Player Movement Paths")
            st.image("assets/All_User_Paths.png", use_container_width=True)

        st.caption("Left: top-down view of the VR escape room. Center: aggregate heatmap (yellow = high density). Right: smoothed head-tracking trajectories (each color = one player).")

        # Time distribution
        st.markdown("---")
        st.markdown("#### How Players Spent Their Time")
        col1, col2 = st.columns(2)
        with col1:
            st.image("assets/average_timeshare_composition.png", use_container_width=True)
        with col2:
            st.image("assets/average_timeshare_by_puzzle_bar.png", use_container_width=True)

        st.markdown(
            "Players spent **37.5%** of their time in the environment (navigating between puzzles) "
            "and **32.9%** on the Hub Puzzle (Cooking Pot) — the most complex puzzle requiring integration "
            "of all four spoke solutions. Each spoke puzzle took 6-10% of total time."
        )

        # Puzzle difficulty
        st.markdown("---")
        st.markdown("#### Puzzle Difficulty (Average Solve Time)")
        puzzle_times = {
            "Protein": "3:11", "Pasta": "2:46", "Water": "2:54",
            "Sunlight": "3:04", "Hub (Cooking Pot)": "8:18", "Overall": "20:13"
        }
        time_cols = st.columns(6)
        for col, (puzzle, time) in zip(time_cols, puzzle_times.items()):
            col.metric(puzzle, time)

        # Facilitator prompts
        st.markdown("---")
        st.markdown("#### Facilitator Prompts: When Did Players Need Help?")
        st.markdown(
            "A trained facilitator observed each session and gave **two types of prompts**:\n"
            "- **Reflective** (blue): guiding questions — *\"What do you think this does?\"*\n"
            "- **Explicit** (red): direct answers — *\"Set the protein to 3\"*\n\n"
            "Average per player: **8.1 reflective** + **3.6 explicit** = 11.7 total prompts per session."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.image("assets/avg_reflective_prompts_per_puzzle.png", use_container_width=True)
        with col2:
            st.image("assets/avg_explicit_prompts_per_puzzle.png", use_container_width=True)

        st.markdown(
            "**Key observations:**\n"
            "- Hub Puzzle needed the most help of both types (most complex puzzle)\n"
            "- Sunlight puzzle rarely needed explicit prompts (easiest puzzle)\n"
            "- Protein puzzle had high explicit prompts but moderate reflective — players understood the goal but couldn't execute\n"
            "- These facilitator prompts serve as **ground truth** for benchmarking the Inside Out system"
        )

        # Data summary
        st.markdown("---")
        st.markdown("#### Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participants", "18")
        col2.metric("With Eye Tracking", "11")
        col3.metric("5s Windows", "5,265")
        col4.metric("Facilitator Prompts", "223")

    with tab_io:
        tab_arch, tab0, tab1, tab2, tab3, tab4 = st.tabs([
            "🧩 Architecture", "🏔 Dominance", "📊 Confidence",
            "⚡ Negotiation", "🔬 Cluster", "▶️ Playback",
        ])
        with tab_arch:
            st.subheader("Inside Out V3: Agent Architecture")

            st.markdown(
                "**Inside Out** is a multi-agent system for interpreting cognitive states in VR, "
                "inspired by Pixar's *Inside Out*. Instead of classifying a player into "
                "a single state (e.g. \"stuck\" or \"not stuck\"), multiple agents each produce their own "
                "interpretation from different sensory channels — perception, action, performance, "
                "temporal context, and population norms. The system's key insight is that **disagreement "
                "between agents is informative**: when the Perceptual Agent says the player is searching "
                "but the Action Agent says they are active, that *tension* reveals something a single "
                "classifier would flatten into an ambiguous average."
            )
            st.markdown(
                "The motivation is that real cognitive states are **multidimensional** — a player can be "
                "visually focused but physically idle, or actively exploring but making no progress. "
                "A single-label system must pick one; Inside Out preserves the full pattern of agreement "
                "and contradiction across agents. The **negotiation layer** then maps these tension patterns "
                "to adaptive responses: consensus triggers intervention, disagreement triggers a probe "
                "(ask before assuming), and alignment means watch."
            )
            st.markdown("---")
            st.markdown(
                "Each agent reads an **exclusive** set of raw features and outputs a label + confidence. "
                "No raw feature is shared between agents — this prevents **echo consensus** "
                "(multiple agents agreeing because they read the same number, not because they see independent signals)."
            )

            st.markdown("---")

            # Agent cards
            agent_info = [
                {
                    "icon": "🔵",
                    "name": "Perceptual Agent",
                    "question": "What is the player looking at?",
                    "features": ["gaze_entropy", "clue_ratio", "switch_rate"],
                    "labels": "focused / searching / locked",
                    "description": "Interprets gaze distribution from eye tracking. High entropy + high switch rate = searching. Low entropy + high clue ratio = focused. Very high clue fixation + low entropy = locked on one spot.",
                    "color": "#2196F3",
                },
                {
                    "icon": "🟢",
                    "name": "Behavioral Agent",
                    "question": "What is the player doing right now?",
                    "features": ["action_count", "idle_time", "error_count"],
                    "labels": "active / hesitant / inactive / failing",
                    "description": "Interprets instantaneous behavior within each 5-second window. Its output label is passed to the Progress Agent via **label flow** (not raw features).",
                    "color": "#4CAF50",
                },
                {
                    "icon": "🟠",
                    "name": "Progress Agent",
                    "question": "Is the player making real progress?",
                    "features": ["time_since_action", "puzzle_elapsed_ratio", "behavioral_label (from Behavioral Agent)"],
                    "labels": "progressing / ineffective_progress / stalled",
                    "description": "Combines macro-temporal signals with Behavioral Agent's pre-interpreted label. Key innovation: **ineffective_progress** detects players who are active but not making meaningful progress — the #1 source of missed detections in earlier versions.",
                    "color": "#FF9800",
                },
                {
                    "icon": "⏱",
                    "name": "Temporal Agent",
                    "question": "How long has this state been going on?",
                    "features": ["Perceptual + Progress labels (sliding window)", "puzzle_elapsed_ratio"],
                    "labels": "transient / persistent / looping",
                    "description": "Meta-agent that reads labels from Perceptual and Progress agents over the last 3 windows. Detects if a state is temporary or has been repeating. Boosted by puzzle_elapsed_ratio: if a player is 3x over median puzzle time, flags as persistent even without label consistency.",
                    "color": "#9C27B0",
                },
                {
                    "icon": "👥",
                    "name": "Population Agent",
                    "question": "How does this player compare to others?",
                    "features": ["All features (compared to K=5 cluster centroids)"],
                    "labels": "exploring / disoriented / actively_solving / cognitively_stuck / transitioning",
                    "description": "Data-driven agent that compares the current window against cluster centroids learned from the full participant corpus (FDG '26 paper). Provides a population-relative perspective that can confirm or challenge rule-based agents.",
                    "color": "#795548",
                },
            ]

            for agent in agent_info:
                with st.container():
                    st.markdown(
                        f"### {agent['icon']} {agent['name']}\n"
                        f"**Question:** *{agent['question']}*\n\n"
                        f"**Exclusive features:** `{'`, `'.join(agent['features'])}`\n\n"
                        f"**Output labels:** {agent['labels']}\n\n"
                        f"{agent['description']}"
                    )
                    st.markdown("---")

            # Data flow diagram
            st.subheader("Data Flow: Label Flow Architecture")
            st.markdown(
                "```\n"
                "Eye Tracker ──→ Perceptual Agent ──→ label ──┐\n"
                "                                              │\n"
                "Controller ───→ Behavioral Agent ──→ label ──┤──→ Negotiation ──→ Support\n"
                "                      │                       │    (pairwise      (watch /\n"
                "                      └─ label flow ─→ Progress Agent  tensions)   probe /\n"
                "                                              │                  intervene)\n"
                "Game Engine ──→ Progress Agent ────→ label ──┤\n"
                "                                              │\n"
                "History ──────→ Temporal Agent ────→ label ──┤\n"
                "                                              │\n"
                "Cluster Data ─→ Population Agent ──→ label ──┘\n"
                "```"
            )

            st.markdown(
                "**Key design principle:** The Behavioral Agent outputs a pre-interpreted label "
                "(active/hesitant/inactive) which the Progress Agent consumes. The Progress Agent "
                "never reads raw `action_count` — it only knows *whether* the player is active, "
                "not *how many* actions they took. This ensures that when Behavioral and Progress "
                "agents agree, their agreement reflects genuinely independent evidence."
            )

            # Negotiation explanation
            st.subheader("Negotiation: How Agents Debate")
            st.markdown(
                "After all agents produce labels, the system checks every pair for **tensions**:\n\n"
                "- **Contradictory tension:** Perceptual says *searching* + Behavioral says *inactive* "
                "= `scanning_but_passive` (is the player exploring or lost?)\n"
                "- **Constructive tension:** Perceptual says *focused* + Progress says *progressing* "
                "= `focused_progress` (agents agree: player is doing well)\n\n"
                "The **pattern of tensions** — not any single agent — determines the system's response:\n\n"
                "| Pattern | Response | Meaning |\n"
                "|---------|----------|--------|\n"
                "| Agents agree: player is stuck | 🚨 **Intervene** | Give a hint |\n"
                "| Agents disagree on what's happening | 🔍 **Probe** | Explore before guessing |\n"
                "| Agents agree: player is OK | 👁 **Watch** | Don't interrupt |"
            )

            # Stateful prompt agent
            st.subheader("Stateful Prompt Agent")
            st.markdown(
                "The final decision passes through a **stateful** layer that tracks history per player:\n\n"
                "- **Cooldown:** No new intervention within 15s of the last one\n"
                "- **Escalation:** 6+ consecutive struggle windows → upgrade probe to intervene\n"
                "- **Recovery:** When player returns to normal → gradually reset escalation\n"
                "- **Fatigue:** Max 8 interventions per puzzle to avoid over-prompting"
            )

        with tab0:
            st.subheader("Who's Winning the Debate?")
            st.markdown(
                "Each line = one agent's confidence over time (smoothed). "
                "When lines cross, the **dominant interpretation shifts**. "
                "⭐ = system intervenes."
            )
            fig = make_dominance_line_chart(pdf)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Agent Dominance (Normalized)")
            st.markdown(
                "Stacked area view: the agent taking up the most space "
                "is **driving the system's interpretation** at that moment. "
                "▼ = intervention, ◆ = probe."
            )
            fig = make_dominance_chart(pdf)
            st.plotly_chart(fig, use_container_width=True)

        with tab1:
            st.subheader("Agent Confidence Over Time")
            st.markdown(
                "Each dot = one 5-second window. Height = agent confidence. "
                "Color = agent's interpretation. When agents are confident about "
                "**different things**, that's where negotiation matters most."
            )
            fig = make_agent_confidence_timeline(pdf)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Disagreement Intensity & System Responses")
            st.markdown(
                "**Red** = contradictory (agents disagree). "
                "**Green** = constructive (agents align). "
                "⭐ = system intervenes. ◆ = system probes."
            )
            fig = make_disagreement_timeline(pdf)
            st.plotly_chart(fig, use_container_width=True)

            # Summary
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Response Categories")
                if "support_category" in pdf.columns:
                    for cat in ["consensus_intervene", "probe", "watch"]:
                        n = (pdf["support_category"] == cat).sum()
                        pct = n / len(pdf) * 100
                        icon = CATEGORY_ICONS.get(cat, "")
                        st.markdown(f"{icon} **{cat}**: {n} ({pct:.0f}%)")

            with col2:
                st.subheader("Top Tensions")
                if "dominant_tension" in pdf.columns:
                    for tension, count in pdf["dominant_tension"].value_counts().head(5).items():
                        pct = count / len(pdf) * 100
                        st.markdown(f"**{tension}**: {count} ({pct:.0f}%)")

        with tab3:
            st.subheader("K-Means Cluster vs Agent Tensions")
            st.markdown(
                "Shows what the agents' negotiation reveals **within** each K-means cluster. "
                "If a single cluster maps to multiple tensions, classification loses information."
            )
            fig = make_cluster_vs_tension_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No cluster_id column found.")

            if "cluster_id" in df.columns:
                st.subheader("Within-Cluster Diversity")
                for cid in sorted(df["cluster_id"].unique()):
                    cdf = df[df["cluster_id"] == cid]
                    n_tensions = cdf["dominant_tension"].nunique()
                    top = cdf["dominant_tension"].value_counts()
                    dominant = top.index[0]
                    dominant_pct = top.values[0] / len(cdf) * 100
                    st.markdown(
                        f"**C{cid}**: {n_tensions} tension types, "
                        f"dominant = *{dominant}* ({dominant_pct:.0f}%)"
                    )

        with tab4:
            st.subheader("Step-by-Step Negotiation Playback")
            st.markdown("Scrub through the timeline to see agents debate each moment.")

            step = st.slider("Time Step", 0, len(pdf) - 1, 0, key="play")
            current = pdf.iloc[step]

            st.markdown(f"**Time:** {current['time_min']:.1f} min | "
                        f"**Puzzle:** {current.get('puzzle_id', '—')}")

            render_negotiation_panel(current)

    with tab5:
        st.subheader("🆚 Rule-Based System vs Inside Out")
        st.markdown(
            "Compares the rule-based prompting engine "
            "(designed from observing 18 players) with Inside Out's multi-agent negotiation. "
            "The rule-based system uses only game logs; Inside Out also uses eye tracking."
        )

        # --- Rule-Based System Logic ---
        with st.expander("How the Rule-Based System Works", expanded=False):
            st.markdown("""
#### State Machine (per puzzle)

Each puzzle has its own independent state machine that tracks the player's engagement:

```
InBetween ──[enter zone]──→ TriggerCheck ──[solvable]──→ Explore
                                 │                          │
                            [hub blocked]          [grab / gaze / 90s]
                                 │                          │
                            SpecialB prompt                 ▼
                            (every 30s)                  Solving
                                                        ↕     ↕
                                                 [engaged]  [3s idle]
                                                        ↕     ↕
                                                     Solving ← PossiblyStuck
                                                               │
                                                          [30s timeout]
                                                               │
                                                        Escalated Prompt
```

#### Prompt Escalation (per puzzle, never resets)

When a player stays in **PossiblyStuck** for 30 seconds without re-engaging, the system fires an escalated prompt:

| Prompt # | Type | Category | Description |
|----------|------|----------|-------------|
| 1–3 | **R** (Reflective) | probe | Open-ended: *"Maybe check the instructions again?"* |
| 4–5 | **V** (Vague) | probe | More specific: *"Something here might be useful"* |
| 6+ | **E** (Explicit) | intervene | Direct solution: *"Use the available clues to proceed"* |

Escalation counters are **per puzzle** and **never reset** — if a player leaves and returns to a puzzle, the count picks up where it left off.

#### Special Prompts (bypass escalation)

| Prompt | Trigger | Category |
|--------|---------|----------|
| **SpecialA** | Wandering for 300s (first) / 30s (repeat) without entering any zone | probe |
| **SpecialB** | In hub area but hub puzzle not yet solvable (need 4 spokes) | probe |
| **SpecialC** | In puzzle zone for 90s without grabbing anything | probe |

#### Key Differences from Inside Out

| Aspect | Rule-Based | Inside Out |
|--------|-----------|------------|
| **Input** | Game logs only (actions, timing) | Game logs + eye tracking (5 agents) |
| **Decision** | Binary idle detection (3s cooldown → 30s stuck) | Multi-agent negotiation with tension patterns |
| **Probe** | Only after entering PossiblyStuck | Detects uncertainty via agent disagreement |
| **Intervene** | Only after 5+ escalations on same puzzle | Consensus among agents that player needs help |
| **Granularity** | One state per puzzle | Rich tension patterns across all features |
""")

        if comp_df is None or len(comp_df) == 0:
            st.warning("No comparison data found. Run `python3 src/compare_systems.py` first.")
        else:
            # Filter to selected player
            comp_player = comp_df[comp_df["participant_id"] == selected_pid].sort_values("window_start").reset_index(drop=True)
            if selected_puzzle != "All" and "puzzle_id" in comp_player.columns:
                comp_player = comp_player[comp_player["puzzle_id"] == selected_puzzle].reset_index(drop=True)

            if len(comp_player) == 0:
                st.info("No comparison data for this player.")
            else:
                # Agreement stats
                agree = (comp_player["io_cat"] == comp_player["expert_cat"]).sum()
                st.metric("Agreement Rate", f"{agree/len(comp_player):.0%}", f"{agree}/{len(comp_player)} windows")

                # Side-by-side timeline
                st.subheader("Decision Timeline")
                st.markdown(
                    "Top row = rule-based decisions. Bottom row = Inside Out decisions. "
                    "**Red** = intervene, **Orange** = probe, **Green** = watch."
                )
                fig = make_comparison_timeline(comp_player)
                st.plotly_chart(fig, use_container_width=True)

                # Disagreement scatter
                st.subheader("Where Do They Disagree?")
                st.markdown(
                    "Each dot = one time window. "
                    "**Blue** = rule-based acts but IO doesn't. "
                    "**Red** = IO intervenes but rule-based doesn't. "
                    "**Orange** = IO probes but rule-based doesn't."
                )
                fig = make_disagreement_scatter(comp_player)
                st.plotly_chart(fig, use_container_width=True)

                # Cross-tabulation
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Decision Cross-Tab")
                    fig = make_agreement_heatmap(comp_player)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Key Differences")
                    n_both_intervene = ((comp_player["io_cat"] == "intervene") & (comp_player["expert_cat"] == "intervene")).sum()
                    n_both_probe = ((comp_player["io_cat"] == "probe") & (comp_player["expert_cat"] == "probe")).sum()
                    n_rb_active_io_watch = ((comp_player["expert_cat"] != "watch") & (comp_player["io_cat"] == "watch")).sum()
                    n_io_active_rb_watch = ((comp_player["io_cat"] != "watch") & (comp_player["expert_cat"] == "watch")).sum()
                    n_rb_probes = (comp_player["expert_cat"] == "probe").sum()
                    n_io_probes = (comp_player["io_cat"] == "probe").sum()

                    st.markdown(f"**Both intervene:** {n_both_intervene}")
                    st.markdown(f"**Both probe:** {n_both_probe}")
                    st.markdown(f"**RB active, IO watch:** {n_rb_active_io_watch}")
                    st.markdown(f"**IO active, RB watch:** {n_io_active_rb_watch}")
                    st.markdown(f"**RB probes:** {n_rb_probes} — R/V/Special prompts")
                    st.markdown(f"**IO probes:** {n_io_probes} — agent disagreement")

                # All-player summary
                st.subheader("All Players Summary")
                summary_rows = []
                for pid in sorted(comp_df["participant_id"].unique()):
                    pf = comp_df[comp_df["participant_id"] == pid]
                    summary_rows.append({
                        "Player": f"P{pid}",
                        "RB Probe": int((pf["expert_cat"] == "probe").sum()),
                        "RB Intervene": int((pf["expert_cat"] == "intervene").sum()),
                        "IO Probe": int((pf["io_cat"] == "probe").sum()),
                        "IO Intervene": int((pf["io_cat"] == "intervene").sum()),
                        "Agreement": f"{(pf['io_cat'] == pf['expert_cat']).mean():.0%}",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("🎯 Facilitator Benchmark")
        st.markdown(
            "Compares IO and Rule-Based system decisions against **real facilitator prompts** — "
            "the actual moments a human facilitator decided to intervene during gameplay. "
            "Reflective prompts map to **probe**, explicit prompts map to **intervene**."
        )

        if tw_df is None or len(tw_df) == 0:
            st.warning("No benchmark data found. Run `python3 src/facilitator_benchmark.py` first.")
        else:
            # --- Per-player three-way timeline ---
            tw_player = tw_df[tw_df["participant_id"] == selected_pid].sort_values("window_start").reset_index(drop=True)
            if selected_puzzle != "All" and "puzzle_id" in tw_player.columns:
                tw_player = tw_player[tw_player["puzzle_id"] == selected_puzzle].reset_index(drop=True)

            if len(tw_player) > 0:
                st.subheader("Three-Way Decision Timeline")
                st.markdown(
                    "Top = facilitator (ground truth), Middle = rule-based, Bottom = IO. "
                    "**Red** = intervene, **Orange** = probe, **Green** = watch."
                )
                fig = make_three_way_timeline(tw_player)
                st.plotly_chart(fig, use_container_width=True)

                # Quick stats for this player
                col1, col2, col3 = st.columns(3)
                fac_active = (tw_player["facilitator_cat"] != "watch").sum()
                io_active = (tw_player["io_cat"] != "watch").sum()
                ex_active = (tw_player["expert_cat"] != "watch").sum()
                col1.metric("Facilitator Active", f"{fac_active}/{len(tw_player)}", f"{fac_active/len(tw_player):.0%}")
                col2.metric("IO Active", f"{io_active}/{len(tw_player)}", f"{io_active/len(tw_player):.0%}")
                col3.metric("Rule-Based Active", f"{ex_active}/{len(tw_player)}", f"{ex_active/len(tw_player):.0%}")

            # --- Global results ---
            st.markdown("---")
            st.subheader("Temporal Tolerance Analysis (All 11 Players)")
            st.markdown(
                "Exact 5-second window matching underestimates performance because facilitator prompts "
                "span multiple windows. This analysis checks whether each system acted within ±N seconds "
                "of a real prompt. **Solid lines** = F1 score, **dashed** = recall."
            )
            st.info(
                "**Baselines:** Facilitator intervenes in 20.8% of windows. "
                "A **random classifier** at this rate achieves F1 = 0.208. "
                "An **always-intervene** system achieves F1 = 0.344. "
                "IO's F1 = 0.529 is **2.5× random** and **1.5× always-intervene**."
            )

            if tol_df is not None and len(tol_df) > 0:
                fig = make_tolerance_chart(tol_df)
                st.plotly_chart(fig, use_container_width=True)

                # Key numbers at ±15s
                tol15 = tol_df[tol_df["tolerance_sec"] == 15]
                if len(tol15) > 0:
                    r = tol15.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("IO Detection (±15s)", f"F1={r['io_f1']:.3f}",
                                f"Recall={r['io_recall']:.0%}, Precision={r['io_precision']:.0%}")
                    col2.metric("Rule-Based Detection (±15s)", f"F1={r['ex_f1']:.3f}",
                                f"Recall={r['ex_recall']:.0%}, Precision={r['ex_precision']:.0%}")
                    col3.metric("Random Baseline", "F1=0.208",
                                "Recall=20.8%, Precision=20.8%")

            # --- Per-puzzle detection ---
            if detail_df is not None and len(detail_df) > 0:
                st.subheader("Detection Rate by Puzzle (±15s)")
                st.markdown("How often each system detected a facilitator prompt, broken down by puzzle.")
                fig = make_puzzle_detection_chart(detail_df)
                st.plotly_chart(fig, use_container_width=True)

                # Reflective vs Explicit
                st.subheader("Detection by Prompt Type")
                col1, col2 = st.columns(2)
                for pt, col in [("reflective", col1), ("explicit", col2)]:
                    sub = detail_df[detail_df["prompt_type"] == pt]
                    if len(sub) == 0:
                        continue
                    io_r = sub["io_hit"].mean()
                    ex_r = sub["expert_hit"].mean()
                    col.markdown(f"**{pt.capitalize()} Prompts** ({len(sub)} total)")
                    col.markdown(f"- IO: **{io_r:.0%}** detected")
                    col.markdown(f"- Rule-Based: **{ex_r:.0%}** detected")

            # --- Cross-tab ---
            st.subheader("Facilitator vs IO Decision Matrix")
            fig = make_three_way_heatmap(tw_df)
            st.plotly_chart(fig, use_container_width=True)

            # --- Distribution comparison ---
            st.subheader("Decision Distribution Comparison")
            dist_data = []
            for cat in ["watch", "probe", "intervene"]:
                dist_data.append({
                    "Category": cat,
                    "IO": f"{(tw_df['io_cat']==cat).sum()} ({(tw_df['io_cat']==cat).mean():.1%})",
                    "Rule-Based": f"{(tw_df['expert_cat']==cat).sum()} ({(tw_df['expert_cat']==cat).mean():.1%})",
                    "Facilitator": f"{(tw_df['facilitator_cat']==cat).sum()} ({(tw_df['facilitator_cat']==cat).mean():.1%})",
                })
            st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)

            # --- Per-player summary ---
            st.subheader("Per-Player Benchmark Summary")
            st.markdown(
                "**Detection Rate** = when the facilitator gave a prompt, "
                "what percentage did each system detect? Higher = better."
            )
            player_rows = []
            for pid in sorted(tw_df["participant_id"].unique()):
                pf = tw_df[tw_df["participant_id"] == pid]
                fac_n = (pf["facilitator_cat"] != "watch").sum()
                io_n = (pf["io_cat"] != "watch").sum()
                ex_n = (pf["expert_cat"] != "watch").sum()
                # Detection rate: when facilitator prompted, did system detect?
                fac_active = pf[pf["facilitator_cat"] != "watch"]
                if len(fac_active) > 0:
                    io_detect = (fac_active["io_cat"] != "watch").mean()
                    ex_detect = (fac_active["expert_cat"] != "watch").mean()
                else:
                    io_detect = 0
                    ex_detect = 0
                player_rows.append({
                    "Player": f"P{pid}",
                    "Fac Prompts": fac_n,
                    "IO Active": io_n,
                    "RB Active": ex_n,
                    "IO Detection": f"{io_detect:.0%}",
                    "RB Detection": f"{ex_detect:.0%}",
                })
            st.dataframe(pd.DataFrame(player_rows), use_container_width=True, hide_index=True)

            # --- Trigger frequency analysis ---
            st.markdown("---")
            st.subheader("Trigger Frequency Analysis: IO vs Facilitator")

            io_total = (tw_df["io_cat"] != "watch").sum()
            fac_total = (tw_df["facilitator_cat"] != "watch").sum()
            io_probe = (tw_df["io_cat"] == "probe").sum()
            io_intervene = (tw_df["io_cat"] == "intervene").sum()
            fac_probe = (tw_df["facilitator_cat"] == "probe").sum()
            fac_intervene = (tw_df["facilitator_cat"] == "intervene").sum()
            n = len(tw_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Facilitator Active", f"{fac_total} ({fac_total/n:.1%})")
            col2.metric("IO Active", f"{io_total} ({io_total/n:.1%})",
                         f"{io_total/fac_total:.1f}x facilitator")
            col3.metric("Rule-Based Active", f"{(tw_df['expert_cat'] != 'watch').sum()} ({(tw_df['expert_cat'] != 'watch').mean():.1%})")

            st.markdown(
                f"**Probe (low-cost monitoring):** Facilitator reflective = {fac_probe} ({fac_probe/n:.1%}), "
                f"IO probe = {io_probe} ({io_probe/n:.1%}) — comparable.\n\n"
                f"**Intervene (disruptive):** Facilitator explicit = {fac_intervene} ({fac_intervene/n:.1%}), "
                f"IO intervene = {io_intervene} ({io_intervene/n:.1%}) — "
                f"IO triggers **{io_intervene/fac_intervene:.0f}x** more interventions than the facilitator."
            )

            st.warning(
                "**Interpretation:** IO's high recall (92.1%) comes with over-triggering on interventions. "
                "The facilitator is very restrained with explicit prompts (only 1.3% of windows), preferring "
                "reflective guidance. IO's probe rate is close to the facilitator's reflective rate, but its "
                "intervene threshold is much more aggressive. This is the primary target for future calibration: "
                "reducing intervene decisions while preserving detection sensitivity."
            )


    with tab_fw:
        st.subheader("🔮 Future Work: Theory-Partitioned Agents")

        st.markdown(
            "We explored an alternative architecture where agents share **all data** "
            "but interpret through different **cognitive theories** — more faithful to the "
            "Inside Out metaphor (Joy and Sadness see the same memory, but interpret differently)."
        )

        # Theory agents table
        st.markdown("### Four Theory Agents")
        theory_data = [
            {"Agent": "Attention Theory", "Theory": "Posner, Lavie",
             "Question": "Is attention appropriately allocated?",
             "States": "engaged / overloaded / fixated / decoupled"},
            {"Agent": "Self-Regulation", "Theory": "Zimmerman, Pintrich",
             "Question": "Is the player adjusting their strategy?",
             "States": "self_regulated / impulsive / disengaged / reflective"},
            {"Agent": "Flow Theory", "Theory": "Csikszentmihalyi",
             "Question": "Is the challenge-skill balance right?",
             "States": "flow / anxiety / frustration / boredom"},
            {"Agent": "Cognitive Load", "Theory": "Sweller",
             "Question": "Is working memory capacity exceeded?",
             "States": "manageable / overloaded / fragmented / automated"},
        ]
        st.dataframe(pd.DataFrame(theory_data), use_container_width=True, hide_index=True)

        # Same data, different interpretation example
        st.markdown("### Same Data, Different Interpretation")
        st.markdown("**Example window:** `action_count=3, time_since=90s, gaze_entropy=1.2`")
        interp_data = [
            {"Agent": "Attention Theory", "Label": "engaged",
             "Reasoning": "Moderate entropy, player is looking at relevant areas"},
            {"Agent": "Self-Regulation", "Label": "impulsive",
             "Reasoning": "3 actions after 90s gap = reactive trial-and-error, not strategic"},
            {"Agent": "Flow Theory", "Label": "anxiety",
             "Reasoning": "Active but time_since is high = challenge exceeding skill"},
            {"Agent": "Cognitive Load", "Label": "manageable",
             "Reasoning": "Low entropy + some actions + no errors = within capacity"},
        ]
        st.dataframe(pd.DataFrame(interp_data), use_container_width=True, hide_index=True)
        st.markdown(
            "Result: 2 agents say OK, 2 say problem → **contradictory tension** → "
            "system probes instead of guessing. This is genuine disagreement from "
            "independent theoretical frameworks, not echo consensus."
        )

        # Results comparison
        st.markdown("---")
        st.markdown("### Experiment Results (±15s tolerance)")

        col1, col2 = st.columns(2)
        col1.markdown("**V3 (Data-Partitioned, current)**")
        col1.metric("F1", "0.529")
        col1.metric("Recall", "92.1%")
        col1.metric("Precision", "37.1%")

        col2.markdown("**Theory-Partitioned (experiment)**")
        col2.metric("F1", "0.531", "+0.002")
        col2.metric("Recall", "88.7%", "-3.4pp")
        col2.metric("Precision", "37.9%", "+0.8pp")

        # Two ceilings
        st.markdown("---")
        st.subheader("Why Not a Bigger Improvement? Two Performance Ceilings")

        st.error(
            "**Ceiling 1: Feature Granularity**\n\n"
            "In 90%+ of missed windows, ALL four theory agents agreed the player was OK "
            "(engaged + self_regulated + flow + manageable). The features showed: "
            "`action_count=2.78` (active), `time_since=86s` (moderate), "
            "`elapsed_ratio=1.15` (near median).\n\n"
            "**The problem is not the theories — it's the data.** "
            "All features capture *how much* the player acts, not *whether those actions are meaningful*. "
            "`action_count=3` could be 3 correct puzzle interactions or 3 random clicks. "
            "No cognitive theory can distinguish these without richer signals:\n"
            "- Semantic action labels (correct vs. wrong object)\n"
            "- Gaze-action coupling (looked at clue then acted on it?)\n"
            "- Spatial trajectory (moving toward solution vs. wandering)"
        )

        st.warning(
            "**Ceiling 2: Ground Truth Noise**\n\n"
            "The facilitator's prompts include two types:\n"
            "1. **Reactive** — responding to observed struggle (detectable)\n"
            "2. **Proactive** — guiding a functioning player toward deeper understanding (undetectable)\n\n"
            "Example: Player 22 had `action_count=5.8, time_since=24s, elapsed_ratio=0.8` — "
            "performing well by every measure — but received 4 facilitator prompts. "
            "These were likely pedagogical guidance, not confusion detection. "
            "No automated system should be expected to detect proactive prompts."
        )

        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations for 80-Person Study")
        st.markdown(
            "1. **Richer features**: Add semantic action labels, gaze-action coupling, "
            "and spatial trajectory to break through the feature granularity ceiling\n"
            "2. **Ground truth refinement**: Tag facilitator prompts as reactive vs. proactive "
            "to eliminate noise ceiling\n"
            "3. **Dual architecture evaluation**: Run both V3 and theory-partitioned on same data "
            "to determine which generalizes better\n"
            "4. **Per-theory calibration**: With more data, identify which cognitive theory "
            "best predicts each *type* of confusion"
        )

        st.info(
            "**Branch:** `experiment/theory-partitioned-agents` — "
            "full implementation available for comparison. "
            "See `docs/future_work_theory_agents.md` for detailed analysis."
        )

        # --- Gaze-Focused Architecture (V4) ---
        st.markdown("---")
        st.subheader("Gaze-Focused Architecture: VR as Total Capture Environment")

        st.markdown(
            "VR is fundamentally a **total-capture simulator** — it records every gaze direction, "
            "head movement, hand position, and object interaction at **~71Hz**. A human facilitator "
            "observing the same player only sees macro behavior (idle, moving, interacting). "
            "The key question: **can richer eye-tracking features close the gap between IO and a human observer?**"
        )
        st.markdown(
            "We tested a V4 architecture that restructures agents around eye tracking, "
            "extracting **19 gaze features** from raw PlayerTracking data (~1.8M gaze samples across 11 users). "
            "3 out of 5 agents now read exclusively from eye tracking."
        )

        # --- Results first ---
        st.markdown("### Results (±15s tolerance)")
        col1, col2 = st.columns(2)
        col1.markdown("**V3 (1 gaze agent / 5)**")
        col1.metric("F1", "0.529")
        col1.metric("Recall", "92.1%")
        col1.metric("Precision", "37.1%")

        col2.markdown("**V4 (3 gaze agents / 5)**")
        col2.metric("F1", "0.517", "-0.012")
        col2.metric("Recall", "90.7%", "-1.4pp")
        col2.metric("Precision", "36.2%", "-0.9pp")

        st.markdown(
            "Nearly identical F1 confirms: **eye tracking alone carries most of the signal**. "
            "The multi-agent framework is robust across different agent configurations."
        )

        # --- Detailed agent documentation in expander ---
        with st.expander("V4 Agent Architecture — Detailed Documentation", expanded=False):
            st.markdown("""
#### Data Pipeline

Raw `PlayerTracking.csv` (~71Hz per user) is windowed into **5-second segments** matching the existing pipeline.
19 features are extracted per window from binocular gaze position, gaze direction, head pose, and gaze target hit names.

---

#### Agent 1: Fixation Agent

**Question:** *How is the player distributing their visual attention?*

**Exclusive features:**

| Feature | Source | What it measures |
|---------|--------|-----------------|
| `fixation_count` | Consecutive frames on same target | Number of distinct fixations in 5s window |
| `fixation_duration_mean` | Duration of each fixation | Average time spent per fixation (deep processing vs skimming) |
| `fixation_duration_max` | Longest single fixation | Whether gaze is "stuck" on one object |
| `fixation_duration_std` | Variance of fixation durations | Regular rhythm (low) vs mixed pattern (high) |
| `revisit_rate` | Proportion of fixations on previously viewed targets | Rechecking behavior — uncertainty signal |

**Output labels:**

| Label | Pattern | Interpretation |
|-------|---------|---------------|
| `focused` | Moderate count, long duration, low revisit | Sustained attention on task-relevant content |
| `scanning` | High count, short duration | Rapid visual search — exploring or lost |
| `locked` | Very long single fixation (>4s), few total | Gaze stuck on one object — processing or frozen |
| `revisiting` | High revisit rate | Repeatedly checking same targets — uncertainty or verification |

---

#### Agent 2: Gaze Semantics Agent

**Question:** *What is the player looking at?*

Uses the VR engine's `GazeTarget_ObjectName` raycast hit (364 unique objects across 11 users),
categorized into: **clue** (diaries, hints, instructions), **puzzle** (interactive objects, snap points),
**environment** (walls, floor, exterior), **other**.

**Exclusive features:**

| Feature | Source | What it measures |
|---------|--------|-----------------|
| `gaze_target_entropy` | Shannon entropy over target name distribution | Diversity of viewed objects (low = concentrated, high = scattered) |
| `clue_dwell` | Proportion of frames on clue objects | Time spent reading instructions/hints |
| `puzzle_dwell` | Proportion of frames on puzzle objects | Time spent looking at interactive elements |
| `env_dwell` | Proportion of frames on environment | Time spent looking at walls/floor (not task-relevant) |
| `puzzle_object_ratio` | (clue + puzzle) / total | Overall task-relevance of gaze |
| `n_unique_targets` | Count of distinct objects viewed | Breadth of visual exploration |

**Output labels:**

| Label | Pattern | Interpretation |
|-------|---------|---------------|
| `fixated_on_clue` | High clue_dwell (>35%), low entropy | Deep engagement with instructions |
| `task_focused` | High puzzle_object_ratio, moderate entropy | Looking at task-relevant objects |
| `environmental_scanning` | High env_dwell (>85%), many targets | Looking at walls/floor — navigating or lost |
| `unfocused` | High entropy + low puzzle_object_ratio | Scattered gaze with no task focus |

---

#### Agent 3: Gaze-Motor Agent

**Question:** *Is the player's gaze purposeful or passive?*

Analyzes the *motor dynamics* of eye movement — not what they look at, but *how* they look.

**Exclusive features:**

| Feature | Source | What it measures |
|---------|--------|-----------------|
| `gaze_head_coupling` | Correlation of gaze direction change vs head rotation change | Low = eyes explore independently (purposeful); High = eyes follow head (passive) |
| `saccade_amplitude_mean` | Mean angular change between consecutive gaze directions | Size of eye jumps — small (detail scanning) vs large (searching) |
| `saccade_amplitude_max` | Largest single angular change | Sudden large eye movement — surprise or reorientation |
| `gaze_dispersion` | Spatial spread of gaze hit points | Tight cluster (concentrated) vs wide spread (dispersed) |

**Output labels:**

| Label | Pattern | Interpretation |
|-------|---------|---------------|
| `purposeful` | Low coupling, moderate saccades | Eyes explore independently of head — active visual search |
| `passive_scanning` | High coupling (>0.30), low saccades | Eyes just follow head rotation — not actively looking |
| `erratic` | Very high saccade amplitude, high dispersion | Rapid random eye movements — overwhelmed or panicking |
| `concentrated` | Low dispersion, low saccades | Tight gaze focus — fixated on one area |

---

#### Agent 4: Behavioral Agent (Game Logs Only)

**Question:** *What is the player physically doing?*

Identical to V3 — uses only game interaction logs, no eye tracking.
Kept unchanged for **ablation**: comparing V4-full (all 5 agents) vs V4-gaze-only (agents 1-3 + temporal).

**Exclusive features:**

| Feature | Source | What it measures |
|---------|--------|-----------------|
| `action_count` | PuzzleLogs interactions per window | Physical actions in 5s |
| `idle_time` | Seconds without any interaction | Inactivity within window |
| `error_count` | Wrong moves per window | Mistakes |
| `time_since_action` | Seconds since last interaction (any window) | Macro-level inactivity |

**Output labels:** `active` / `inactive` / `hesitant` / `failing`

---

#### Agent 5: Temporal Agent

**Question:** *Is the current pattern new or has it been going on?*

Reads **labels** (not raw features) from agents 1-4 over the past 3 windows.
Detects persistence and looping.

**Output labels:**
- `transient` — pattern just appeared, may resolve on its own
- `persistent` — 2+ agents stable for 3+ windows
- `looping` — stuck-related labels repeating (e.g., `locked`, `inactive`, `passive_scanning`)

---

#### New Tension Patterns (Gaze-Specific)

These tensions are **only possible with rich gaze data** — a human facilitator or
game-log-only system cannot detect them:

| Tension | Agents | What it means | V4 Frequency |
|---------|--------|---------------|-------------|
| `acting_while_looking_away` | Semantics (env_scanning) vs Behavioral (active) | Hands busy but eyes on walls — blind trial-and-error | 856 (16.3%) |
| `focused_but_idle` | Fixation (focused) vs Behavioral (inactive) | Focused gaze but no physical action — thinking or stuck? | 630 (12.0%) |
| `uncertain_checking` | Fixation (revisiting) + Behavioral (hesitant) | Rechecking targets with tentative action — unsure | 615 (11.7%) |
| `watching_task_but_idle` | Semantics (task_focused) vs Behavioral (inactive) | Looking at puzzle objects but not touching — intimidated? | 553 (10.5%) |
| `frozen` | Fixation (locked) vs Behavioral (inactive) | Gaze stuck + body still — frozen, needs help | 286 (5.4%) |
| `reading_but_not_acting` | Semantics (fixated_on_clue) vs Behavioral (inactive) | Reading instructions but not acting — doesn't understand? | 29 (0.6%) |

A facilitator sees "the player is standing still" for all six patterns above.
Inside Out V4 sees six *different* states, each requiring a different response.
""")

        st.info(
            "**Branch:** `experiment/gaze-focused-agents` — "
            "full V4 implementation with 19 gaze features extraction pipeline."
        )

        # --- Gaze-Action Coupling ---
        st.markdown("---")
        st.subheader("Gaze-Action Coupling: Can Eye Tracking Reveal Action Quality?")

        st.markdown(
            "A key limitation identified earlier: `action_count=3` could mean 3 correct puzzle interactions "
            "or 3 random clicks. **No game-log feature can tell them apart.** "
            "We tested whether eye tracking data can break through this ceiling by asking: "
            "*what was the player looking at in the 3 seconds before each action?*"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Informed Actions", "15.9%", help="Looked at relevant clue/puzzle object before acting")
        col2.metric("Blind Actions", "76.7%", help="Looked at environment/unrelated before acting")
        col3.metric("Misguided Actions", "7.4%", help="Looked at wrong puzzle object before acting")

        with st.expander("Gaze-Action Coupling — Method and Findings", expanded=False):
            st.markdown("""
#### Method

For each of the **759 interactions** in PuzzleLogs (11 users), we extract the gaze targets
from PlayerTracking in the **3 seconds before** the action timestamp. Then classify:

| Classification | Pre-action gaze | Example |
|---------------|----------------|---------|
| **Informed** | Looked at the clue or puzzle object being interacted with | Read "Pasta Note" diary → grabbed pasta bowl |
| **Blind** | Looked at walls, floor, or unrelated objects | Stared at wall → grabbed random object |
| **Misguided** | Looked at a different puzzle object | Looked at Water puzzle → interacted with Protein puzzle |

#### Cross-tabulation with Outcome

| | RightMove | WrongMove | Total |
|---|---|---|---|
| **Informed** | 84 (69%) | 7 (6%) | 121 |
| **Blind** | 302 (52%) | 126 (22%) | 582 |
| **Misguided** | 32 (57%) | 11 (20%) | 56 |

Key findings:
- **Informed actions have 69% success rate** vs blind actions at 52% — looking before acting works
- **Blind + WrongMove = 126 cases** — the clearest "trial-and-error" signal
- But **blind actions still succeed 52% of the time** — players use spatial memory, not just gaze

#### Impact on Detection Performance

| Version | F1 (±15s) | Change |
|---------|-----------|--------|
| V4 gaze agents only | 0.517 | baseline |
| V4 + gaze-action coupling | 0.521 | +0.004 |

**Modest improvement (+0.004 F1)** because:
1. Only **10.7%** of 5-second windows contain any action — coupling features are zero for 89% of windows
2. The distribution is heavily skewed (76.7% blind) — not enough variance to be discriminative
3. Blind actions often succeed — "blind" does not always mean "struggling"

#### Why This Matters Despite Small F1 Gain

The finding that **76.7% of actions are blind** is itself a contribution — it reveals that:

- Players in VR rely heavily on **spatial memory and proprioception**, not visual confirmation
- The `action_count` feature in current systems (V3 and rule-based) fundamentally **cannot distinguish
  informed problem-solving from random trial-and-error**
- This distinction is **only possible because VR records both gaze and interaction simultaneously** —
  a human facilitator cannot see where the player was looking at the moment they grabbed an object

For the 80-person study, the coupling signal could become more powerful with:
- **Per-puzzle coupling rates** — some puzzles may show stronger informed/blind differences
- **Temporal coupling trends** — a player switching from informed to blind actions = frustration onset
- **Richer action semantics** from Unity (correct object vs wrong object) instead of just RightMove/WrongMove
""")

        st.info(
            "**Branch:** `experiment/gaze-focused-agents` — "
            "includes gaze-action coupling extraction and integration."
        )

        # --- Learnable Integration Weights ---
        st.markdown("---")
        st.subheader("Learnable Integration Weights: Neural Network Layer Analogy")

        st.markdown(
            "The current system already has a **layered structure** analogous to a neural network — "
            "but all weights between layers are hand-tuned, not learned from data."
        )

        st.markdown(
            "```\n"
            "Layer 1 (Input):       Raw features (8 values per 5s window)\n"
            "    ↓  rule-based, interpretable\n"
            "Layer 2 (Agents):      5 interpretations (label + confidence)\n"
            "    ↓  rule-based, interpretable\n"
            "Layer 3 (Negotiation): Tension type, intensity, confidence spread\n"
            "    ↓  LEARNABLE weights ← this is the only layer that changes\n"
            "Layer 4 (Decision):    P(watch), P(probe), P(intervene)\n"
            "```"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current (Hand-Tuned)**")
            st.code(
                "score = 0.30 * struggle\n"
                "      + 0.20 * temporal\n"
                "      + 0.15 * elapsed\n"
                "      + 0.15 * momentum\n"
                "      + 0.10 * pre_collapse\n"
                "# weights chosen by trial-and-error",
                language="python"
            )
        with col2:
            st.markdown("**Proposed (Learned)**")
            st.code(
                "score = W[0] * struggle\n"
                "      + W[1] * temporal\n"
                "      + W[2] * elapsed\n"
                "      + W[3] * momentum\n"
                "      + W[4] * pre_collapse\n"
                "# W learned from facilitator data",
                language="python"
            )

        st.success(
            "**Why this is NOT just an MLP:**\n\n"
            "A standard neural network takes raw features and outputs a decision — "
            "the intermediate representations are opaque. Our approach keeps **Layers 1-3 "
            "fully interpretable**: you can always trace a decision back to named agent labels "
            "and identified tensions. Only the **final weighting** (how to combine these "
            "interpretable signals) is learned from data.\n\n"
            "This preserves the core contribution — multi-agent negotiation with interpretable "
            "disagreement — while allowing data-driven optimization of how disagreements are resolved."
        )

        st.markdown(
            "**Requirements for implementation:**\n"
            "- 80-person study data (~25,000+ windows) for sufficient training examples\n"
            "- Leave-one-participant-out cross-validation\n"
            "- Comparison: hand-tuned vs learned vs end-to-end MLP baseline\n"
            "- This isolates the contribution of the multi-agent structure vs learned integration"
        )

        # --- 80-Person Study Plan ---
        st.markdown("---")
        st.subheader("80-Person Study: From Pilot to Main Evaluation")

        st.markdown(
            "The 18-person pilot (11 with eye tracking) served as a **formative evaluation** — "
            "iterating the system design, diagnosing failure modes, and calibrating agent thresholds. "
            "The planned 80-person study is the **main evaluation** that addresses the pilot's limitations."
        )

        st.markdown("### What 80 Participants Changes")
        change_data = [
            {"Pilot Limitation": "n=11 with eye tracking",
             "80-Person Study": "n=60+ with full eye tracking (assuming ~75% capture rate)",
             "Impact": "~25,000+ windows across 60+ independent participants; per-participant bootstrap for confidence intervals"},
            {"Pilot Limitation": "Rule-based STUCK rarely triggered",
             "80-Person Study": "More diverse player behaviors across 80 sessions",
             "Impact": "Full R→V→E escalation coverage; stronger baseline comparison"},
            {"Pilot Limitation": "No user outcome data",
             "80-Person Study": "Completion rate, time-to-solve, error recovery, post-task surveys",
             "Impact": "Can link IO decisions to actual learning/performance outcomes"},
            {"Pilot Limitation": "Single facilitator as ground truth",
             "80-Person Study": "Two facilitators independently observing (at least 20 overlapping sessions)",
             "Impact": "Inter-rater reliability (Cohen's kappa); validates ground truth quality"},
            {"Pilot Limitation": "Offline analysis only",
             "80-Person Study": "A/B deployment: 40 facilitator-only vs 40 IO-assisted",
             "Impact": "Causal evidence that IO improves facilitator decision-making"},
        ]
        st.dataframe(pd.DataFrame(change_data), use_container_width=True, hide_index=True)

        st.markdown("### Study Design Recommendations")
        st.markdown(
            "1. **Tag facilitator prompt intent** — At each prompt, facilitator marks: "
            "*reactive* (responding to observed struggle) vs *proactive* (pedagogical guidance). "
            "This separates detectable from undetectable prompts and eliminates the ground truth noise ceiling.\n\n"
            "2. **A/B condition** — 40 participants with facilitator only, 40 with IO assisting the facilitator. "
            "Even if IO only serves as a second opinion (not autonomous), this yields causal user outcome data.\n\n"
            "3. **Inter-rater reliability** — Two facilitators independently observe at least 20 sessions. "
            "Report Cohen's kappa to validate that facilitator prompts are a reliable ground truth.\n\n"
            "4. **Richer features** — Add semantic action labels (correct vs wrong object), "
            "gaze-action coupling (looked at clue then acted?), and spatial trajectory "
            "(toward solution vs wandering) to break through the feature granularity ceiling.\n\n"
            "5. **Paper structure** — 18-person pilot → formative evaluation (system design iteration). "
            "80-person study → main evaluation (validation). Two-phase design is standard at CHI."
        )


if __name__ == "__main__":
    main()
