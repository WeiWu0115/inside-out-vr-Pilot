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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

DATA_PATH = os.path.join(os.path.dirname(__file__), "outputs", "agent_outputs.csv")

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

AGENT_LABEL_COLORS = {
    "focused": "#2196F3", "searching": "#FF9800", "locked": "#F44336", "ambiguous": "#BDBDBD",
    "active": "#4CAF50", "hesitant": "#FFC107", "inactive": "#E91E63", "unknown": "#BDBDBD",
    "progressing": "#4CAF50", "stalled": "#FF5722", "failing": "#B71C1C",
    "transient": "#90CAF9", "persistent": "#FF9800", "looping": "#F44336",
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
    for agent in ["attention", "action", "performance", "temporal"]:
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


def make_agent_confidence_timeline(pdf):
    """Show agent confidence over time with color = label."""
    agents = [
        ("attention", "Attention Agent"),
        ("action", "Action Agent"),
        ("performance", "Performance Agent"),
        ("temporal", "Temporal Agent"),
    ]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
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

    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_layout(
        height=600, template="plotly_white",
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
        ("attention", "🔵 Attention"),
        ("action", "🟢 Action"),
        ("performance", "🟠 Performance"),
        ("temporal", "⏱ Temporal"),
    ]

    # Agent cards
    cols = st.columns(4)
    for col, (agent, icon) in zip(cols, agents):
        label = current.get(f"{agent}_label", "unknown")
        conf = current.get(f"{agent}_confidence", 0.0)
        reasoning = current.get(f"{agent}_reasoning", "")
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
    d_type = current.get("disagreement_type", "unstructured")
    d_intensity = current.get("disagreement_intensity", 0)
    tension = current.get("dominant_tension", "none")
    support = current.get("suggested_support", "none")
    s_conf = current.get("support_confidence", 0)
    rationale = current.get("support_rationale", "")
    category = current.get("support_category", "watch")

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Inside Out: Multi-Agent VR", layout="wide")
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
    n_intervene = pdf[pdf.get("support_category", pd.Series()) == "consensus_intervene"].shape[0] if "support_category" in pdf.columns else 0
    n_probe = pdf[pdf.get("support_category", pd.Series()) == "probe"].shape[0] if "support_category" in pdf.columns else 0
    st.sidebar.markdown(f"**Windows:** {len(pdf)}")
    st.sidebar.markdown(f"**Duration:** {pdf['time_min'].max() - pdf['time_min'].min():.1f} min")
    st.sidebar.markdown(f"**🚨 Interventions:** {n_intervene}")
    st.sidebar.markdown(f"**🔍 Probes:** {n_probe}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Agent Confidence",
        "⚡ Negotiation Timeline",
        "🔬 Cluster vs Agents",
        "▶️ Playback",
    ])

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


if __name__ == "__main__":
    main()
