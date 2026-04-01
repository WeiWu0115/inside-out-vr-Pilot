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
        ("attention", "Attention", "#2196F3"),
        ("action", "Action", "#4CAF50"),
        ("performance", "Performance", "#FF9800"),
        ("temporal", "Temporal", "#9C27B0"),
    ]

    time = pdf["time_min"].values

    # Collect raw confidences
    raw = {}
    for agent, _, _ in agents:
        raw[agent] = pdf[f"{agent}_confidence"].fillna(0).values

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
        ("attention", "Attention", "#2196F3"),
        ("action", "Action", "#4CAF50"),
        ("performance", "Performance", "#FF9800"),
        ("temporal", "Temporal", "#9C27B0"),
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
    n_intervene = (pdf["support_category"] == "consensus_intervene").sum() if "support_category" in pdf.columns else 0
    n_probe = (pdf["support_category"] == "probe").sum() if "support_category" in pdf.columns else 0
    st.sidebar.markdown(f"**Windows:** {len(pdf)}")
    st.sidebar.markdown(f"**Duration:** {pdf['time_min'].max() - pdf['time_min'].min():.1f} min")
    st.sidebar.markdown(f"**🚨 Interventions:** {n_intervene}")
    st.sidebar.markdown(f"**🔍 Probes:** {n_probe}")

    # Tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🏔 Agent Dominance",
        "📊 Agent Confidence",
        "⚡ Negotiation Timeline",
        "🔬 Cluster vs Agents",
        "▶️ Playback",
    ])

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


if __name__ == "__main__":
    main()
