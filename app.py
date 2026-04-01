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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "outputs", "agent_outputs.csv")

AGENT_COLORS = {
    # Attention
    "focused": "#2196F3",
    "searching": "#FF9800",
    "locked": "#F44336",
    "ambiguous": "#BDBDBD",
    # Action
    "active": "#4CAF50",
    "hesitant": "#FFC107",
    "inactive": "#E91E63",
    "unknown": "#BDBDBD",
    # Performance
    "progressing": "#4CAF50",
    "stalled": "#FF5722",
    "failing": "#B71C1C",
    # Temporal
    "transient": "#90CAF9",
    "persistent": "#FF9800",
    "looping": "#F44336",
}

SUPPORT_COLORS = {
    "procedural_hint": "#F44336",
    "spatial_hint": "#FF5722",
    "reorientation_prompt": "#FF9800",
    "encouragement_and_spatial_hint": "#FFC107",
    "light_guidance": "#FFEB3B",
    "monitor": "#90CAF9",
    "wait": "#C8E6C9",
    "none": "#EEEEEE",
}

SUPPORT_ICONS = {
    "procedural_hint": "🔧",
    "spatial_hint": "🧭",
    "reorientation_prompt": "🔄",
    "encouragement_and_spatial_hint": "💡",
    "light_guidance": "👆",
    "monitor": "👁",
    "wait": "⏸",
    "none": "",
}

PATTERN_DESCRIPTIONS = {
    "focused_but_stuck": "Focused on clues but can't proceed → procedural hint",
    "searching_without_grounding": "Scanning environment without direction → spatial hint",
    "searching_and_hesitant": "Looking around, acting tentatively → encouragement + spatial hint",
    "searching_but_progressing": "Scattered attention but making progress → wait",
    "active_but_unguided": "Acting frequently but failing → light guidance",
    "productive_struggle": "Focused effort, temporarily stalled → wait",
    "locked_and_idle": "Fixated on clue area, doing nothing → reorientation",
    "progressing_but_ambiguous": "Making progress, attention unclear → monitor",
    "no_clear_pattern": "No strong signal from agents",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time_min"] = df["window_start"] / 60.0
    return df


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def make_agent_timeline(pdf, show_support=True):
    """Build a multi-row timeline showing agent states + support suggestions."""

    n_rows = 5 if show_support else 4
    row_titles = ["Attention Agent", "Action Agent", "Performance Agent", "Temporal Agent"]
    if show_support:
        row_titles.append("Support Suggestion")

    heights = [1] * 4 + ([1.2] if show_support else [])

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_titles=row_titles,
        row_heights=heights,
    )

    time = pdf["time_min"]

    # --- Agent state rows ---
    agent_rows = [
        ("attention_state", 1),
        ("action_state", 2),
        ("performance_state", 3),
        ("temporal_state", 4),
    ]

    for col, row_num in agent_rows:
        states = pdf[col].unique()
        for state in states:
            mask = pdf[col] == state
            color = AGENT_COLORS.get(state, "#BDBDBD")
            fig.add_trace(
                go.Scatter(
                    x=time[mask],
                    y=[state] * mask.sum(),
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="square"),
                    name=state,
                    showlegend=(row_num == 1),
                    hovertemplate=(
                        f"<b>{col.replace('_state','').title()} Agent</b>: {state}<br>"
                        "Time: %{x:.1f} min<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row_num, col=1,
            )

    # --- Support suggestion row ---
    if show_support:
        for support_type in pdf["suggested_support"].unique():
            mask = pdf["suggested_support"] == support_type
            if support_type == "none":
                continue
            color = SUPPORT_COLORS.get(support_type, "#BDBDBD")
            icon = SUPPORT_ICONS.get(support_type, "")
            fig.add_trace(
                go.Scatter(
                    x=time[mask],
                    y=[support_type] * mask.sum(),
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=color,
                        symbol="diamond",
                        line=dict(width=1, color="#333"),
                    ),
                    name=f"{icon} {support_type}",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>Support</b>: {icon} {support_type}<br>"
                        "Time: %{x:.1f} min<br>"
                        "Pattern: %{customdata}<br>"
                        "<extra></extra>"
                    ),
                    customdata=pdf.loc[mask, "disagreement_pattern"],
                ),
                row=n_rows, col=1,
            )

    fig.update_xaxes(title_text="Time (minutes)", row=n_rows, col=1)
    fig.update_layout(
        height=180 * n_rows,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    return fig


def make_disagreement_timeline(pdf):
    """Show disagreement score over time with pattern annotations."""
    fig = go.Figure()

    # Disagreement score line
    fig.add_trace(go.Scatter(
        x=pdf["time_min"],
        y=pdf["disagreement_score"],
        mode="lines+markers",
        marker=dict(size=4, color="#5C6BC0"),
        line=dict(width=1.5, color="#5C6BC0"),
        name="Disagreement Score",
        hovertemplate=(
            "Time: %{x:.1f} min<br>"
            "Score: %{y}<br>"
            "Pattern: %{customdata}<br>"
            "<extra></extra>"
        ),
        customdata=pdf["disagreement_pattern"],
    ))

    # Highlight intervention points
    interventions = pdf[~pdf["suggested_support"].isin(["none", "wait", "monitor"])]
    if len(interventions) > 0:
        fig.add_trace(go.Scatter(
            x=interventions["time_min"],
            y=interventions["disagreement_score"],
            mode="markers",
            marker=dict(size=12, color="#F44336", symbol="star", line=dict(width=1, color="#B71C1C")),
            name="Intervention Point",
            hovertemplate=(
                "<b>INTERVENTION</b><br>"
                "Time: %{x:.1f} min<br>"
                "Support: %{customdata}<br>"
                "<extra></extra>"
            ),
            customdata=interventions["suggested_support"],
        ))

    fig.update_layout(
        height=250,
        xaxis_title="Time (minutes)",
        yaxis_title="Disagreement Score",
        yaxis=dict(dtick=1, range=[0, 5]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_support_summary_chart(pdf):
    """Pie chart of support suggestions."""
    counts = pdf["suggested_support"].value_counts()
    colors = [SUPPORT_COLORS.get(s, "#BDBDBD") for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def make_cluster_vs_pattern_heatmap(pdf):
    """Cross-tabulation: K-means cluster vs agent disagreement pattern."""
    if "cluster_id" not in pdf.columns:
        return None

    ct = pd.crosstab(pdf["cluster_id"], pdf["disagreement_pattern"], normalize="index")
    ct = ct.round(3)

    cluster_labels = {
        0: "C0: Transition",
        1: "C1: Waiting",
        2: "C2: Active Solving",
        3: "C3: Exploration",
        4: "C4: Stuck-on-Clue",
    }
    y_labels = [cluster_labels.get(c, f"C{c}") for c in ct.index]

    fig = go.Figure(go.Heatmap(
        z=ct.values,
        x=ct.columns,
        y=y_labels,
        colorscale="YlOrRd",
        text=[[f"{v:.0%}" for v in row] for row in ct.values],
        texttemplate="%{text}",
        hovertemplate="Cluster: %{y}<br>Pattern: %{x}<br>Proportion: %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=350,
        xaxis_title="Agent Disagreement Pattern",
        yaxis_title="K-Means Cluster (FDG)",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(tickangle=30),
    )
    return fig


def make_playback_timeline(pdf, current_step):
    """Animated-style view: show agent states up to current_step."""
    window = pdf.iloc[:current_step + 1]
    current = pdf.iloc[current_step]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Attention", "Action", "Performance", "Temporal"],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    agents = [
        ("attention_state", 1, 1),
        ("action_state", 1, 2),
        ("performance_state", 2, 1),
        ("temporal_state", 2, 2),
    ]

    for col, r, c in agents:
        states = window[col].unique()
        for state in states:
            mask = window[col] == state
            color = AGENT_COLORS.get(state, "#BDBDBD")
            fig.add_trace(
                go.Scatter(
                    x=window["time_min"][mask],
                    y=[state] * mask.sum(),
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.4),
                    showlegend=False,
                ),
                row=r, col=c,
            )

        # Highlight current
        curr_state = current[col]
        curr_color = AGENT_COLORS.get(curr_state, "#BDBDBD")
        fig.add_trace(
            go.Scatter(
                x=[current["time_min"]],
                y=[curr_state],
                mode="markers",
                marker=dict(size=16, color=curr_color, symbol="star", line=dict(width=2, color="#000")),
                showlegend=False,
            ),
            row=r, col=c,
        )

    fig.update_layout(
        height=400,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Inside Out: Multi-Agent VR Cognitive State",
        layout="wide",
    )

    st.title("🧠 Inside Out")
    st.caption("Multi-Agent Negotiation of Cognitive States in VR Escape Room")

    df = load_data()

    # --- Sidebar ---
    st.sidebar.header("Controls")

    participants = sorted(df["participant_id"].unique())
    selected_pid = st.sidebar.selectbox(
        "Player",
        participants,
        format_func=lambda x: f"Player {x}",
    )

    pdf = df[df["participant_id"] == selected_pid].sort_values("window_start").reset_index(drop=True)

    puzzles = ["All"] + list(pdf["puzzle_id"].unique())
    selected_puzzle = st.sidebar.selectbox("Puzzle", puzzles)
    if selected_puzzle != "All":
        pdf = pdf[pdf["puzzle_id"] == selected_puzzle].reset_index(drop=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Windows:** {len(pdf)}")
    st.sidebar.markdown(f"**Duration:** {pdf['time_min'].max() - pdf['time_min'].min():.1f} min")

    n_interventions = pdf[~pdf["suggested_support"].isin(["none", "wait", "monitor"])].shape[0]
    st.sidebar.markdown(f"**Interventions:** {n_interventions}")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Agent Timeline",
        "⚡ Disagreement & Interventions",
        "🔬 Cluster vs Agent Analysis",
        "▶️ Playback",
    ])

    # --- Tab 1: Agent Timeline ---
    with tab1:
        st.subheader("Agent Interpretations Over Time")
        st.markdown(
            "Each row shows one agent's interpretation of the player's cognitive state. "
            "The bottom row shows when the system would intervene and how."
        )
        fig = make_agent_timeline(pdf)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Disagreement ---
    with tab2:
        st.subheader("Disagreement Score & Intervention Points")
        st.markdown(
            "Higher scores = more agents disagree. "
            "⭐ marks moments where the system would actively intervene."
        )
        fig = make_disagreement_timeline(pdf)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Support Distribution")
            fig = make_support_summary_chart(pdf)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Disagreement Patterns")
            pattern_counts = pdf["disagreement_pattern"].value_counts()
            for pattern, count in pattern_counts.items():
                desc = PATTERN_DESCRIPTIONS.get(pattern, "")
                pct = count / len(pdf) * 100
                st.markdown(f"**{pattern}** ({count}, {pct:.0f}%) — {desc}")

    # --- Tab 3: Cluster vs Agent ---
    with tab3:
        st.subheader("K-Means Cluster vs Agent Disagreement Pattern")
        st.markdown(
            "Shows how FDG's clustering (single-label) maps to multi-agent disagreement patterns. "
            "If a single cluster contains diverse patterns, it means classification loses information."
        )
        fig = make_cluster_vs_pattern_heatmap(df)  # use full dataset
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cluster_id column found in data.")

        # Per-cluster breakdown
        if "cluster_id" in df.columns:
            st.subheader("Within-Cluster Diversity")
            for cid in sorted(df["cluster_id"].unique()):
                cdf = df[df["cluster_id"] == cid]
                n_patterns = cdf["disagreement_pattern"].nunique()
                dominant = cdf["disagreement_pattern"].value_counts().index[0]
                dominant_pct = cdf["disagreement_pattern"].value_counts().values[0] / len(cdf) * 100
                st.markdown(
                    f"**C{cid}**: {n_patterns} distinct patterns, "
                    f"dominant = *{dominant}* ({dominant_pct:.0f}%)"
                )

    # --- Tab 4: Playback ---
    with tab4:
        st.subheader("Step-by-Step Playback")
        st.markdown("Scrub through the timeline to see how agents interpret each moment.")

        step = st.slider(
            "Time Step",
            min_value=0,
            max_value=len(pdf) - 1,
            value=0,
            key="playback_step",
        )

        current = pdf.iloc[step]

        # Current state summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", f"{current['time_min']:.1f} min")
            st.metric("Puzzle", current.get("puzzle_id", "—"))
        with col2:
            st.metric("Attention", current["attention_state"])
            st.metric("Action", current["action_state"])
        with col3:
            st.metric("Performance", current["performance_state"])
            st.metric("Temporal", current["temporal_state"])

        # Disagreement info
        pattern = current["disagreement_pattern"]
        support = current["suggested_support"]
        icon = SUPPORT_ICONS.get(support, "")

        if support not in ("none", "wait", "monitor"):
            st.error(f"**{icon} INTERVENE NOW:** {support} — Pattern: *{pattern}*")
        elif support == "monitor":
            st.info(f"**👁 MONITORING** — Pattern: *{pattern}*")
        elif support == "wait":
            st.success(f"**⏸ WAIT** — Pattern: *{pattern}*")
        else:
            st.markdown(f"No intervention — *{pattern}*")

        # Agent panel visualization
        fig = make_playback_timeline(pdf, step)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
