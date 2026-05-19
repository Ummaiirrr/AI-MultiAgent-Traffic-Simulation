"""
app.py
======
MODULE 4: Streamlit Web Interface Dashboard
===========================================
AI Multi-Agent Traffic Simulation System — University Lab Final Project

Architecture:
  - Streamlit session_state is used to persist simulation and ML objects
    across re-renders (Streamlit re-runs the entire script on each interaction).
  - The simulation grid is rendered as a Plotly heatmap with scatter overlays
    for vehicle agents and their destinations.
  - ML evaluation results (mean ± std) are displayed in a styled metrics block.
  - All heavy computation (data generation, ML training) runs once and is
    cached in session_state to prevent unnecessary re-runs.

Run with:
    streamlit run app.py
"""

import time
import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---- Local modules ----------------------------------------------------------
# These imports work when all four .py files are in the same directory.
from data_pipeline import run_pipeline, OUTPUT_CLEAN
from traffic_predictor import run_full_evaluation, CLASS_NAMES, N_RUNS
from simulator import TrafficSimulation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Simulation System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [data-testid="stApp"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    h1, h2, h3 { color: #58a6ff; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* ---------- Metric Cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .metric-card .label {
        font-size: 12px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-card .std {
        font-size: 13px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* ---------- Status bar ---------- */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-green  { background: #1a472a; color: #56d364; }
    .badge-yellow { background: #3b2a12; color: #d29922; }
    .badge-red    { background: #4b1113; color: #f85149; }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 13px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 14px;
        margin-top: 10px;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover { background-color: #2ea043; }

    /* ---------- Code / report block ---------- */
    .report-block {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #c9d1d9;
        white-space: pre;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Initialise Streamlit session_state keys on first load."""
    defaults = {
        "sim"             : None,
        "sim_running"     : False,
        "sim_step"        : 0,
        "ml_results"      : None,
        "data_ready"      : False,
        "agent_history"   : [],    # list of congestion snapshots for chart
        "tick_stats"      : [],    # list of dicts from sim.tick()
        "model_type"      : "MLP",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ---------------------------------------------------------------------------
# Sidebar — Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚦 Control Panel")
    st.markdown("---")

    st.markdown('<div class="section-header">⚙️ Simulation Parameters</div>', unsafe_allow_html=True)
    grid_size       = st.slider("Grid Size (N×N)", 10, 30, 20, step=2)
    spawn_rate      = st.slider("Cars Spawned per Batch", 2, 20, 8)
    signal_cycle    = st.slider("Signal Cycle (steps)", 4, 20, 8)
    max_agents      = st.slider("Max Active Agents", 10, 60, 30)
    sim_speed_ms    = st.slider("Step Interval (ms)", 50, 1000, 200, step=50)

    st.markdown('<div class="section-header">🧠 ML Predictor</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model Type", ["MLP", "KNN"], index=0)
    st.session_state["model_type"] = model_choice

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        btn_data = st.button("📊 Generate Data", use_container_width=True)
    with col_b:
        btn_train = st.button("🤖 Train Model", use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        btn_start = st.button("▶ Start Sim", use_container_width=True)
    with col_d:
        btn_stop  = st.button("⏹ Stop Sim",  use_container_width=True)

    btn_reset = st.button("🔄 Reset Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e;'>AI Lab Final Project<br>"
        "Multi-Agent Traffic Simulation<br>"
        "UCI Metro Interstate Dataset</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Button Actions
# ---------------------------------------------------------------------------
if btn_data:
    with st.spinner("Generating & preprocessing 50,000-record dataset…"):
        run_pipeline()
        st.session_state["data_ready"] = True
    st.success("✅ Dataset generated and saved to traffic_clean.csv")

if btn_train:
    if not st.session_state["data_ready"] and not os.path.exists(OUTPUT_CLEAN):
        st.warning("⚠️ Please generate the dataset first.")
    else:
        with st.spinner(f"Training {model_choice} across {N_RUNS} independent splits…"):
            results = run_full_evaluation(model_type=model_choice)
            st.session_state["ml_results"] = results
        st.success(f"✅ {model_choice} evaluation complete.")

if btn_start:
    if st.session_state["sim"] is None or btn_reset:
        st.session_state["sim"]         = TrafficSimulation(
            rows=grid_size, cols=grid_size, signal_cycle=signal_cycle,
        )
        st.session_state["sim_step"]    = 0
        st.session_state["agent_history"] = []
        st.session_state["tick_stats"]  = []
    st.session_state["sim"].spawn_agents(spawn_rate)
    st.session_state["sim_running"] = True

if btn_stop:
    st.session_state["sim_running"] = False

if btn_reset:
    st.session_state["sim"]         = None
    st.session_state["sim_running"] = False
    st.session_state["sim_step"]    = 0
    st.session_state["agent_history"] = []
    st.session_state["tick_stats"]  = []


# ---------------------------------------------------------------------------
# Helper — Build Plotly Grid Figure
# ---------------------------------------------------------------------------
def build_grid_figure(sim: TrafficSimulation) -> go.Figure:
    """
    Render the simulation grid as a Plotly heatmap with:
      - Base layer: cell type (road / obstacle / signal state).
      - Congestion overlay: semi-transparent heatmap showing vehicle density.
      - Scatter layer: each agent as a coloured circle marker.
      - Destination markers: diamond shapes.

    Args:
        sim: Active TrafficSimulation instance.

    Returns:
        go.Figure ready for st.plotly_chart().
    """
    grid_state = sim.get_grid_state()
    grid_arr   = np.array(grid_state["grid"])
    cong_arr   = np.array(grid_state["congestion"])
    agents     = sim.get_agent_positions()
    rows, cols = grid_arr.shape

    # ---- Background: cell type heatmap (0=road, 1=obstacle, 2=red, 3=green) --
    bg_z = grid_arr.astype(float)
    bg_colorscale = [
        [0.00, "#1c2128"],   # road
        [0.33, "#3d1f1f"],   # obstacle
        [0.67, "#4b1113"],   # signal red
        [1.00, "#1a472a"],   # signal green
    ]

    fig = make_subplots(rows=1, cols=1)

    # Cell-type base
    fig.add_trace(go.Heatmap(
        z=bg_z,
        colorscale=bg_colorscale,
        showscale=False,
        zmin=0, zmax=3,
        opacity=1.0,
        hovertemplate="Cell type: %{z}<extra></extra>",
    ))

    # Congestion overlay
    if cong_arr.max() > 0:
        norm_cong = cong_arr / max(cong_arr.max(), 1)
        fig.add_trace(go.Heatmap(
            z=norm_cong,
            colorscale=[
                [0.0,  "rgba(0,0,0,0)"],
                [0.3,  "rgba(255,200,0,0.25)"],
                [0.7,  "rgba(255,100,0,0.45)"],
                [1.0,  "rgba(255,0,0,0.70)"],
            ],
            showscale=True,
            colorbar=dict(
                title="Congestion",
                title_font_color="#8b949e",
                tickfont_color="#8b949e",
                bgcolor="#161b22",
                bordercolor="#30363d",
                thickness=14,
                len=0.6,
            ),
            zmin=0, zmax=1,
            hovertemplate="Density: %{z:.2f}<extra></extra>",
        ))

    # Agent markers
    if agents:
        agent_rows  = [a["row"] for a in agents]
        agent_cols  = [a["col"] for a in agents]
        agent_colors = [a["color"] for a in agents]
        agent_text  = [f"Agent {a['agent_id']}<br>Steps: {a['total_steps']}" for a in agents]
        dest_rows   = [a["dest_row"] for a in agents]
        dest_cols   = [a["dest_col"] for a in agents]

        # Vehicles
        fig.add_trace(go.Scatter(
            x=agent_cols, y=agent_rows,
            mode="markers",
            marker=dict(
                color=agent_colors,
                size=10,
                symbol="circle",
                line=dict(color="white", width=1),
            ),
            text=agent_text,
            hovertemplate="%{text}<extra></extra>",
            name="Vehicles",
        ))

        # Destinations
        fig.add_trace(go.Scatter(
            x=dest_cols, y=dest_rows,
            mode="markers",
            marker=dict(
                color=agent_colors,
                size=7,
                symbol="diamond",
                opacity=0.6,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="Destination<extra></extra>",
            name="Destinations",
        ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, cols - 0.5],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.5, rows - 0.5],
            scaleanchor="x",
        ),
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font_color="#e6edf3",
        ),
        height=480,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper — Build Tick-Stats Line Chart
# ---------------------------------------------------------------------------
def build_stats_chart(tick_stats: list[dict]) -> go.Figure:
    """
    Plot active agent count and per-step arrivals over simulation time.

    Args:
        tick_stats: List of dicts from sim.tick().

    Returns:
        go.Figure.
    """
    if not tick_stats:
        return go.Figure()

    df = pd.DataFrame(tick_stats)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["n_agents"],
        mode="lines", name="Active Agents",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["n_arrived"],
        mode="lines", name="Arrivals/Step",
        line=dict(color="#56d364", width=2),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        title=dict(text="Simulation Timeline", font=dict(color="#8b949e", size=12)),
        xaxis=dict(showgrid=True, gridcolor="#21262d", title="Step"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper — ML metric chart (per-run bar chart)
# ---------------------------------------------------------------------------
def build_ml_bar_chart(metrics_per_run: dict, model_type: str) -> go.Figure:
    """
    Grouped bar chart of per-run metric values for all four metrics.

    Args:
        metrics_per_run: Dict metric → list of per-run floats.
        model_type: Label for chart title.

    Returns:
        go.Figure.
    """
    colors = {"accuracy": "#58a6ff", "precision": "#56d364",
              "recall": "#d29922", "f1": "#f85149"}
    runs = [f"Run {i}" for i in range(1, N_RUNS + 1)]
    fig  = go.Figure()
    for metric, vals in metrics_per_run.items():
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=runs,
            y=vals,
            marker_color=colors[metric],
            opacity=0.85,
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        title=dict(
            text=f"{model_type} — Per-Run Metrics ({N_RUNS} Splits)",
            font=dict(color="#8b949e", size=12),
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True, gridcolor="#21262d",
            range=[0, 1.05], title="Score",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='font-size:28px; margin-bottom:0;'>"
    "🚦 AI Multi-Agent Traffic Simulation System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e; font-size:13px; margin-top:4px;'>"
    "UCI Metro Interstate Dataset · A* Pathfinding · MLP / KNN Predictor · "
    "Real-Time Congestion Dashboard</p>",
    unsafe_allow_html=True,
)

tab_sim, tab_ml, tab_data = st.tabs(["🗺️ Simulation", "🧠 ML Evaluation", "📊 Dataset"])


# ============================================================
# TAB 1: SIMULATION
# ============================================================
with tab_sim:
    sim_col, info_col = st.columns([3, 1])

    with sim_col:
        st.markdown('<div class="section-header">City Grid — Real-Time Heatmap</div>',
                    unsafe_allow_html=True)
        grid_placeholder = st.empty()
        stats_placeholder = st.empty()

    with info_col:
        st.markdown('<div class="section-header">Live Statistics</div>',
                    unsafe_allow_html=True)
        kpi_step    = st.empty()
        kpi_agents  = st.empty()
        kpi_arrived = st.empty()

        st.markdown('<div class="section-header">Legend</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style='font-size:12px; color:#8b949e; line-height:2;'>
            🟫 <b>Road</b> — traversable cell<br>
            ⬛ <b>Obstacle</b> — building / wall<br>
            🔴 <b>Red Signal</b> — vehicles wait<br>
            🟢 <b>Green Signal</b> — pass through<br>
            🔵 <b>Vehicle</b> — active agent<br>
            💠 <b>Diamond</b> — agent destination<br>
            🔥 <b>Heat Overlay</b> — congestion density
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">A* Heuristic</div>',
                    unsafe_allow_html=True)
        st.latex(r"f(n) = g(n) + h(n)")
        st.latex(r"h(n) = d_{Manhattan} + \lambda \cdot \rho_{local}")
        st.markdown(
            "<small style='color:#8b949e;'>"
            "g(n): path cost so far<br>"
            "h(n): Manhattan dist + congestion penalty<br>"
            "λ=2.0 (congestion weight)</small>",
            unsafe_allow_html=True,
        )

    # ---- Simulation loop -----------------------------------------------------
    if st.session_state["sim_running"] and st.session_state["sim"] is not None:
        sim = st.session_state["sim"]

        # Spawn more cars if below threshold
        if len(sim.agents) < max_agents // 2:
            sim.spawn_agents(spawn_rate)

        # Tick the simulation
        tick_info = sim.tick()
        st.session_state["sim_step"]  = tick_info["step"]
        st.session_state["tick_stats"].append(tick_info)

        # Limit history length
        if len(st.session_state["tick_stats"]) > 300:
            st.session_state["tick_stats"] = st.session_state["tick_stats"][-300:]

        # Render grid
        fig_grid = build_grid_figure(sim)
        grid_placeholder.plotly_chart(fig_grid, use_container_width=True, key=f"grid_{tick_info['step']}")

        # Render stats chart
        stats_fig = build_stats_chart(st.session_state["tick_stats"])
        stats_placeholder.plotly_chart(stats_fig, use_container_width=True, key=f"stats_{tick_info['step']}")

        # KPIs
        kpi_step.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Simulation Step</div>"
            f"<div class='value'>{tick_info['step']}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        kpi_agents.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Active Agents</div>"
            f"<div class='value'>{tick_info['n_agents']}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        kpi_arrived.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>Arrived This Step</div>"
            f"<div class='value'>{tick_info['n_arrived']}</div>"
            f"</div>", unsafe_allow_html=True,
        )

        # Loop — re-run the script after a brief pause
        time.sleep(sim_speed_ms / 1000)
        st.rerun()

    elif st.session_state["sim"] is not None and not st.session_state["sim_running"]:
        # Show last static frame
        sim = st.session_state["sim"]
        fig_grid = build_grid_figure(sim)
        grid_placeholder.plotly_chart(fig_grid, use_container_width=True)
        if st.session_state["tick_stats"]:
            stats_placeholder.plotly_chart(
                build_stats_chart(st.session_state["tick_stats"]),
                use_container_width=True,
            )
        step = st.session_state["sim_step"]
        kpi_step.markdown(
            f"<div class='metric-card'><div class='label'>Simulation Step</div>"
            f"<div class='value'>{step}</div></div>", unsafe_allow_html=True,
        )
        kpi_agents.markdown(
            f"<div class='metric-card'><div class='label'>Active Agents</div>"
            f"<div class='value'>{len(sim.agents)}</div></div>", unsafe_allow_html=True,
        )
        kpi_arrived.markdown(
            "<div class='metric-card'><div class='label'>Status</div>"
            "<div class='value' style='font-size:16px; color:#8b949e;'>Paused</div></div>",
            unsafe_allow_html=True,
        )
    else:
        grid_placeholder.markdown(
            "<div style='text-align:center; padding:80px; color:#8b949e;'>"
            "<h3>▶ Press Start Sim to begin</h3>"
            "<p>Configure parameters in the sidebar, then click <b>▶ Start Sim</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 2: ML EVALUATION
# ============================================================
with tab_ml:
    if st.session_state["ml_results"] is None:
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#8b949e;'>"
            "<h3>🤖 No ML Results Yet</h3>"
            "<p>Click <b>📊 Generate Data</b> then <b>🤖 Train Model</b> in the sidebar.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        res        = st.session_state["ml_results"]
        means      = res["means"]
        stds       = res["stds"]
        model_type = st.session_state["model_type"]

        # ---- Grading-sheet metrics block ------------------------------------
        st.markdown('<div class="section-header">📋 Evaluation Results — Mean ± Std Dev</div>',
                    unsafe_allow_html=True)

        m_labels = {
            "accuracy" : "Accuracy",
            "precision": "Precision (macro)",
            "recall"   : "Recall (macro)",
            "f1"       : "F1-Score (macro)",
        }
        cols = st.columns(4)
        for idx, (key, label) in enumerate(m_labels.items()):
            with cols[idx]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>{label}</div>"
                    f"<div class='value'>{means[key]:.4f}</div>"
                    f"<div class='std'>± {stds[key]:.4f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Formatted grading-sheet text output ---------------------------
        st.markdown('<div class="section-header">📄 Grading-Sheet Formatted Output</div>',
                    unsafe_allow_html=True)
        report_lines = [
            f"Model: {model_type}  |  Runs: {N_RUNS}  |  Split: 80/20",
            "─" * 50,
        ]
        for key, label in m_labels.items():
            report_lines.append(f"  {label:<24}: {means[key]:.4f} ± {stds[key]:.4f}")
        report_lines.append("─" * 50)
        st.markdown(
            f"<div class='report-block'>{'<br>'.join(report_lines)}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Charts row ----------------------------------------------------
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            st.markdown('<div class="section-header">📈 Per-Run Metric Comparison</div>',
                        unsafe_allow_html=True)
            bar_fig = build_ml_bar_chart(res["metrics_per_run"], model_type)
            st.plotly_chart(bar_fig, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="section-header">🔢 Confusion Matrix (Final Run)</div>',
                        unsafe_allow_html=True)
            cm = np.array(res["last_cm"])
            cm_fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="True", color="Count"),
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                color_continuous_scale="Blues",
                text_auto=True,
            )
            cm_fig.update_layout(
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#e6edf3",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                coloraxis_colorbar=dict(
                    tickfont_color="#e6edf3",
                    title_font_color="#8b949e",
                ),
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        # ---- Full classification report ------------------------------------
        st.markdown('<div class="section-header">📊 Full Classification Report (Final Run)</div>',
                    unsafe_allow_html=True)
        report_html = res["classification_report"].replace("\n", "<br>").replace(" ", "&nbsp;")
        st.markdown(
            f"<div class='report-block'>{report_html}</div>",
            unsafe_allow_html=True,
        )

        # ---- Saved assets notice ------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        col_x, col_y = st.columns(2)
        with col_x:
            if res.get("cm_path") and os.path.exists(res["cm_path"]):
                with open(res["cm_path"], "rb") as f:
                    st.download_button(
                        "⬇ Download Confusion Matrix PNG",
                        data=f,
                        file_name="confusion_matrix.png",
                        mime="image/png",
                    )
        with col_y:
            if res.get("curve_path") and os.path.exists(res["curve_path"]):
                with open(res["curve_path"], "rb") as f:
                    st.download_button(
                        "⬇ Download Metric Curves PNG",
                        data=f,
                        file_name="training_curves.png",
                        mime="image/png",
                    )


# ============================================================
# TAB 3: DATASET
# ============================================================
with tab_data:
    st.markdown('<div class="section-header">📂 Dataset Overview</div>',
                unsafe_allow_html=True)

    if os.path.exists(OUTPUT_CLEAN):
        df = pd.read_csv(OUTPUT_CLEAN)

        # KPIs
        dk1, dk2, dk3, dk4 = st.columns(4)
        with dk1:
            st.markdown(
                f"<div class='metric-card'><div class='label'>Total Records</div>"
                f"<div class='value'>{len(df):,}</div></div>",
                unsafe_allow_html=True,
            )
        with dk2:
            st.markdown(
                f"<div class='metric-card'><div class='label'>Features</div>"
                f"<div class='value'>{df.shape[1] - 1}</div></div>",
                unsafe_allow_html=True,
            )
        with dk3:
            n_classes = df["Congestion_Level"].nunique()
            st.markdown(
                f"<div class='metric-card'><div class='label'>Target Classes</div>"
                f"<div class='value'>{n_classes}</div></div>",
                unsafe_allow_html=True,
            )
        with dk4:
            missing = df.isnull().sum().sum()
            st.markdown(
                f"<div class='metric-card'><div class='label'>Missing Values</div>"
                f"<div class='value'>{missing}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Class distribution chart
        dist_col, table_col = st.columns([2, 3])
        with dist_col:
            st.markdown('<div class="section-header">Target Class Distribution</div>',
                        unsafe_allow_html=True)
            class_counts = df["Congestion_Level"].value_counts().sort_index()
            dist_fig = go.Figure(go.Bar(
                x=[CLASS_NAMES[i] for i in class_counts.index],
                y=class_counts.values,
                marker_color=["#56d364", "#d29922", "#f85149"],
                text=class_counts.values,
                textposition="outside",
            ))
            dist_fig.update_layout(
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#e6edf3",
                margin=dict(l=0, r=0, t=20, b=0),
                height=240,
                yaxis=dict(showgrid=True, gridcolor="#21262d"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(dist_fig, use_container_width=True)

        with table_col:
            st.markdown('<div class="section-header">Sample Records (first 100)</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df.head(100),
                use_container_width=True,
                height=260,
            )

        # Feature descriptions
        st.markdown('<div class="section-header">Feature Descriptions</div>',
                    unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Feature"     : ["Hour", "Day_of_Week", "Month", "Temperature_K",
                             "Rain_mm", "Snow_mm", "Cloud_Coverage",
                             "Holiday_Encoded", "Weather_*", "Congestion_Level"],
            "Type"        : ["Numeric", "Numeric", "Numeric", "Numeric",
                             "Numeric", "Numeric", "Numeric",
                             "Encoded", "One-Hot", "Target (int)"],
            "Description" : [
                "Hour of day (0–23) — captures rush-hour patterns",
                "Day of week (0=Mon … 6=Sun)",
                "Calendar month (1–12) — seasonal signal",
                "Ambient temperature in Kelvin (scaled)",
                "Rainfall in mm/hour (clipped, scaled)",
                "Snowfall in mm/hour (clipped, scaled)",
                "Cloud coverage percentage (0–100, scaled)",
                "LabelEncoded holiday indicator (0=None, >0=holiday name)",
                "One-hot columns for 8 weather categories",
                "0=Free Flow (<2000 veh/hr) | 1=Moderate | 2=Heavy (>4500 veh/hr)",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            "<div style='text-align:center; padding:60px; color:#8b949e;'>"
            "<h3>📂 No Dataset Found</h3>"
            "<p>Click <b>📊 Generate Data</b> in the sidebar to create the dataset.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
