"""Interactive dashboard — three-panel Streamlit companion.

Run: streamlit run app/dashboard.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.calibration import Parameters, h_sigma
from model.analytical import simulate_forward, find_optimal_alpha, compute_shadow_price_of_delay
from model.policy import compute_regime_comparison, compute_delay_cost_curve, score_domain, DEFAULT_DOMAINS
from model.evolutionary import simulate_evolution, sweep_sigma

st.set_page_config(page_title="Sovereignty Model — Interactive Companion", layout="wide")

COLORS = {
    "drift": "#b0a896", "destructive": "#c4756a", "designed": "#5a8a6a",
    "present": "#8B4049", "future": "#5a7a8a", "dependency": "#b89a3c",
}

st.title("Two Kinds of Sovereignty — Economic Model")
st.markdown("*Interactive companion to the formal model. Explore how allocation, constraint intensity, and window dynamics shape sovereignty trajectories.*")

# PDF download link
import pathlib
pdf_path = pathlib.Path(__file__).parent / "static" / "sovereignty_model.pdf"
if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download formal model specification (PDF)",
            data=f,
            file_name="sovereignty_model.pdf",
            mime="application/pdf",
        )

st.sidebar.header("Model Parameters")
D0 = st.sidebar.slider("Initial dependency D₀", 0.0, 1.0, 0.67, 0.01)
R = st.sidebar.slider("R&D intensity (% GDP)", 0.01, 0.06, 0.022, 0.001, format="%.3f")
T = st.sidebar.slider("Time horizon (years)", 10, 50, 30)
gamma = st.sidebar.slider("Window closure rate γ", 0.01, 0.3, 0.1, 0.01)
rho = st.sidebar.slider("Discount rate ρ", 0.01, 0.10, 0.03, 0.01)

params = Parameters(D0=D0, R=R, T=float(T), gamma=gamma, rho=rho)

tab1, tab2, tab3 = st.tabs(["Layer 1: Analytical Core", "Layer 2: Evolutionary Simulation", "Layer 3: Policy Interface"])

with tab1:
    st.header("Optimal Allocation & Constraint Design")
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Exploitation share α", 0.0, 1.0, 0.5, 0.01, key="alpha1")
    with col2:
        sigma = st.slider("Constraint intensity σ", 0.0, 15.0, 2.0, 0.1, key="sigma1")

    result = simulate_forward(alpha, sigma, params)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Present Sovereignty Kp", "Future Sovereignty Kf", "Dependency D"))
    fig.add_trace(go.Scatter(x=result.t, y=result.Kp, line=dict(color=COLORS["present"], width=2), name="Kp"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.t, y=result.Kf, line=dict(color=COLORS["future"], width=2), name="Kf"), row=1, col=2)
    fig.add_trace(go.Scatter(x=result.t, y=result.D, line=dict(color=COLORS["dependency"], width=2), name="D"), row=1, col=3)
    fig.update_layout(height=350, showlegend=False, margin=dict(t=40, b=20))
    fig.update_xaxes(title_text="Years")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Discounted Total Value V", f"{result.V_total:.4f}")

    st.subheader("Constraint Instrument h(σ)")
    sigmas_plot = np.linspace(0, 15, 200)
    h_vals = [h_sigma(s, params.a, params.b) for s in sigmas_plot]
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=sigmas_plot, y=h_vals, line=dict(color=COLORS["designed"], width=2)))
    fig_h.add_vline(x=sigma, line_dash="dash", line_color=COLORS["destructive"], annotation_text=f"Current σ={sigma}")
    fig_h.update_layout(height=300, xaxis_title="σ", yaxis_title="h(σ)", margin=dict(t=20, b=20))
    st.plotly_chart(fig_h, use_container_width=True)

with tab2:
    st.header("Selection Pressure Dynamics")
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Number of firms", 50, 500, 200, 50)
    with col2:
        seed = st.number_input("Random seed", value=42, step=1)

    sigma_vals = st.multiselect("σ values to compare", options=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0], default=[0.0, 2.0, 15.0])

    if st.button("Run Simulation", key="run_evo"):
        evo_params = Parameters(**{**params.__dict__, "N": N})
        fig_evo = go.Figure()
        sigma_color_map = {0.0: COLORS["drift"], 2.0: COLORS["designed"], 15.0: COLORS["destructive"]}
        for sv in sorted(sigma_vals):
            r = simulate_evolution(sv, evo_params, seed=int(seed))
            color = sigma_color_map.get(sv, "#666666")
            fig_evo.add_trace(go.Scatter(x=r.t, y=r.aggregate_capability, line=dict(color=color, width=2), name=f"σ = {sv}"))
        fig_evo.update_layout(height=400, xaxis_title="Years", yaxis_title="Aggregate Capability C(t)",
                              title="Evolutionary Dynamics Under Selection Pressure", margin=dict(t=40, b=20))
        st.plotly_chart(fig_evo, use_container_width=True)

with tab3:
    st.header("Policy Recommendations")

    st.subheader("Three-Regime Comparison")
    regimes = compute_regime_comparison(params)
    fig_reg = make_subplots(rows=1, cols=3, subplot_titles=("Present Sovereignty", "Future Sovereignty", "Dependency"))
    regime_colors = {"status_quo": COLORS["drift"], "designed": COLORS["designed"], "overreaction": COLORS["destructive"]}
    regime_labels = {"status_quo": "Status Quo", "designed": "Designed Pressure", "overreaction": "Overreaction"}
    for key, r in regimes.items():
        c = regime_colors[key]
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.Kp, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key), row=1, col=1)
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.Kf, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key, showlegend=False), row=1, col=2)
        fig_reg.add_trace(go.Scatter(x=r.t, y=r.D, line=dict(color=c, width=2), name=regime_labels[key], legendgroup=key, showlegend=False), row=1, col=3)
    fig_reg.update_layout(height=350, margin=dict(t=40, b=20))
    fig_reg.update_xaxes(title_text="Years")
    st.plotly_chart(fig_reg, use_container_width=True)

    cols = st.columns(3)
    for i, (key, r) in enumerate(regimes.items()):
        with cols[i]:
            st.metric(regime_labels[key], f"V = {r.V_total:.4f}")

    st.subheader("Shadow Price of Delay")
    delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=15.0, params=params)
    fig_delay = go.Figure()
    fig_delay.add_trace(go.Scatter(x=delays, y=costs, fill="tozeroy", fillcolor="rgba(139,64,73,0.15)", line=dict(color=COLORS["present"], width=2)))
    fig_delay.update_layout(height=300, xaxis_title="Delay (years)", yaxis_title="Cumulative cost", margin=dict(t=20, b=20))
    st.plotly_chart(fig_delay, use_container_width=True)

    st.subheader("Domain Priority Scoring")
    domain_data = []
    for name, assessment in DEFAULT_DOMAINS.items():
        s = score_domain(**assessment)
        domain_data.append({"Domain": name.replace("_", " ").title(), "Score": s, **assessment})
    for d in sorted(domain_data, key=lambda x: x["Score"], reverse=True):
        st.progress(d["Score"], text=f"**{d['Domain']}** — Score: {d['Score']:.2f} (Window: {d['W_remaining']:.1f}, Capability: {d['capability']:.1f}, Strategic Value: {d['strategic_value']:.1f})")
