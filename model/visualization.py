"""Visualization functions for all three model layers.

Dual output: Matplotlib for static figures (paper), Plotly for
interactive (dashboard).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.facecolor": "#FAFAF7",
    "axes.facecolor": "#FAFAF7",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e8e6e1",
    "grid.alpha": 0.5,
})

from model.calibration import Parameters, h_sigma

COLORS = {
    "drift": "#b0a896",
    "destructive": "#c4756a",
    "designed": "#5a8a6a",
    "present": "#8B4049",
    "future": "#5a7a8a",
    "dependency": "#b89a3c",
    "status_quo": "#b0a896",
    "overreaction": "#c4756a",
}


def plot_hump_shape(sigma_max=15.0, params=None, simulated_data=None, save_path=None):
    if params is None:
        params = Parameters()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sigmas = np.linspace(0, sigma_max, 200)
    h_vals = [h_sigma(s, params.a, params.b) for s in sigmas]
    ax.plot(sigmas, h_vals, color=COLORS["designed"], linewidth=2, label="Analytical h(σ)")
    sigma_star = 1.0 / params.b
    h_star = h_sigma(sigma_star, params.a, params.b)
    ax.plot(sigma_star, h_star, "o", color=COLORS["designed"], markersize=8)
    ax.annotate(f"σ* = {sigma_star:.1f}", (sigma_star, h_star),
                xytext=(sigma_star + 1, h_star + 0.05), fontsize=10, color=COLORS["designed"])
    if simulated_data:
        sim_sigmas = sorted(simulated_data.keys())
        sim_means = [np.mean(simulated_data[s]) for s in sim_sigmas]
        sim_stds = [np.std(simulated_data[s]) for s in sim_sigmas]
        baseline = sim_means[0] if sim_means[0] > 0 else 1.0
        sim_h = [m / baseline for m in sim_means]
        sim_h_err = [s / baseline for s in sim_stds]
        ax.errorbar(sim_sigmas, sim_h, yerr=sim_h_err, fmt="s", color=COLORS["future"],
                    markersize=6, capsize=3, label="Simulated (Layer 2)")
    ax.axvspan(0, 0.3, alpha=0.05, color=COLORS["drift"])
    ax.axvspan(sigma_star - 1, sigma_star + 1, alpha=0.08, color=COLORS["designed"])
    ax.axvspan(sigma_max - 3, sigma_max, alpha=0.05, color=COLORS["destructive"])
    ax.text(0.15, 0.97, "Absent", transform=ax.transAxes, fontsize=9, color=COLORS["drift"], va="top")
    ax.text(0.38, 0.97, "Designed", transform=ax.transAxes, fontsize=9, color=COLORS["designed"], va="top")
    ax.text(0.8, 0.97, "Destructive", transform=ax.transAxes, fontsize=9, color=COLORS["destructive"], va="top")
    ax.set_xlabel("Constraint intensity σ")
    ax.set_ylabel("Exploration productivity h(σ)")
    ax.set_title("The Hump Shape: Constraint and Innovation")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regime_comparison(regimes, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    labels = {
        "status_quo": ("Status Quo (Presentism)", COLORS["status_quo"]),
        "designed": ("Designed Pressure", COLORS["designed"]),
        "overreaction": ("Overreaction", COLORS["overreaction"]),
    }
    ax = axes[0]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.Kp, color=color, linewidth=2, label=label)
    ax.set_title("Present Sovereignty Kp")
    ax.set_xlabel("Years")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.5)

    ax = axes[1]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.Kf, color=color, linewidth=2, label=label)
    ax.set_title("Future Sovereignty Kf")
    ax.set_xlabel("Years")
    ax.grid(True, linewidth=0.5)

    ax = axes[2]
    for key, result in regimes.items():
        label, color = labels[key]
        ax.plot(result.t, result.D, color=color, linewidth=2, label=label)
    ax.set_title("Dependency D")
    ax.set_xlabel("Years")
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_delay_costs(delays, costs, domain_name="Default", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(delays, 0, costs, alpha=0.15, color=COLORS["present"])
    ax.plot(delays, costs, color=COLORS["present"], linewidth=2)
    ax.set_xlabel("Delay (years)")
    ax.set_ylabel("Cumulative cost of delay")
    ax.set_title(f"Shadow Price of Delay — {domain_name}")
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_evolutionary_trajectories(results_by_sigma, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sigma_colors = {0.0: COLORS["drift"], 2.0: COLORS["designed"], 15.0: COLORS["destructive"]}
    sigma_labels = {0.0: "σ = 0 (No pressure)", 2.0: "σ = 2 (Designed)", 15.0: "σ = 15 (Destructive)"}
    for sigma, result in sorted(results_by_sigma.items()):
        color = sigma_colors.get(sigma, "#666666")
        label = sigma_labels.get(sigma, f"σ = {sigma}")
        ax.plot(result.t, result.aggregate_capability, color=color, linewidth=2, label=label)
    ax.set_xlabel("Years")
    ax.set_ylabel("Aggregate Capability C(t)")
    ax.set_title("Evolutionary Dynamics Under Selection Pressure")
    ax.legend(framealpha=0.9)
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_optimal_alpha(alpha_result, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alpha_result.alpha_grid, alpha_result.V_grid, color=COLORS["future"], linewidth=2)
    ax.axvline(alpha_result.alpha_star, color=COLORS["designed"], linestyle="--", linewidth=1.5)
    ax.plot(alpha_result.alpha_star, alpha_result.V_star, "o", color=COLORS["designed"], markersize=8)
    ax.annotate(f"α* = {alpha_result.alpha_star:.2f}",
                (alpha_result.alpha_star, alpha_result.V_star),
                xytext=(alpha_result.alpha_star + 0.05, alpha_result.V_star),
                fontsize=10, color=COLORS["designed"])
    ax.set_xlabel("Allocation α (exploitation share)")
    ax.set_ylabel("Discounted total value V")
    ax.set_title("Optimal Sovereignty Allocation")
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_firm_distribution(capability_history, time_points=None, save_path=None):
    if time_points is None:
        n = len(capability_history)
        time_points = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    fig, axes = plt.subplots(1, len(time_points), figsize=(3 * len(time_points), 3.5), sharey=True)
    for ax, t_idx in zip(axes, time_points):
        caps = capability_history[t_idx]
        ax.hist(caps, bins=30, color=COLORS["designed"], alpha=0.7, edgecolor="white")
        ax.set_title(f"t = {t_idx}", fontsize=10)
        ax.set_xlabel("Capability")
        if t_idx == time_points[0]:
            ax.set_ylabel("Count")
    plt.suptitle("Capability Distribution Over Time", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
