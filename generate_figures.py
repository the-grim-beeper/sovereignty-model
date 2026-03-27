"""Generate all static figures and run cross-layer validation.

Run: python generate_figures.py
"""
import numpy as np
from pathlib import Path

from model.calibration import Parameters
from model.analytical import simulate_forward, find_optimal_alpha
from model.evolutionary import simulate_evolution, sweep_sigma
from model.policy import compute_regime_comparison, compute_delay_cost_curve
from model.visualization import (
    plot_hump_shape,
    plot_regime_comparison,
    plot_delay_costs,
    plot_evolutionary_trajectories,
    plot_optimal_alpha,
    plot_firm_distribution,
)

OUT = Path("output/figures")
OUT.mkdir(parents=True, exist_ok=True)

params = Parameters()

print("=== Generating static figures ===\n")

# 1. Hump shape (analytical only)
print("1. Hump shape...")
plot_hump_shape(save_path=str(OUT / "hump_shape_analytical.png"), params=params)

# 2. Optimal α
print("2. Optimal allocation...")
opt = find_optimal_alpha(sigma=2.0, params=params)
plot_optimal_alpha(opt, save_path=str(OUT / "optimal_alpha.png"))
print(f"   α* = {opt.alpha_star:.3f}, V* = {opt.V_star:.4f}")

# 3. Regime comparison
print("3. Regime comparison...")
regimes = compute_regime_comparison(params)
plot_regime_comparison(regimes, save_path=str(OUT / "regime_comparison.png"))
for name, r in regimes.items():
    print(f"   {name}: V = {r.V_total:.4f}")

# 4. Delay cost curve
print("4. Delay costs...")
delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=15.0, params=params)
plot_delay_costs(delays, costs, save_path=str(OUT / "delay_costs.png"))

# 5. Evolutionary trajectories
print("5. Evolutionary trajectories...")
evo_results = {}
for sigma in [0.0, 2.0, 15.0]:
    evo_results[sigma] = simulate_evolution(sigma, params, seed=42)
plot_evolutionary_trajectories(evo_results, save_path=str(OUT / "evolutionary_trajectories.png"))

# 5b. Firm distribution
print("5b. Firm distribution...")
r_designed = simulate_evolution(2.0, params, seed=42)
plot_firm_distribution(r_designed.capability_history, save_path=str(OUT / "firm_distribution.png"))

# 6. Cross-layer validation: h(σ) simulated vs analytical
print("6. Cross-layer h(σ) validation...")
sweep_sigmas = np.linspace(0, 12, 13).tolist()
sweep_results = sweep_sigma(sweep_sigmas, params, n_replications=20, seed=42)
plot_hump_shape(
    params=params,
    simulated_data=sweep_results,
    save_path=str(OUT / "hump_shape_validated.png"),
)

means = [np.mean(sweep_results[s]) for s in sweep_sigmas]
baseline = means[0]
norm_means = [m / baseline for m in means]
peak_idx = np.argmax(norm_means)
print(f"   Simulated peak at σ ≈ {sweep_sigmas[peak_idx]:.1f}")
print(f"   Analytical peak at σ* = {1.0 / params.b:.1f}")

print(f"\n=== All figures saved to {OUT} ===")
