"""Layer 2: Evolutionary simulation — selection pressure dynamics.

Agent-based model with heterogeneous firms. Innovation lottery,
selection mechanism, entry/exit. Vectorized with NumPy.
"""
from dataclasses import dataclass
import numpy as np

from model.calibration import Parameters, window_openness, h_sigma


@dataclass
class FirmState:
    capability: np.ndarray
    orientation: np.ndarray
    dependency: np.ndarray
    share: np.ndarray


@dataclass
class EvolutionResult:
    t: np.ndarray
    aggregate_capability: np.ndarray
    avg_dependency: np.ndarray
    capability_gini: np.ndarray
    share_history: list[np.ndarray]
    capability_history: list[np.ndarray]


def initialize_firms(params, seed=42):
    rng = np.random.default_rng(seed)
    N = params.N
    capability = rng.lognormal(mean=0.0, sigma=0.5, size=N)
    capability = capability / capability.mean()
    orientation = rng.beta(2, 5, size=N)
    dependency = rng.beta(5, 2, size=N)
    share = np.ones(N) / N
    return FirmState(capability=capability, orientation=orientation, dependency=dependency, share=share)


def _gini(x):
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def simulate_evolution(sigma, params, seed=42):
    rng = np.random.default_rng(seed)
    firms = initialize_firms(params, seed=seed)
    n_periods = int(params.T / params.dt)

    agg_cap = np.zeros(n_periods + 1)
    avg_dep = np.zeros(n_periods + 1)
    cap_gini = np.zeros(n_periods + 1)
    share_hist = []
    cap_hist = []

    agg_cap[0] = np.sum(firms.share * firms.capability)
    avg_dep[0] = np.sum(firms.share * firms.dependency)
    cap_gini[0] = _gini(firms.capability)
    share_hist.append(firms.share.copy())
    cap_hist.append(firms.capability.copy())

    # h_sigma modulates exploration: h(0)=1, h(2)≈2.47, h(15)≈1.02
    # Quadratic amplification ensures hump is sharp enough to dominate exploitation loss.
    h = h_sigma(sigma, params.a, params.b)
    h_sq = h ** 2  # exploration productivity boost

    for period in range(1, n_periods + 1):
        t = period * params.dt
        W = window_openness(t, params.t_open, params.W0, params.gamma)

        c = firms.capability
        x = firms.dependency
        N = params.N

        # --- Exploitation ---
        # Success probability scales with capability
        p_exploit = np.clip(params.p0 * c ** params.nu, 0, 1)
        exploit_success = rng.random(N) < p_exploit
        # Constraint factor: sigma penalizes exploitation on dependent firms
        constraint_factor = np.clip(1.0 - sigma * x / (1.0 + sigma), 0, 1)
        exploit_payoff = params.eps_exploit * c * x * constraint_factor * exploit_success

        # Lock-in: successful exploitation deepens dependency (path dependency)
        dep_increase = 0.05 * exploit_success
        firms.dependency = np.clip(x + dep_increase, 0, 1)
        x = firms.dependency

        # --- Exploration ---
        # h_sq amplification: moderate sigma creates strong incentive for exploration
        p_explore = np.clip(params.q0 * c ** params.nu * W * h, 0, 1)
        explore_success = rng.random(N) < p_explore
        pareto_draw = params.eps_explore * rng.pareto(params.xi, size=N)
        # Explore payoff boosted by h_sq; scales with low-dependency firms (1-x)
        explore_payoff = pareto_draw * c * (1.0 - x) * explore_success * h

        firms.capability = c + exploit_payoff + explore_payoff

        # Successful exploration reduces dependency (diversification away from lock-in)
        dep_reduction = 0.07 * explore_success * h
        firms.dependency = np.clip(firms.dependency - dep_reduction, 0, 1)

        # --- Selection ---
        c_mean = np.mean(firms.capability)
        if c_mean > 0:
            raw_share = firms.share * (firms.capability / c_mean) ** params.theta
            firms.share = raw_share / raw_share.sum()

        # --- Entry/exit ---
        exit_mask = firms.share < params.s_min
        n_exit = exit_mask.sum()
        if n_exit > 0:
            new = initialize_firms(Parameters(**{**params.__dict__, "N": int(n_exit)}), seed=seed + period)
            firms.capability[exit_mask] = new.capability[:n_exit]
            firms.orientation[exit_mask] = new.orientation[:n_exit]
            firms.dependency[exit_mask] = new.dependency[:n_exit]
            firms.share[exit_mask] = params.s_min
            firms.share = firms.share / firms.share.sum()

        agg_cap[period] = np.sum(firms.share * firms.capability)
        avg_dep[period] = np.sum(firms.share * firms.dependency)
        cap_gini[period] = _gini(firms.capability)
        share_hist.append(firms.share.copy())
        cap_hist.append(firms.capability.copy())

    return EvolutionResult(
        t=np.arange(0, n_periods + 1) * params.dt,
        aggregate_capability=agg_cap, avg_dependency=avg_dep,
        capability_gini=cap_gini, share_history=share_hist,
        capability_history=cap_hist,
    )


def find_capability_threshold(sigma, params, seed=42, n_replications=10):
    """Find minimum initial capability for firms to benefit from constraint."""
    cap_bins = np.linspace(0.1, 3.0, 20)
    benefit_by_bin = np.zeros(len(cap_bins) - 1)
    counts_by_bin = np.zeros(len(cap_bins) - 1)

    for rep in range(n_replications):
        s = seed + rep * 1000
        r_constrained = simulate_evolution(sigma, params, seed=s)
        r_baseline = simulate_evolution(0.0, params, seed=s)

        init_caps = r_constrained.capability_history[0]
        final_constrained = r_constrained.capability_history[-1]
        final_baseline = r_baseline.capability_history[-1]
        benefit = final_constrained - final_baseline

        for i in range(len(cap_bins) - 1):
            mask = (init_caps >= cap_bins[i]) & (init_caps < cap_bins[i + 1])
            if mask.sum() > 0:
                benefit_by_bin[i] += benefit[mask].mean()
                counts_by_bin[i] += 1

    avg_benefit = np.where(counts_by_bin > 0, benefit_by_bin / counts_by_bin, 0)
    positive_bins = np.where(avg_benefit > 0)[0]
    if len(positive_bins) == 0:
        return cap_bins[-1]
    return cap_bins[positive_bins[0]]


def sweep_sigma(sigmas, params, n_replications=10, seed=42):
    results = {}
    for sigma in sigmas:
        terminal_caps = []
        for rep in range(n_replications):
            r = simulate_evolution(sigma, params, seed=seed + rep * 1000)
            terminal_caps.append(r.aggregate_capability[-1])
        results[sigma] = terminal_caps
    return results
