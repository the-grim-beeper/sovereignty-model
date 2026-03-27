"""Shared parameters, empirical anchors, and utility functions.

All parameters documented with sources. This module is the single source
of truth for calibration across all three model layers.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    """Model parameters with empirical calibration."""

    # --- Layer 1: Analytical ---
    R: float = 0.022          # EU R&D/GDP (EIB)
    kappa: float = 0.6        # Exploitation diminishing returns
    beta: float = 0.5         # Exploration returns exponent
    A: float = 1.0            # TFP for exploitation (normalized)
    delta_p: float = 0.05     # Present sovereignty depreciation (5%/yr)
    delta_f: float = 0.03     # Future sovereignty depreciation (3%/yr)
    D0: float = 0.67          # Initial dependency (Synergy Research Group)
    D_bar: float = 0.95       # Lock-in ceiling
    lam: float = 0.3          # Lock-in speed
    eta: float = 0.1          # Present sovereignty → dependency reduction
    a: float = 2.0            # Hump shape: amplitude
    b: float = 0.5            # Hump shape: decay rate → σ* = 1/b = 2.0
    omega: float = 0.5        # Weight on present sovereignty
    psi: float = 1.0          # Dependency penalty
    rho: float = 0.03         # Discount rate (3%)

    # --- Window parameters ---
    W0: float = 1.0           # Initial window openness
    gamma: float = 0.1        # Window closure rate
    t_open: float = 0.0       # Window opening time

    # --- Layer 2: Evolutionary ---
    N: int = 500              # Number of firms
    p0: float = 0.7           # Base exploitation success probability
    q0: float = 0.15          # Base exploration success probability
    nu: float = 0.3           # Capability exponent
    eps_exploit: float = 0.05 # Exploitation payoff scale
    eps_explore: float = 0.1  # Exploration payoff location (Pareto)
    xi: float = 2.5           # Exploration payoff shape (Pareto)
    theta: float = 0.5        # Selection intensity
    s_min: float = 0.0005     # Minimum market share (1/N * 0.25)

    # --- Simulation ---
    T: float = 30.0           # Time horizon (years)
    dt: float = 1.0           # Layer 2 period length (years)


def window_openness(t: float, t_open: float, W0: float, gamma: float) -> float:
    """Compute Perez window openness at time t."""
    if t < t_open:
        return 0.0
    return W0 * np.exp(-gamma * (t - t_open))


def h_sigma(sigma: float, a: float = 2.0, b: float = 0.5) -> float:
    """Constraint instrument productivity modifier. h(σ) = 1 + a·σ·exp(-b·σ)"""
    return 1.0 + a * sigma * np.exp(-b * sigma)


def phi_dependency(D: float, scale: float = 0.1) -> float:
    """Dependency-accelerated depreciation. φ(D) = scale * D²"""
    return scale * D ** 2


def parameter_sweep(param_name, values, base_params=None, alpha=0.3, sigma=2.0):
    """Sweep a single parameter, running forward simulation for each value."""
    from model.analytical import simulate_forward
    if base_params is None:
        base_params = Parameters()
    results = {}
    for v in values:
        p = Parameters(**{**base_params.__dict__, param_name: v})
        results[v] = simulate_forward(alpha, sigma, p)
    return results
