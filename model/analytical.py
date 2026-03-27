"""Layer 1: Analytical core — sovereignty allocation optimal control."""
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from model.calibration import Parameters, window_openness, h_sigma, phi_dependency


class SimulationResult(NamedTuple):
    t: np.ndarray
    Kp: np.ndarray
    Kf: np.ndarray
    D: np.ndarray
    W: np.ndarray
    V_total: float


class OptimalAlphaResult(NamedTuple):
    alpha_star: float
    V_star: float
    alpha_grid: np.ndarray
    V_grid: np.ndarray


def state_derivatives(t, state, alpha, sigma, params):
    Kp, Kf, D = state
    D = np.clip(D, 0.0, 1.0)

    exploitation_input = alpha * params.R
    dKp_dt = params.A * exploitation_input ** params.kappa - params.delta_p * Kp - phi_dependency(D) * Kp

    W = window_openness(t, params.t_open, params.W0, params.gamma)
    exploration_input = (1.0 - alpha) * params.R
    g = exploration_input ** params.beta * h_sigma(sigma, params.a, params.b)
    dKf_dt = W * g - params.delta_f * Kf

    dD_dt = params.lam * (params.D_bar - D) * D - params.eta * Kp * D

    return (dKp_dt, dKf_dt, dD_dt)


def simulate_forward(alpha, sigma, params, Kp0=0.1, Kf0=0.0):
    y0 = [Kp0, Kf0, params.D0]
    t_span = (0.0, params.T)
    t_eval = np.arange(0, params.T + params.dt, params.dt)

    def ode_rhs(t, y):
        return state_derivatives(t, y, alpha, sigma, params)

    sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, method="RK45", max_step=0.5)

    Kp = sol.y[0]
    Kf = sol.y[1]
    D = np.clip(sol.y[2], 0.0, 1.0)
    t = sol.t
    W = np.array([window_openness(ti, params.t_open, params.W0, params.gamma) for ti in t])

    V_flow = params.omega * Kp + (1.0 - params.omega) * Kf - params.psi * D ** 2
    discount = np.exp(-params.rho * t)
    V_total = np.trapezoid(V_flow * discount, t)

    return SimulationResult(t=t, Kp=Kp, Kf=Kf, D=D, W=W, V_total=V_total)


def compute_shadow_price_of_delay(alpha, sigma, delay_years, params):
    V_now = simulate_forward(alpha, sigma, params).V_total
    p_delay = Parameters(**{**params.__dict__, "t_open": params.t_open + delay_years})
    V_delayed = simulate_forward(alpha, sigma, p_delay).V_total
    return V_now - V_delayed


def find_optimal_alpha(sigma, params, n_grid=50):
    alpha_grid = np.linspace(0.01, 0.99, n_grid)
    V_grid = np.array([simulate_forward(a, sigma, params).V_total for a in alpha_grid])

    best_idx = np.argmax(V_grid)
    lo = alpha_grid[max(0, best_idx - 1)]
    hi = alpha_grid[min(len(alpha_grid) - 1, best_idx + 1)]

    result = minimize_scalar(lambda a: -simulate_forward(a, sigma, params).V_total, bounds=(lo, hi), method="bounded")

    return OptimalAlphaResult(alpha_star=result.x, V_star=-result.fun, alpha_grid=alpha_grid, V_grid=V_grid)


def compute_comparative_statics(params=None, alpha=0.3, sigma=2.0):
    """Compute comparative statics for key parameters."""
    from model.calibration import parameter_sweep
    if params is None:
        from model.calibration import Parameters
        params = Parameters()
    statics = {}
    for param, values in [
        ("lam", [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("gamma", [0.05, 0.1, 0.15, 0.2, 0.25]),
        ("beta", [0.3, 0.4, 0.5, 0.6, 0.7]),
    ]:
        results = parameter_sweep(param, values, params, alpha, sigma)
        statics[param] = {v: r.V_total for v, r in results.items()}
    return statics
