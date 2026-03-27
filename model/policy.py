"""Layer 3: Policy design interface — strategic necessity instrument."""
from typing import NamedTuple
import numpy as np

from model.calibration import Parameters
from model.analytical import simulate_forward, find_optimal_alpha, compute_shadow_price_of_delay


class PolicyRecommendation(NamedTuple):
    alpha_star: float
    sigma_star: float
    domain_scores: dict[str, float]
    delay_cost_1yr: float


def compute_delay_cost_curve(alpha, sigma, max_delay, params, n_points=20):
    delays = np.linspace(0, max_delay, n_points)
    costs = np.zeros(n_points)
    for i, d in enumerate(delays):
        if d == 0:
            costs[i] = 0.0
        else:
            costs[i] = compute_shadow_price_of_delay(alpha, sigma, d, params)
    return delays, costs


def compute_regime_comparison(params):
    r_sq = simulate_forward(alpha=0.8, sigma=0.0, params=params)
    sigma_star = 1.0 / params.b
    opt = find_optimal_alpha(sigma=sigma_star, params=params)
    r_designed = simulate_forward(alpha=opt.alpha_star, sigma=sigma_star, params=params)
    r_over = simulate_forward(alpha=0.2, sigma=15.0, params=params)
    return {"status_quo": r_sq, "designed": r_designed, "overreaction": r_over}


def score_domain(W_remaining, capability, strategic_value, w_weight=0.4, c_weight=0.3, s_weight=0.3):
    return w_weight * W_remaining + c_weight * capability + s_weight * strategic_value


DEFAULT_DOMAINS = {
    "scientific_ai": {"W_remaining": 0.7, "capability": 0.4, "strategic_value": 0.8},
    "quantum": {"W_remaining": 0.6, "capability": 0.6, "strategic_value": 0.7},
    "fusion": {"W_remaining": 0.8, "capability": 0.5, "strategic_value": 0.9},
}


def compute_policy_recommendation(D_obs, C_obs, W_obs, params, domains=None):
    if domains is None:
        domains = DEFAULT_DOMAINS
    adj_params = Parameters(**{**params.__dict__, "D0": D_obs, "W0": W_obs})
    sigma_star_base = 1.0 / params.b
    sigma_star = sigma_star_base * C_obs
    opt = find_optimal_alpha(sigma=sigma_star, params=adj_params)
    domain_scores = {name: score_domain(**assessment) for name, assessment in domains.items()}
    delay_cost = compute_shadow_price_of_delay(opt.alpha_star, sigma_star, 1.0, adj_params)
    return PolicyRecommendation(
        alpha_star=opt.alpha_star, sigma_star=sigma_star,
        domain_scores=domain_scores, delay_cost_1yr=delay_cost,
    )
