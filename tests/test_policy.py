"""Tests for Layer 3: Policy design interface."""
import pytest
import numpy as np
from model.policy import (
    compute_delay_cost_curve,
    compute_regime_comparison,
    score_domain,
    compute_policy_recommendation,
)
from model.calibration import Parameters


class TestDelayCostCurve:
    def test_returns_positive_costs(self):
        p = Parameters()
        delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=10.0, params=p)
        assert np.all(costs >= 0)

    def test_monotonically_increasing(self):
        p = Parameters()
        delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=10.0, params=p)
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1] - 1e-10

    def test_zero_delay_zero_cost(self):
        p = Parameters()
        delays, costs = compute_delay_cost_curve(alpha=0.3, sigma=2.0, max_delay=5.0, params=p)
        assert costs[0] == pytest.approx(0.0, abs=1e-10)


class TestRegimeComparison:
    def test_returns_three_regimes(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert "status_quo" in regimes
        assert "designed" in regimes
        assert "overreaction" in regimes

    def test_designed_beats_status_quo_on_kf(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert regimes["designed"].Kf[-1] > regimes["status_quo"].Kf[-1]

    def test_designed_beats_overreaction_on_total_value(self):
        p = Parameters()
        regimes = compute_regime_comparison(params=p)
        assert regimes["designed"].V_total > regimes["overreaction"].V_total


class TestDomainScoring:
    def test_open_window_scores_higher(self):
        s_open = score_domain(W_remaining=0.8, capability=0.5, strategic_value=0.7)
        s_closed = score_domain(W_remaining=0.1, capability=0.5, strategic_value=0.7)
        assert s_open > s_closed

    def test_higher_capability_scores_higher(self):
        s_high = score_domain(W_remaining=0.5, capability=0.8, strategic_value=0.7)
        s_low = score_domain(W_remaining=0.5, capability=0.2, strategic_value=0.7)
        assert s_high > s_low

    def test_score_in_unit_interval(self):
        s = score_domain(W_remaining=0.5, capability=0.5, strategic_value=0.5)
        assert 0 <= s <= 1


class TestPolicyRecommendation:
    def test_returns_all_instruments(self):
        p = Parameters()
        rec = compute_policy_recommendation(D_obs=0.67, C_obs=0.5, W_obs=0.7, params=p)
        assert hasattr(rec, "alpha_star")
        assert hasattr(rec, "sigma_star")
        assert hasattr(rec, "domain_scores")

    def test_high_dependency_shifts_toward_exploration(self):
        p = Parameters()
        rec_high = compute_policy_recommendation(D_obs=0.9, C_obs=0.5, W_obs=0.7, params=p)
        rec_low = compute_policy_recommendation(D_obs=0.2, C_obs=0.5, W_obs=0.7, params=p)
        assert rec_high.alpha_star < rec_low.alpha_star
