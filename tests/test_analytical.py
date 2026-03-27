"""Tests for Layer 1: Analytical core."""
import pytest
import numpy as np
from model.analytical import (
    state_derivatives,
    simulate_forward,
    compute_shadow_price_of_delay,
    find_optimal_alpha,
)
from model.calibration import Parameters


class TestStateDerivatives:
    def test_returns_three_derivatives(self):
        p = Parameters()
        state = [0.5, 0.3, 0.67]
        derivs = state_derivatives(0.0, state, alpha=0.5, sigma=0.0, params=p)
        assert len(derivs) == 3

    def test_zero_investment_decays_sovereignty(self):
        p = Parameters()
        state = [1.0, 0.0, 0.5]
        dKp, _, _ = state_derivatives(0.0, state, alpha=0.0, sigma=0.0, params=p)
        assert dKp < 0

    def test_full_exploitation_grows_present(self):
        p = Parameters()
        state = [0.1, 0.0, 0.1]
        dKp, _, _ = state_derivatives(0.0, state, alpha=1.0, sigma=0.0, params=p)
        assert dKp > 0

    def test_dependency_increases_without_intervention(self):
        p = Parameters()
        state = [0.0, 0.0, 0.5]
        _, _, dD = state_derivatives(0.0, state, alpha=0.5, sigma=0.0, params=p)
        assert dD > 0

    def test_dependency_clamped_to_unit_interval(self):
        p = Parameters()
        result = simulate_forward(alpha=0.5, sigma=0.0, params=p)
        assert np.all(result.D >= 0)
        assert np.all(result.D <= 1)

    def test_window_modulates_exploration(self):
        p = Parameters()
        r1 = simulate_forward(alpha=0.3, sigma=1.0, params=p)
        p2 = Parameters(t_open=100.0)
        r2 = simulate_forward(alpha=0.3, sigma=1.0, params=p2)
        assert r1.Kf[-1] > r2.Kf[-1]


class TestSimulateForward:
    def test_output_shape(self):
        p = Parameters(T=10.0, dt=1.0)
        result = simulate_forward(alpha=0.5, sigma=0.0, params=p)
        assert len(result.t) > 1
        assert len(result.Kp) == len(result.t)
        assert len(result.Kf) == len(result.t)
        assert len(result.D) == len(result.t)

    def test_designed_pressure_increases_future_sovereignty(self):
        p = Parameters()
        r_no = simulate_forward(alpha=0.3, sigma=0.0, params=p)
        r_opt = simulate_forward(alpha=0.3, sigma=2.0, params=p)
        assert r_opt.Kf[-1] > r_no.Kf[-1]

    def test_excessive_pressure_yields_less_than_optimal(self):
        p = Parameters()
        r_opt = simulate_forward(alpha=0.3, sigma=2.0, params=p)
        r_high = simulate_forward(alpha=0.3, sigma=20.0, params=p)
        assert r_opt.Kf[-1] > r_high.Kf[-1]


class TestShadowPrice:
    def test_delay_cost_positive(self):
        p = Parameters()
        cost = compute_shadow_price_of_delay(alpha=0.3, sigma=2.0, delay_years=1.0, params=p)
        assert cost > 0

    def test_delay_cost_increases_with_delay(self):
        p = Parameters()
        c1 = compute_shadow_price_of_delay(alpha=0.3, sigma=2.0, delay_years=1.0, params=p)
        c5 = compute_shadow_price_of_delay(alpha=0.3, sigma=2.0, delay_years=5.0, params=p)
        assert c5 > c1


class TestOptimalAlpha:
    def test_optimal_alpha_interior(self):
        p = Parameters()
        result = find_optimal_alpha(sigma=2.0, params=p)
        assert 0 < result.alpha_star < 1

    def test_optimal_alpha_with_closed_window_favors_exploitation(self):
        p = Parameters(t_open=100.0)
        result = find_optimal_alpha(sigma=0.0, params=p)
        assert result.alpha_star > 0.7
