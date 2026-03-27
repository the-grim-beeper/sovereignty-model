"""Tests for the calibration module (TDD — written before implementation)."""
import math
import pytest
from model.calibration import Parameters, window_openness, h_sigma, phi_dependency, parameter_sweep


class TestParameterDefaults:
    def test_D0_default(self):
        p = Parameters()
        assert p.D0 == pytest.approx(0.67)

    def test_R_default(self):
        p = Parameters()
        assert p.R == pytest.approx(0.022)

    def test_depreciation_positive(self):
        p = Parameters()
        assert p.delta_p > 0
        assert p.delta_f > 0

    def test_beta_in_unit_interval(self):
        p = Parameters()
        assert 0 < p.beta < 1

    def test_kappa_in_unit_interval(self):
        p = Parameters()
        assert 0 < p.kappa < 1


class TestWindowOpenness:
    def test_at_open_time_equals_W0(self):
        """At exactly t_open the window should be fully open (W0)."""
        assert window_openness(1.0, t_open=1.0, W0=1.0, gamma=0.1) == pytest.approx(1.0)

    def test_decays_over_time(self):
        """Window openness should decrease as time increases past t_open."""
        w1 = window_openness(2.0, t_open=1.0, W0=1.0, gamma=0.1)
        w2 = window_openness(5.0, t_open=1.0, W0=1.0, gamma=0.1)
        assert w1 > w2

    def test_zero_before_open(self):
        """Before the window opens, openness must be zero."""
        assert window_openness(0.5, t_open=1.0, W0=1.0, gamma=0.1) == 0.0


class TestHSigma:
    def test_h_at_zero(self):
        """h(0) should equal 1 (the additive constant)."""
        assert h_sigma(0.0) == pytest.approx(1.0)

    def test_hump_shape(self):
        """h should rise then fall — i.e. peak is not at sigma=0."""
        h0 = h_sigma(0.0)
        h_peak = h_sigma(1.0 / 0.5)  # σ* = 1/b = 2.0
        h_far = h_sigma(10.0)
        assert h_peak > h0
        assert h_peak > h_far

    def test_always_positive(self):
        """h(σ) should be positive for all σ >= 0."""
        for sigma in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]:
            assert h_sigma(sigma) > 0

    def test_peak_at_one_over_b(self):
        """Peak of h(σ) is at σ* = 1/b."""
        b = 0.5
        sigma_star = 1.0 / b  # = 2.0
        h_at_star = h_sigma(sigma_star, b=b)
        # Slight perturbations should give lower values
        assert h_at_star >= h_sigma(sigma_star - 0.01, b=b)
        assert h_at_star >= h_sigma(sigma_star + 0.01, b=b)


class TestParameterSweep:
    def test_sweep_returns_results(self):
        results = parameter_sweep("lam", [0.1, 0.3, 0.5])
        assert len(results) == 3

    def test_higher_lock_in_speed_increases_dependency(self):
        results = parameter_sweep("lam", [0.1, 0.5])
        assert results[0.5].D[-1] > results[0.1].D[-1]

    def test_faster_window_closure_reduces_kf(self):
        results = parameter_sweep("gamma", [0.05, 0.2])
        assert results[0.05].Kf[-1] > results[0.2].Kf[-1]
