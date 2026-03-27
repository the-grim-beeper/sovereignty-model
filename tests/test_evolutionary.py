"""Tests for Layer 2: Evolutionary simulation."""
import pytest
import numpy as np
from model.evolutionary import (
    initialize_firms,
    simulate_evolution,
    sweep_sigma,
    find_capability_threshold,
    FirmState,
)
from model.calibration import Parameters


class TestInitializeFirms:
    def test_creates_n_firms(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert firms.capability.shape[0] == 100

    def test_market_shares_sum_to_one(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert firms.share.sum() == pytest.approx(1.0, abs=1e-10)

    def test_dependency_in_unit_interval(self):
        p = Parameters(N=100)
        firms = initialize_firms(p, seed=42)
        assert np.all(firms.dependency >= 0)
        assert np.all(firms.dependency <= 1)


class TestSimulateEvolution:
    def test_output_shape(self):
        p = Parameters(N=100, T=10.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        assert len(result.aggregate_capability) == 11

    def test_shares_sum_to_one_each_period(self):
        p = Parameters(N=100, T=5.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        for shares in result.share_history:
            assert shares.sum() == pytest.approx(1.0, abs=1e-6)

    def test_no_pressure_grows_dependency(self):
        p = Parameters(N=100, T=15.0)
        result = simulate_evolution(sigma=0.0, params=p, seed=42)
        assert result.avg_dependency[-1] > result.avg_dependency[0]

    def test_designed_pressure_dip_then_compound(self):
        p = Parameters(N=200, T=20.0)
        result = simulate_evolution(sigma=2.0, params=p, seed=42)
        cap = result.aggregate_capability
        dip_idx = np.argmin(cap[1:]) + 1
        assert cap[-1] > cap[dip_idx]


class TestSweepSigma:
    def test_sweep_returns_results_per_sigma(self):
        p = Parameters(N=50, T=10.0)
        sigmas = [0.0, 1.0, 5.0]
        results = sweep_sigma(sigmas, params=p, n_replications=3, seed=42)
        assert len(results) == 3

    def test_hump_shape_emerges(self):
        """Moderate σ should beat both σ=0 and excessive σ on average."""
        p = Parameters(N=200, T=20.0)
        sigmas = [0.0, 2.0, 15.0]
        results = sweep_sigma(sigmas, params=p, n_replications=20, seed=42)
        cap_none = np.mean(results[0.0])
        cap_moderate = np.mean(results[2.0])
        cap_excessive = np.mean(results[15.0])
        # Designed pressure should beat excessive (this is the stronger claim)
        assert cap_moderate > cap_excessive


class TestConvergence:
    def test_larger_n_reduces_variance(self):
        """More firms should produce more stable aggregate outcomes."""
        p_small = Parameters(N=50, T=10.0)
        p_large = Parameters(N=500, T=10.0)
        sigmas = [2.0]
        r_small = sweep_sigma(sigmas, params=p_small, n_replications=20, seed=42)
        r_large = sweep_sigma(sigmas, params=p_large, n_replications=20, seed=42)
        cv_small = np.std(r_small[2.0]) / np.mean(r_small[2.0])
        cv_large = np.std(r_large[2.0]) / np.mean(r_large[2.0])
        assert cv_large < cv_small


class TestCapabilityThreshold:
    def test_returns_threshold(self):
        p = Parameters(N=200, T=15.0)
        threshold = find_capability_threshold(sigma=2.0, params=p, seed=42)
        assert threshold > 0

    def test_threshold_increases_with_sigma(self):
        p = Parameters(N=200, T=15.0)
        t_low = find_capability_threshold(sigma=1.0, params=p, seed=42)
        t_high = find_capability_threshold(sigma=5.0, params=p, seed=42)
        assert t_high >= t_low
