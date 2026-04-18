"""Unit tests for backend.optimizer -- Black-Litterman and Sharpe MVO."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.optimizer import (
    neg_sharpe,
    optimise_weights,
    shrinkage_cov,
    black_litterman_mu,
    allocate,
)


@pytest.fixture
def sample_returns():
    """60-day return panel for 5 tickers."""
    np.random.seed(42)
    data = np.random.normal(0.0005, 0.02, (60, 5))
    idx = pd.bdate_range("2024-10-01", periods=60)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    return pd.DataFrame(data, index=idx, columns=tickers)


@pytest.fixture
def sample_cov(sample_returns):
    return shrinkage_cov(sample_returns)


# ── Neg Sharpe ───────────────────────────────────────────────────────

class TestNegSharpe:
    def test_returns_scalar(self):
        w = np.array([0.5, 0.5])
        mu = np.array([0.01, 0.02])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        result = neg_sharpe(w, mu, cov)
        assert isinstance(result, float)

    def test_negative_for_positive_returns(self):
        w = np.array([0.5, 0.5])
        mu = np.array([0.01, 0.02])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        assert neg_sharpe(w, mu, cov) < 0  # negative because it's neg Sharpe


# ── Optimise Weights ─────────────────────────────────────────────────

class TestOptimiseWeights:
    def test_weights_sum_to_one(self, sample_cov):
        mu = np.array([0.01, 0.02, 0.015, 0.005, 0.012])
        w = optimise_weights(mu, sample_cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_within_bounds(self, sample_cov):
        mu = np.array([0.01, 0.02, 0.015, 0.005, 0.012])
        w = optimise_weights(mu, sample_cov, min_w=0.05, max_w=0.40)
        assert np.all(w >= 0.05 - 1e-6)
        assert np.all(w <= 0.40 + 1e-6)

    def test_equal_weight_fallback_when_infeasible(self):
        # min_w * n > 1 triggers equal weight fallback
        mu = np.array([0.01, 0.02])
        cov = np.eye(2) * 0.04
        w = optimise_weights(mu, cov, min_w=0.6, max_w=0.9)
        np.testing.assert_array_almost_equal(w, [0.5, 0.5])

    def test_all_positive(self, sample_cov):
        mu = np.array([0.01, 0.02, 0.015, 0.005, 0.012])
        w = optimise_weights(mu, sample_cov)
        assert np.all(w >= -1e-8)


# ── Shrinkage Covariance ─────────────────────────────────────────────

class TestShrinkageCov:
    def test_positive_semi_definite(self, sample_returns):
        cov = shrinkage_cov(sample_returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_symmetric(self, sample_returns):
        cov = shrinkage_cov(sample_returns)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_shape(self, sample_returns):
        cov = shrinkage_cov(sample_returns)
        n = sample_returns.shape[1]
        assert cov.shape == (n, n)

    def test_shrinkage_reduces_condition_number(self, sample_returns):
        raw_cov = sample_returns.cov().values
        shrunk = shrinkage_cov(sample_returns)
        cond_raw = np.linalg.cond(raw_cov)
        cond_shrunk = np.linalg.cond(shrunk)
        assert cond_shrunk <= cond_raw


# ── Black-Litterman ──────────────────────────────────────────────────

class TestBlackLitterman:
    def test_returns_no_nan(self, sample_returns):
        tickers = list(sample_returns.columns)
        scores = {t: np.random.randn() for t in tickers}
        bl_mu = black_litterman_mu(sample_returns, tickers, scores)
        assert not np.any(np.isnan(bl_mu))

    def test_returns_correct_length(self, sample_returns):
        tickers = list(sample_returns.columns)
        scores = {t: 0.5 for t in tickers}
        bl_mu = black_litterman_mu(sample_returns, tickers, scores)
        assert len(bl_mu) == len(tickers)

    def test_zero_scores_return_equilibrium(self, sample_returns):
        tickers = list(sample_returns.columns)
        scores = {t: 0.0 for t in tickers}
        bl_mu = black_litterman_mu(sample_returns, tickers, scores)
        # With all-zero scores, score_std < 1e-8, so view_returns = equilibrium
        # BL posterior should equal equilibrium
        cov = shrinkage_cov(sample_returns)
        eq_w = np.full(len(tickers), 1.0 / len(tickers))
        equilibrium = 2.5 * cov @ eq_w
        np.testing.assert_array_almost_equal(bl_mu, equilibrium, decimal=6)

    def test_positive_views_tilt_upward(self, sample_returns):
        tickers = list(sample_returns.columns)
        # Strong positive view on first ticker
        scores = {t: 0.0 for t in tickers}
        scores[tickers[0]] = 5.0
        bl_mu = black_litterman_mu(sample_returns, tickers, scores)
        # First ticker should have the highest expected return
        assert bl_mu[0] == bl_mu.max()


# ── Allocate (integration) ───────────────────────────────────────────

class TestAllocate:
    def test_allocation_sums_to_one(self, sample_returns):
        close = (1 + sample_returns).cumprod() * 100
        tickers = list(close.columns)
        scores = {t: np.random.randn() for t in tickers}
        alloc = allocate(close, tickers, scores, lookback=60)
        assert abs(sum(alloc.values()) - 1.0) < 1e-6

    def test_all_tickers_allocated(self, sample_returns):
        close = (1 + sample_returns).cumprod() * 100
        tickers = list(close.columns)
        scores = {t: 0.5 for t in tickers}
        alloc = allocate(close, tickers, scores, lookback=60)
        assert set(alloc.keys()) == set(tickers)

    def test_equal_weight_with_insufficient_data(self):
        # Only 5 rows -- should trigger equal weight fallback
        idx = pd.bdate_range("2024-01-01", periods=5)
        close = pd.DataFrame(
            np.random.rand(5, 3) * 100,
            index=idx, columns=["A", "B", "C"]
        )
        alloc = allocate(close, ["A", "B", "C"], lookback=60)
        expected = 1.0 / 3
        for v in alloc.values():
            assert abs(v - expected) < 1e-6
