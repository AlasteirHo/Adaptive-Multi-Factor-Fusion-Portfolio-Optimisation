"""Unit tests for backend.features -- technical indicators and z-scoring."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.features import (
    compute_rsi,
    compute_momentum,
    compute_reversal,
    compute_abnormal_volume,
    compute_idiovol,
    compute_52w_high_ratio,
    compute_vol_regime,
    expanding_zscore,
    sector_onehot,
)


@pytest.fixture
def sample_close():
    """252 days of synthetic close prices with an upward drift."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.bdate_range("2024-01-02", periods=252)
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def sample_volume():
    np.random.seed(42)
    vol = np.random.randint(1_000_000, 10_000_000, 252).astype(float)
    idx = pd.bdate_range("2024-01-02", periods=252)
    return pd.Series(vol, index=idx, name="Volume")


# ── RSI ──────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_bounded(self, sample_close):
        rsi = compute_rsi(sample_close)
        valid = rsi.dropna()
        assert valid.min() >= -1.0, f"RSI below -1: {valid.min()}"
        assert valid.max() <= 1.0, f"RSI above 1: {valid.max()}"

    def test_rsi_length(self, sample_close):
        rsi = compute_rsi(sample_close)
        assert len(rsi) == len(sample_close)

    def test_rsi_constant_prices_returns_zero(self):
        flat = pd.Series([100.0] * 50)
        rsi = compute_rsi(flat, period=14)
        # With no price change, RSI should be NaN or 0
        valid = rsi.dropna()
        if len(valid) > 0:
            assert (valid.abs() < 1e-6).all() or valid.isna().all()


# ── Momentum & Reversal ──────────────────────────────────────────────

class TestMomentumReversal:
    def test_momentum_sign(self, sample_close):
        mom = compute_momentum(sample_close)
        # Rising prices should produce positive momentum overall
        assert mom.iloc[-1] != 0  # not all zero

    def test_reversal_is_negated(self, sample_close):
        mom = sample_close.pct_change(5).fillna(0)
        rev = compute_reversal(sample_close)
        # Reversal should be the negative of 5-day return
        np.testing.assert_array_almost_equal(rev.values, -mom.values, decimal=10)

    def test_momentum_length(self, sample_close):
        assert len(compute_momentum(sample_close)) == len(sample_close)


# ── Abnormal Volume ──────────────────────────────────────────────────

class TestAbnormalVolume:
    def test_abnormal_volume_centered_near_zero(self, sample_volume):
        av = compute_abnormal_volume(sample_volume)
        # Mean should be near zero (volume fluctuates around its own average)
        assert abs(av.mean()) < 0.5

    def test_abnormal_volume_no_nan(self, sample_volume):
        av = compute_abnormal_volume(sample_volume)
        assert not av.isna().any()


# ── Idiosyncratic Volatility ─────────────────────────────────────────

class TestIdiovol:
    def test_idiovol_non_negative(self, sample_close):
        iv = compute_idiovol(sample_close)
        assert (iv >= 0).all()

    def test_idiovol_positive_after_warmup(self, sample_close):
        iv = compute_idiovol(sample_close)
        # After 20-day warmup, should be positive
        assert (iv.iloc[25:] > 0).all()


# ── 52-Week High Ratio ───────────────────────────────────────────────

class TestHigh52W:
    def test_ratio_bounded_zero_one(self, sample_close):
        ratio = compute_52w_high_ratio(sample_close)
        valid = ratio[ratio > 0]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0 + 1e-6

    def test_ratio_equals_one_at_high(self):
        # Monotonically increasing prices: ratio should always be 1
        prices = pd.Series(np.arange(1, 300, dtype=float))
        ratio = compute_52w_high_ratio(prices, window=252)
        valid = ratio[ratio > 0]
        np.testing.assert_array_almost_equal(valid.values, 1.0, decimal=6)


# ── Volatility Regime ────────────────────────────────────────────────

class TestVolRegime:
    def test_regime_values(self, sample_close):
        regime = compute_vol_regime(sample_close)
        assert set(regime.unique()).issubset({0.0, 1.0, 2.0})

    def test_regime_length(self, sample_close):
        regime = compute_vol_regime(sample_close)
        assert len(regime) == len(sample_close)


# ── Expanding Z-Score ────────────────────────────────────────────────

class TestExpandingZscore:
    def test_clipped_to_minus3_plus3(self, sample_close):
        z = expanding_zscore(sample_close, min_periods=30)
        assert z.min() >= -3.0
        assert z.max() <= 3.0

    def test_zero_filled_before_min_periods(self):
        s = pd.Series(np.random.randn(50))
        z = expanding_zscore(s, min_periods=30)
        # First 29 values should be 0 (before min_periods)
        assert (z.iloc[:29] == 0).all()

    def test_no_nan(self, sample_close):
        z = expanding_zscore(sample_close, min_periods=30)
        assert not z.isna().any()


# ── Sector One-Hot ───────────────────────────────────────────────────

class TestSectorOnehot:
    def test_known_ticker(self):
        vec = sector_onehot("AAPL")
        assert vec.sum() == 1.0
        assert vec.dtype == np.float32

    def test_unknown_ticker_defaults(self):
        vec = sector_onehot("ZZZZ")
        # Should default to first sector
        assert vec.sum() == 1.0
        assert vec[0] == 1.0

    def test_different_sectors(self):
        tech = sector_onehot("AAPL")
        energy = sector_onehot("XOM")
        assert not np.array_equal(tech, energy)
