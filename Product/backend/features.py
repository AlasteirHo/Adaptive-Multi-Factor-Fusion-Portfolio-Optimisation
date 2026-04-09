"""
Feature engineering: technical indicators, z-scoring, and context features.
Mirrors notebook cell 6.
"""

import numpy as np
import pandas as pd

from .config import (
    FACTOR_COLS,
    FWD_HORIZON,
    HIGH_52W_WINDOW,
    IDIOVOL_WINDOW,
    MOMENTUM_PERIOD,
    REVERSAL_PERIOD,
    RSI_PERIOD,
    SECTOR_MAP,
    SECTORS,
    VOLATILITY_WINDOW,
    VOLUME_AVG_WINDOW,
)


def compute_rsi(close, period=RSI_PERIOD):
    price_delta = close.diff()
    avg_gain = price_delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-price_delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return ((100 - 100 / (1 + relative_strength)) - 50) / 50


def compute_momentum(close):
    return close.pct_change(MOMENTUM_PERIOD).fillna(0)


def compute_reversal(close):
    return -close.pct_change(REVERSAL_PERIOD).fillna(0)


def compute_abnormal_volume(volume):
    avg_vol = volume.rolling(VOLUME_AVG_WINDOW, min_periods=10).mean()
    return (volume / avg_vol.replace(0, np.nan)).fillna(1.0) - 1.0


def compute_idiovol(close, window=IDIOVOL_WINDOW):
    return close.pct_change().rolling(window, min_periods=10).std().fillna(0)


def compute_52w_high_ratio(close, window=HIGH_52W_WINDOW):
    rolling_max = close.rolling(window, min_periods=20).max()
    return (close / rolling_max.replace(0, np.nan)).fillna(0)


def compute_vol_regime(close):
    realized_volatility = close.pct_change().rolling(VOLATILITY_WINDOW).std()
    quantile_33 = realized_volatility.expanding().quantile(0.33)
    quantile_66 = realized_volatility.expanding().quantile(0.66)
    regime_series = pd.Series(1, index=close.index)
    regime_series[realized_volatility <= quantile_33] = 0
    regime_series[realized_volatility > quantile_66] = 2
    return regime_series.astype(float)


def expanding_zscore(series, min_periods=30):
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std().replace(0, np.nan)
    return ((series - expanding_mean) / expanding_std).clip(-3, 3).fillna(0)


def sector_onehot(ticker):
    one_hot_vector = np.zeros(len(SECTORS), dtype=np.float32)
    sector_name = SECTOR_MAP.get(ticker, SECTORS[0])
    if sector_name in SECTORS:
        one_hot_vector[SECTORS.index(sector_name)] = 1.0
    return one_hot_vector


def build_features(master):
    all_features = {}
    for ticker, df in master.items():
        ticker_df = df.copy()
        ticker_df["rsi"] = compute_rsi(ticker_df["Close"])
        ticker_df["momentum"] = compute_momentum(ticker_df["Close"])
        ticker_df["reversal"] = compute_reversal(ticker_df["Close"])
        ticker_df["abnormal_volume"] = compute_abnormal_volume(ticker_df["Volume"])
        ticker_df["idiovol"] = compute_idiovol(ticker_df["Close"])
        ticker_df["high_52w_ratio"] = compute_52w_high_ratio(ticker_df["Close"])
        ticker_df["vol_regime"] = compute_vol_regime(ticker_df["Close"])

        ticker_df["z_news_sentiment"] = expanding_zscore(ticker_df["news_sentiment"])
        ticker_df["z_social_sentiment"] = expanding_zscore(ticker_df["social_sentiment"])
        ticker_df["z_rsi"] = expanding_zscore(ticker_df["rsi"])
        ticker_df["z_momentum"] = expanding_zscore(ticker_df["momentum"])
        ticker_df["z_reversal"] = expanding_zscore(ticker_df["reversal"])
        ticker_df["z_abnormal_volume"] = expanding_zscore(ticker_df["abnormal_volume"])
        ticker_df["z_idiovol"] = expanding_zscore(ticker_df["idiovol"])
        ticker_df["z_52w_high_ratio"] = expanding_zscore(ticker_df["high_52w_ratio"])

        ticker_df["volatility_regime"] = ticker_df["vol_regime"]
        ticker_df["news_intensity"] = ticker_df["news_available"].rolling(5).mean().fillna(0)
        ticker_df["social_intensity"] = (
            ticker_df["social_sentiment"].abs().rolling(5).mean().fillna(0)
        )
        one_hot_vector = sector_onehot(ticker)
        for sector_idx, sector_label in enumerate(SECTORS):
            ticker_df[f"sector_{sector_label}"] = one_hot_vector[sector_idx]

        for h in [1, 5, FWD_HORIZON]:
            ticker_df[f"fwd_return_{h}d"] = ticker_df["Close"].pct_change(h).shift(-h)
        all_features[ticker] = ticker_df
    n_cols = len(next(iter(all_features.values())).columns)
    print(f"Features built: {len(all_features)} tickers x {n_cols} columns")
    return all_features
