"""Portfolio optimisation: Black-Litterman posterior and Sharpe MVO."""

import numpy as np
from sklearn.covariance import LedoitWolf

from scipy.optimize import minimize

from .config import (
    BL_DELTA,
    BL_OMEGA_SCALE,
    BL_TAU,
    MAX_WEIGHT,
    MIN_WEIGHT,
)


def neg_sharpe(weights, expected_returns, covariance_matrix):
    portfolio_return = weights @ expected_returns
    portfolio_std = np.sqrt(max(weights @ covariance_matrix @ weights, 1e-10))
    return -(portfolio_return / portfolio_std)


def optimise_weights(expected_returns, covariance_matrix, min_w=MIN_WEIGHT, max_w=MAX_WEIGHT):
    n_assets = len(expected_returns)
    if min_w * n_assets > 1.0:
        return np.full(n_assets, 1 / n_assets)
    initial_weights = np.clip(np.full(n_assets, 1 / n_assets), min_w, max_w)
    initial_weights /= initial_weights.sum()
    opt_result = minimize(
        neg_sharpe, initial_weights,
        args=(expected_returns, covariance_matrix),
        method="SLSQP",
        bounds=[(min_w, max_w)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if opt_result.success:
        clipped_weights = np.clip(opt_result.x, min_w, max_w)
        final_weights = clipped_weights / clipped_weights.sum()
    else:
        final_weights = np.full(n_assets, 1 / n_assets)

    assert abs(final_weights.sum() - 1.0) < 1e-6, \
        f"Weight sum-to-one violated: {final_weights.sum():.8f}"
    assert np.all(final_weights >= -1e-8), \
        f"Negative weight detected: {final_weights.min():.8f}"
    assert np.all(final_weights <= max_w + 1e-6), \
        f"Weight cap exceeded: {final_weights.max():.6f} > {max_w}"
    return final_weights


def shrinkage_cov(returns):
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_


def black_litterman_mu(returns, tickers, composite_scores, Sigma=None):
    n_assets = len(tickers)
    if Sigma is None:
        Sigma = shrinkage_cov(returns)
    equal_weights = np.full(n_assets, 1.0 / n_assets)
    equilibrium_returns = BL_DELTA * Sigma @ equal_weights
    score_array = np.array(
        [composite_scores.get(ticker, 0.0) for ticker in tickers], dtype=float
    )
    return_std = float(returns.std().values.mean())
    score_std = float(score_array.std())
    view_returns = (
        (score_array / score_std * return_std)
        if score_std > 1e-8
        else equilibrium_returns.copy()
    )
    tau_cov = BL_TAU * Sigma
    omega = np.diag(np.diag(tau_cov)) * BL_OMEGA_SCALE
    prior_plus_omega = tau_cov + omega
    try:
        prior_plus_omega_inv = np.linalg.inv(prior_plus_omega)
    except np.linalg.LinAlgError:
        return returns.mean().values
    bl_returns = equilibrium_returns + tau_cov @ prior_plus_omega_inv @ (
        view_returns - equilibrium_returns
    )
    assert not np.any(np.isnan(bl_returns)), "BL posterior contains NaN values"
    return bl_returns


def allocate(close_panel, selected_tickers, composite_scores=None, lookback=60):
    available_tickers = [t for t in selected_tickers if t in close_panel.columns]
    returns = close_panel[available_tickers].tail(lookback).pct_change().dropna()
    if len(returns) < 10:
        return {t: 1 / len(available_tickers) for t in available_tickers}
    covariance_matrix = shrinkage_cov(returns)
    if composite_scores:
        expected_returns = black_litterman_mu(
            returns, available_tickers, composite_scores, Sigma=covariance_matrix
        )
    else:
        expected_returns = returns.mean().values
    weights = optimise_weights(expected_returns, covariance_matrix)
    allocation = {t: float(w) for t, w in zip(available_tickers, weights)}
    assert abs(sum(allocation.values()) - 1.0) < 1e-6, \
        f"Allocation does not sum to 1: {sum(allocation.values()):.8f}"
    return allocation
