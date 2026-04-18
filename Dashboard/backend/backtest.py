"""Backtest engine and benchmark strategies."""

from dataclasses import dataclass, field as dc_field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_END,
    BACKTEST_START,
    FACTOR_COLS,
    FINRA_TAF_PER_SH,
    INITIAL_NAV,
    N_FACTORS,
    REBALANCE_DAYS,
    SEC_FEE_RATE,
    SLIPPAGE_BPS,
    STATIC_WEIGHTS,
    TOP_N_STOCKS,
)
from .model import get_composite_scores, seed_rng, train_model
from .optimizer import allocate


def get_execution_price(date, ticker, price_data, side="buy", slippage_bps=SLIPPAGE_BPS):
    if ticker in price_data:
        ticker_prices = price_data[ticker]
        trade_date = pd.Timestamp(date).normalize()
        if trade_date in ticker_prices.index:
            open_price = float(ticker_prices.loc[trade_date, "Open"])
            if open_price > 0:
                sign = 1 if side == "buy" else -1
                return open_price * (1 + sign * slippage_bps / 10_000)
    return None


@dataclass
class BacktestResult:
    name:              str
    nav_series:        pd.Series    = dc_field(default_factory=pd.Series)
    returns_series:    pd.Series    = dc_field(default_factory=pd.Series)
    weight_history:    pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    attention_history: List[Dict]   = dc_field(default_factory=list)
    trade_log:         List[Dict]   = dc_field(default_factory=list)
    rebalance_dates:   List         = dc_field(default_factory=list)
    metrics:           Dict         = dc_field(default_factory=dict)
    stop_out_count:    int          = 0

    def compute_metrics(self):
        daily_returns = self.returns_series.dropna()
        if len(daily_returns) < 2:
            return
        annualised_return = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
        annualised_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualised_return / annualised_vol if annualised_vol > 0 else 0.0
        nav_series = self.nav_series
        max_drawdown = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
        self.metrics = {
            "Sharpe Ratio":      round(sharpe_ratio, 4),
            "Annualised Return": round(annualised_return, 4),
            "Annualised Vol":    round(annualised_vol, 4),
            "Max Drawdown":      round(max_drawdown, 4),
            "Calmar Ratio":      round(annualised_return / abs(max_drawdown), 4)
                                 if max_drawdown != 0 else float("nan"),
            "Total Return":      round((nav_series.iloc[-1] / nav_series.iloc[0]) - 1, 4),
        }


def transaction_costs(sell_proceeds, shares_sold):
    return SEC_FEE_RATE * sell_proceeds + min(FINRA_TAF_PER_SH * shares_sold, 8.30)


def _build_panel(price_data, start, end, col):
    ticker_frames = {
        ticker: price_data[ticker].loc[
            (price_data[ticker].index >= start) & (price_data[ticker].index <= end), col
        ]
        for ticker in price_data
    }
    return pd.DataFrame(ticker_frames).sort_index().ffill()


def _static_scores(date, feature_data, tickers, use_sentiment, weights):
    composite_scores = {}
    factor_weights = np.array(weights, dtype=float)
    if not use_sentiment:
        factor_weights[0] = factor_weights[1] = 0
        total = factor_weights.sum()
        factor_weights = factor_weights / total if total > 0 else np.ones(N_FACTORS) / (N_FACTORS - 2)
    for ticker in tickers:
        ticker_df = feature_data.get(ticker)
        if ticker_df is None or date not in ticker_df.index:
            continue
        factor_row = ticker_df.loc[date, FACTOR_COLS].values.astype(np.float32)
        if not use_sentiment:
            factor_row[0] = factor_row[1] = 0
        composite_scores[ticker] = float(np.dot(factor_row, factor_weights))
    return composite_scores


def run_backtest(name, feature_data, price_data,
                 model=None, start=BACKTEST_START, end=BACKTEST_END,
                 initial_nav=INITIAL_NAV, top_n=TOP_N_STOCKS,
                 use_sentiment=True, use_adaptive=True, static_weights=None,
                 retrain_every=0, stop_loss_pct=None,
                 progress_callback=None):
    """
    Run a strategy backtest.

    Parameters
    ----------
    retrain_every : int
        Walk-forward retraining interval in trading days. 0 = no retraining
        during the backtest (pre-trained model used throughout). Mutually
        exclusive with ``stop_loss_pct`` (see note below).
    stop_loss_pct : float | None
        Per-position stop-loss threshold. If set, liquidate any held position
        whose intraday Low falls below entry_price * (1 - stop_loss_pct). The
        fill is at the stop trigger with one-way slippage applied. Cash is
        held until the next rebalance.
    progress_callback : callable, optional
        Called with (day_index, total_days, date, nav) after each trading day.

    Notes
    -----
    Stop-loss is a post-training portfolio rule and is never interleaved
    with walk-forward retraining. When both ``retrain_every > 0`` and
    ``stop_loss_pct`` are requested, walk-forward retraining is disabled with
    a warning and the pre-trained model is used for the entire backtest.
    """
    seed_rng()

    if stop_loss_pct is not None and retrain_every and retrain_every > 0:
        print(
            f"[{name}] WARNING: stop-loss is a post-training rule and cannot be "
            f"combined with walk-forward retraining in the same run. Disabling "
            f"walk-forward retraining; the pre-trained model will be used "
            f"throughout."
        )
        retrain_every = 0

    static_weights = static_weights or STATIC_WEIGHTS
    result = BacktestResult(name=name)
    close_panel = _build_panel(price_data, start, end, "Close")
    low_panel = _build_panel(price_data, start, end, "Low") if stop_loss_pct else None
    trading_days = close_panel.index
    portfolio_nav = initial_nav
    current_holdings = {}
    available_cash = initial_nav
    nav_by_date = {}
    weight_records = []
    attention_records = []
    trade_records = []
    rebalance_date_list = []
    days_since_rebalance = REBALANCE_DAYS
    days_since_retrain = 0
    entry_prices = {}
    n_stop_outs = 0

    for day_index, date in enumerate(trading_days):
        # On non-rebalance days, liquidate any position whose intraday Low
        # has breached entry * (1 - stop_loss_pct). Fill at the stop price,
        # cash held until next rebalance.
        if stop_loss_pct and current_holdings and days_since_rebalance < REBALANCE_DAYS:
            triggered = []
            for ticker, current_shares in list(current_holdings.items()):
                if ticker not in entry_prices:
                    continue
                if ticker not in low_panel.columns:
                    continue
                day_low = low_panel.loc[date, ticker]
                if np.isnan(day_low):
                    continue
                stop_trigger = entry_prices[ticker] * (1.0 - stop_loss_pct)
                if day_low <= stop_trigger:
                    fill_price = stop_trigger * (1 - SLIPPAGE_BPS / 10_000)
                    sell_proceeds = current_shares * fill_price
                    available_cash += sell_proceeds - transaction_costs(
                        sell_proceeds, current_shares)
                    trade_records.append({
                        "date": date, "ticker": ticker, "action": "STOP",
                        "shares": current_shares, "price": fill_price,
                        "value": sell_proceeds,
                    })
                    triggered.append(ticker)
                    n_stop_outs += 1
            for ticker in triggered:
                del current_holdings[ticker]
                entry_prices.pop(ticker, None)

        portfolio_nav = available_cash + sum(
            current_shares * close_panel.loc[date, ticker]
            for ticker, current_shares in current_holdings.items()
            if ticker in close_panel.columns
        )
        nav_by_date[date] = portfolio_nav

        if days_since_rebalance >= REBALANCE_DAYS:
            days_since_rebalance = 0

            # Signals and position sizing use T-1 close (T close is unknown
            # at T open); enforces no-look-ahead at rebalance.
            if day_index == 0:
                days_since_rebalance = REBALANCE_DAYS
                continue
            signal_date = trading_days[day_index - 1]

            rebalance_nav = available_cash + sum(
                current_shares * close_panel.loc[signal_date, ticker]
                for ticker, current_shares in current_holdings.items()
                if ticker in close_panel.columns
            )

            if use_adaptive and retrain_every > 0 and days_since_retrain >= retrain_every:
                days_since_retrain = 0
                prev_state = {k: v.clone() for k, v in model.state_dict().items()}
                candidate, _, _ = train_model(
                    feature_data, train_end=str(signal_date.date()),
                    verbose=False, warm_start_state=prev_state,
                )
                old_scores, _ = get_composite_scores(
                    model, feature_data, signal_date,
                    [t for t in feature_data if t in close_panel.columns])
                new_scores, _ = get_composite_scores(
                    candidate, feature_data, signal_date,
                    [t for t in feature_data if t in close_panel.columns])
                old_spread = (max(old_scores.values()) - min(old_scores.values())) if old_scores else 0
                new_spread = (max(new_scores.values()) - min(new_scores.values())) if new_scores else 0
                if new_spread >= old_spread:
                    model = candidate

            valid_tickers = [
                ticker for ticker in feature_data
                if ticker in close_panel.columns
                and not np.isnan(close_panel.loc[date, ticker])
            ]
            attention_weights = {}
            if use_adaptive and model:
                composite_scores, attention_weights = get_composite_scores(
                    model, feature_data, signal_date, valid_tickers
                )
            else:
                composite_scores = _static_scores(
                    signal_date, feature_data, valid_tickers, use_sentiment, static_weights
                )
            if not composite_scores:
                days_since_rebalance += 1
                continue

            clean_scores = {t: s for t, s in composite_scores.items() if not np.isnan(s)}
            ranked_tickers = sorted(clean_scores.items(), key=lambda item: item[1], reverse=True)
            selected_tickers = [ticker for ticker, _ in ranked_tickers[:top_n]]
            if not selected_tickers:
                days_since_rebalance += 1
                continue

            history_start_index = max(0, day_index - 60)
            price_history = close_panel.loc[
                close_panel.index.isin(trading_days[history_start_index:day_index]),
                selected_tickers,
            ]
            target_weights = allocate(
                price_history, selected_tickers,
                {ticker: composite_scores[ticker] for ticker in selected_tickers},
                lookback=len(price_history),
            )

            # Sells
            new_holdings = {}
            total_sell_proceeds = total_shares_sold = 0.0
            for ticker, current_shares in current_holdings.items():
                if current_shares <= 0:
                    continue
                execution_price = get_execution_price(date, ticker, price_data, side="sell")
                if execution_price is None:
                    new_holdings[ticker] = current_shares
                    continue
                current_position_value = current_shares * execution_price
                target_position_value = target_weights.get(ticker, 0) * rebalance_nav
                if target_position_value >= current_position_value:
                    new_holdings[ticker] = current_shares
                    continue
                if target_position_value < current_position_value - 1:
                    sell_amount = current_position_value - target_position_value
                    shares_to_sell = sell_amount / execution_price
                    total_sell_proceeds += sell_amount
                    total_shares_sold += shares_to_sell
                    new_holdings[ticker] = max(0, current_shares - shares_to_sell)
                    trade_records.append({
                        "date": date, "ticker": ticker, "action": "SELL",
                        "shares": shares_to_sell, "price": execution_price,
                        "value": sell_amount,
                    })
                else:
                    new_holdings[ticker] = current_shares

            available_cash += total_sell_proceeds - transaction_costs(
                total_sell_proceeds, total_shares_sold
            )

            # Buys
            pending_buys = {}
            for ticker, target_weight in target_weights.items():
                execution_price = get_execution_price(date, ticker, price_data, side="buy")
                if execution_price is None:
                    continue
                current_position_value = new_holdings.get(ticker, 0) * execution_price
                target_position_value = target_weight * rebalance_nav
                if target_position_value > current_position_value + 1:
                    buy_amount = target_position_value - current_position_value
                    pending_buys[ticker] = (buy_amount, execution_price)

            total_buy_value = sum(amt for amt, _ in pending_buys.values())
            if total_buy_value > available_cash and total_buy_value > 0:
                scale = available_cash / total_buy_value
                pending_buys = {t: (amt * scale, px) for t, (amt, px) in pending_buys.items()}
                total_buy_value = available_cash

            for ticker, (buy_amount, execution_price) in pending_buys.items():
                shares_to_buy = buy_amount / execution_price
                new_holdings[ticker] = new_holdings.get(ticker, 0) + shares_to_buy
                trade_records.append({
                    "date": date, "ticker": ticker, "action": "BUY",
                    "shares": shares_to_buy, "price": execution_price,
                    "value": buy_amount,
                })
            available_cash -= total_buy_value

            current_holdings = {
                t: s for t, s in new_holdings.items() if s > 1e-6
            }
            rebalance_date_list.append(date)
            weight_records.append({"date": date, **target_weights})
            if attention_weights:
                attention_records.append({"date": date, **attention_weights})

            # Anchor each holding's stop-loss reference to its rebalance
            # execution price; the stop buffer resets per period.
            if stop_loss_pct:
                entry_prices = {}
                for ticker in current_holdings:
                    anchor_price = get_execution_price(
                        date, ticker, price_data, side="buy")
                    if anchor_price is None:
                        anchor_price = (
                            float(close_panel.loc[date, ticker])
                            if ticker in close_panel.columns else None
                        )
                    if anchor_price is not None:
                        entry_prices[ticker] = anchor_price

        days_since_rebalance += 1
        days_since_retrain += 1

        if progress_callback:
            progress_callback(
                day_index, len(trading_days), date, portfolio_nav,
                nav_by_date, weight_records, trade_records,
            )

    result.nav_series = pd.Series(nav_by_date, name=name)
    result.returns_series = result.nav_series.pct_change().dropna()
    result.weight_history = (
        pd.DataFrame(weight_records).set_index("date").fillna(0)
        if weight_records else pd.DataFrame()
    )
    result.attention_history = attention_records
    result.trade_log = trade_records
    result.rebalance_dates = rebalance_date_list
    result.stop_out_count = n_stop_outs
    result.compute_metrics()
    m = result.metrics
    stop_suffix = f" | Stops={n_stop_outs}" if stop_loss_pct else ""
    print(f"[{name:20s}] Sharpe={m.get('Sharpe Ratio', 0):+.3f} | "
          f"Return={m.get('Total Return', 0) * 100:+.1f}% | "
          f"MaxDD={m.get('Max Drawdown', 0) * 100:.1f}% | "
          f"{len(trade_records)} trades{stop_suffix}")
    return result


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_spy_bah(spy_returns, start=BACKTEST_START, end=BACKTEST_END,
                initial_nav=INITIAL_NAV):
    date_mask = (spy_returns.index >= start) & (spy_returns.index <= end)
    spy_period_returns = spy_returns[date_mask]
    nav_series = initial_nav * (1 - SLIPPAGE_BPS / 10_000) * (1 + spy_period_returns).cumprod()
    result = BacktestResult(name="SPY Buy-and-Hold")
    result.nav_series = pd.Series(nav_series, name="SPY Buy-and-Hold")
    result.returns_series = spy_period_returns
    result.compute_metrics()
    m = result.metrics
    print(f"[{'SPY Buy-and-Hold':20s}] Sharpe={m.get('Sharpe Ratio', 0):+.3f} | "
          f"Return={m.get('Total Return', 0) * 100:+.1f}% | "
          f"MaxDD={m.get('Max Drawdown', 0) * 100:.1f}%")
    return result


def run_equal_weight(price_data, start=BACKTEST_START, end=BACKTEST_END,
                     initial_nav=INITIAL_NAV):
    close_panel = _build_panel(price_data, start, end, "Close")
    daily_returns = close_panel.pct_change().fillna(0)
    n_stocks = close_panel.shape[1]
    target_weight = 1.0 / n_stocks

    for day_index, date in enumerate(daily_returns.index):
        if day_index > 0 and day_index % REBALANCE_DAYS == 0:
            cum_returns = (1 + daily_returns.iloc[max(0, day_index - REBALANCE_DAYS):day_index]).prod()
            drift_weights = cum_returns / cum_returns.sum()
            turnover = (drift_weights - target_weight).abs().sum() / 2
            daily_returns.loc[date] -= SLIPPAGE_BPS / 10_000 * turnover

    portfolio_returns = daily_returns.mean(axis=1)
    nav_series = initial_nav * (1 + portfolio_returns).cumprod()
    result = BacktestResult(name="Equal-Weight(1/N)")
    result.nav_series = pd.Series(nav_series, name="Equal-Weight(1/N)")
    result.returns_series = portfolio_returns
    result.compute_metrics()
    m = result.metrics
    print(f"[{'Equal-Weight(1/N)':20s}] Sharpe={m.get('Sharpe Ratio', 0):+.3f} | "
          f"Return={m.get('Total Return', 0) * 100:+.1f}% | "
          f"MaxDD={m.get('Max Drawdown', 0) * 100:.1f}%")
    return result
