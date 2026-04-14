"""
Data loading: price data (yfinance), sentiment CSVs, and master dataset assembly.
Mirrors notebook cell 4.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    BENCHMARK_TICKER,
    DATA_START,
    BACKTEST_END,
    DATE_COL,
    NEWS_CSV_SUFFIX,
    NEWS_SENTIMENT_DIR,
    SENTIMENT_SCORE_COL,
    SOCIAL_CSV_SUFFIX,
    SOCIAL_SENTIMENT_DIR,
    TICKERS,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)


def fetch_price_data(tickers=None, start=DATA_START, end=BACKTEST_END):
    tickers = tickers or TICKERS
    price_data = {}
    print(f"Fetching daily OHLCV for {len(tickers)} tickers ...", end=" ")
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index).normalize()
            price_data[ticker] = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as error:
            print(f"\n  Warning: {ticker} - {error}")
    print(f"done ({len(price_data)}/{len(tickers)})")
    return price_data


def _load_csv(path, label):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=[DATE_COL])
        df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize()
        df = df.drop_duplicates(DATE_COL).set_index(DATE_COL)
        sentiment_series = df[SENTIMENT_SCORE_COL].astype(float)
        sentiment_series.name = label
        return sentiment_series
    except Exception:
        return None


def _social_ticker(ticker):
    return ticker.replace("-", ".")


def load_sentiment_data(tickers=None, start=DATA_START, end=BACKTEST_END,
                        news_dir=None, social_dir=None):
    tickers = tickers or TICKERS
    news_dir = news_dir or NEWS_SENTIMENT_DIR
    social_dir = social_dir or SOCIAL_SENTIMENT_DIR
    print(f"Loading sentiment from:\n  News   : {news_dir}\n  Social : {social_dir}")
    date_range = pd.bdate_range(start=start, end=end, freq="C")
    sentiment = {}
    missing_news, missing_social = [], []
    for ticker in tickers:
        news_path = news_dir / f"{ticker}{NEWS_CSV_SUFFIX}"
        social_path = social_dir / f"{_social_ticker(ticker)}{SOCIAL_CSV_SUFFIX}"
        news_series = _load_csv(news_path, "news_sentiment")
        social_series = _load_csv(social_path, "social_sentiment")
        if news_series is None:
            missing_news.append(ticker)
        if social_series is None:
            missing_social.append(ticker)
        aligned_df = pd.DataFrame(index=date_range)
        aligned_df["news_sentiment"] = (
            news_series.reindex(date_range) if news_series is not None else 0.0
        )
        aligned_df["social_sentiment"] = (
            social_series.reindex(date_range) if social_series is not None else 0.0
        )
        aligned_df["news_available"] = (
            news_series.reindex(date_range).notna().astype(float)
            if news_series is not None else 0.0
        )
        sentiment[ticker] = aligned_df.fillna(0.0).clip(-1.0, 1.0)
    print(f"Loaded sentiment for {len(sentiment)}/{len(tickers)} tickers")
    if missing_news:
        print(f"  Missing news CSVs  : {missing_news}")
    if missing_social:
        print(f"  Missing social CSVs: {missing_social}")
    return sentiment


def build_master_dataset(price_data, sentiment_data):
    master = {}
    for ticker in price_data:
        price_df = price_data[ticker].copy()
        if ticker in sentiment_data:
            merged = price_df.join(sentiment_data[ticker].shift(1), how="left")
        else:
            merged = price_df.copy()
            merged[["news_sentiment", "social_sentiment", "news_available"]] = 0.0
        master[ticker] = merged.fillna(0.0)
    return master


def fetch_spy_returns(start=None, end=None):
    from .config import BACKTEST_START, BACKTEST_END
    start = start or BACKTEST_START
    end = end or BACKTEST_END
    print(f"Fetching SPY benchmark ({start} to {end})...")
    spy = yf.download(BENCHMARK_TICKER, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).normalize()
    ret = spy["Close"].pct_change().dropna().rename("SPY")
    print(f"  SPY: {len(ret)} trading days")
    return ret


def load_all_data(tickers=None, start=DATA_START, end=BACKTEST_END):
    """Convenience: fetch prices, sentiment, build master, and SPY returns."""
    tickers = tickers or TICKERS
    price_data = fetch_price_data(tickers, start, end)
    sentiment_data = load_sentiment_data(tickers, start, end)
    master_data = build_master_dataset(price_data, sentiment_data)
    print(f"Master dataset ready: {len(master_data)} tickers")
    spy_returns = fetch_spy_returns()
    return price_data, sentiment_data, master_data, spy_returns
