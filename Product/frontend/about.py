"""About / Home page for the Adaptive Fusion POC demo app."""

import pandas as pd
import streamlit as st

from backend.config import (
    METRICS_PATH,
    NEWS_SENTIMENT_DIR,
    RAW_NEWS_DIR,
    RAW_TWEETS_DIR,
    SOCIAL_SENTIMENT_DIR,
    TICKERS,
)

st.title("Adaptive Fusion POC")
st.markdown(
    "A real-time demonstration of the adaptive sentiment fusion portfolio strategy. "
    "Use the pages in the sidebar to run data collection and simulate the portfolio."
)
st.divider()

# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------
st.subheader("System Status")


def count_files(directory, pattern="*.csv"):
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Raw News Files", f"{count_files(RAW_NEWS_DIR)} / {len(TICKERS)}")
    st.metric("Raw Tweet Files", f"{count_files(RAW_TWEETS_DIR)} / {len(TICKERS)}")

with col2:
    st.metric("News Sentiment Files", f"{count_files(NEWS_SENTIMENT_DIR)} / {len(TICKERS)}")
    st.metric("Tweet Sentiment Files", f"{count_files(SOCIAL_SENTIMENT_DIR)} / {len(TICKERS)}")

with col3:
    # Scan processed sentiment CSVs for the overall data date range
    _all_dates = []
    for _dir in (NEWS_SENTIMENT_DIR, SOCIAL_SENTIMENT_DIR):
        if _dir.exists():
            for _csv in _dir.glob("*.csv"):
                try:
                    _dates = pd.read_csv(_csv, usecols=["date"])["date"]
                    _all_dates.extend(_dates.tolist())
                except Exception:
                    pass
    if _all_dates:
        _parsed = pd.to_datetime(_all_dates)
        st.metric("Data Available", f"{_parsed.min().date()} to {_parsed.max().date()}")
    else:
        st.metric("Data Available", "--")

st.divider()

# ---------------------------------------------------------------------------
# Performance snapshot
# ---------------------------------------------------------------------------
st.subheader("Strategy Performance Snapshot")

if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)
    fmt = {
        "Sharpe Ratio":      "{:.4f}",
        "Annualised Return": "{:.2%}",
        "Annualised Vol":    "{:.2%}",
        "Max Drawdown":      "{:.2%}",
        "Calmar Ratio":      "{:.4f}",
        "Total Return":      "{:.2%}",
    }
    display_df = metrics_df.copy()
    for col, f in fmt.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f.format(x))
    st.dataframe(display_df.set_index("Strategy"), width="stretch")
else:
    st.info("Metrics summary not found. Run the portfolio simulation first.")

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.subheader("Navigation Guide")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "**Data Collection**\n"
        "- Launch GDELT news scraper for any date range and ticker subset\n"
        "- Launch Twitter/X scraper in a controlled browser session\n"
        "- Monitor live logs as scraping progresses"
    )
with c2:
    st.markdown(
        "**Portfolio Simulation**\n"
        "- Select strategy (Price-Only, Static-Fusion, Adaptive Fixed, Adaptive WF)\n"
        "- Train the Adaptive Fusion neural network live\n"
        "- Run backtest and view NAV, drawdown, weights, attention, and trade log\n"
        "- Compare all strategies side-by-side against SPY and Equal-Weight benchmarks"
    )
