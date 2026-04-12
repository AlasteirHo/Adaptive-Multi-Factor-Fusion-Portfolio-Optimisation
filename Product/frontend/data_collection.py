"""Data Collection page.

Lets the user start / monitor GDELT news and Twitter/X scrapers in real time.
Each scraper runs as a subprocess; stdout is streamed line-by-line into a
scrollable log area.  Both scrapers can run simultaneously (one instance each).

Also provides a FIN-RoBERTa sentiment classification panel that processes
raw scraped data into daily sentiment CSVs consumed by the backtest pipeline.
"""

import queue
import subprocess
import sys
import threading
import time
from datetime import date, timedelta

import streamlit as st

from backend.config import PYTHON_EXE, RUNNERS_DIR, TICKERS

st.title("Data Collection")
st.markdown(
    "Start a news or tweet scraping session and watch the log stream in real time. "
    "Scraped data lands in `Raw_Data/` exactly as the standalone scripts produce it. "
    "Then use the **Sentiment Classification** section below to label the raw data "
    "with FIN-RoBERTa."
)
st.divider()

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
for key, default in {
    "gdelt_proc": None,      "gdelt_log": [],      "gdelt_q": None,
    "twitter_proc": None,    "twitter_log": [],    "twitter_q": None,
    "sentiment_proc": None,  "sentiment_log": [],  "sentiment_q": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enqueue_stdout(proc, q):
    """Thread target: read process stdout line by line and put onto queue."""
    try:
        for line in iter(proc.stdout.readline, ""):
            q.put(line)
    finally:
        q.put(None)  # sentinel: process finished


def _is_running(state_prefix):
    """Check if a scraper subprocess is currently alive."""
    proc = st.session_state.get(f"{state_prefix}_proc")
    return proc is not None and proc.poll() is None


def _start_scraper(runner_script, extra_args, state_prefix):
    """Launch a scraper subprocess. No-op if one is already running."""
    if _is_running(state_prefix):
        return

    cmd = [PYTHON_EXE, "-u", str(runner_script)] + extra_args
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    q = queue.Queue()
    thread = threading.Thread(target=_enqueue_stdout, args=(proc, q), daemon=True)
    thread.start()
    st.session_state[f"{state_prefix}_proc"] = proc
    st.session_state[f"{state_prefix}_log"] = []
    st.session_state[f"{state_prefix}_q"] = q


def _stop_scraper(state_prefix):
    proc = st.session_state.get(f"{state_prefix}_proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    st.session_state[f"{state_prefix}_proc"] = None
    st.session_state[f"{state_prefix}_q"] = None


def _drain_queue(state_prefix, max_lines=500):
    """Drain whatever is in the queue into the log list. Returns True if done."""
    q = st.session_state.get(f"{state_prefix}_q")
    if q is None:
        return True
    done = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            done = True
            break
        st.session_state[f"{state_prefix}_log"].append(line.rstrip())
        if len(st.session_state[f"{state_prefix}_log"]) > max_lines:
            st.session_state[f"{state_prefix}_log"].pop(0)
    return done


def _render_scraper_panel(title, state_prefix, runner_script, default_tickers,
                          extra_notes=""):
    """Render a single scraper control panel. Does NOT call st.rerun()."""
    st.subheader(title)

    running = _is_running(state_prefix)

    col_dates, col_tickers = st.columns([1, 2])
    with col_dates:
        start_date = st.date_input(
            "Start date", value=date.today() - timedelta(days=7),
            key=f"{state_prefix}_start", disabled=running,
        )
        end_date = st.date_input(
            "End date", value=date.today(), key=f"{state_prefix}_end",
            disabled=running,
        )
    with col_tickers:
        selected_tickers = st.multiselect(
            "Tickers", options=TICKERS, default=default_tickers,
            key=f"{state_prefix}_tickers", disabled=running,
        )

    btn_col1, btn_col2, status_col = st.columns([1, 1, 4])
    with btn_col1:
        start_clicked = st.button(
            "Start", key=f"{state_prefix}_start_btn",
            disabled=running or not selected_tickers or start_date > end_date,
            type="primary",
        )
    with btn_col2:
        stop_clicked = st.button(
            "Stop", key=f"{state_prefix}_stop_btn", disabled=not running,
        )
    with status_col:
        if running:
            st.warning("Instance running (max 1 at a time).")
        else:
            proc = st.session_state.get(f"{state_prefix}_proc")
            if proc is not None:
                rc = proc.poll()
                if rc == 0:
                    st.success("Completed")
                else:
                    st.error(f"Exited with code {rc}")
            else:
                st.info("Idle")

    # --- actions (no st.rerun here; handled at page level) ---
    needs_rerun = False

    if start_clicked and not _is_running(state_prefix):
        args = [
            "--start", str(start_date), "--end", str(end_date),
            "--tickers", *selected_tickers,
        ]
        _start_scraper(runner_script, args, state_prefix)
        needs_rerun = True

    if stop_clicked and running:
        _stop_scraper(state_prefix)
        st.session_state[f"{state_prefix}_log"].append("[Stopped by user]")
        needs_rerun = True

    if extra_notes:
        st.caption(extra_notes)

    # Drain queue every render (picks up output even after process exits)
    _drain_queue(state_prefix)

    log_lines = st.session_state[f"{state_prefix}_log"]
    log_text = "\n".join(log_lines) if log_lines else "No output yet."
    st.session_state[f"{state_prefix}_log_area"] = log_text
    st.text_area(
        "Log output", height=300,
        key=f"{state_prefix}_log_area", disabled=True, label_visibility="collapsed",
    )

    return needs_rerun


# ---------------------------------------------------------------------------
# Layout: both panels render fully before any rerun
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    rerun_gdelt = _render_scraper_panel(
        title="GDELT News Scraper",
        state_prefix="gdelt",
        runner_script=RUNNERS_DIR / "gdelt_runner.py",
        default_tickers=["AAPL", "MSFT", "NVDA"],
        extra_notes=(
            "Queries the GDELT API for financial news headlines. "
            "Results are appended to Raw_Data/gdelt_news_data/. "
            "Rate limited to ~3 s per request."
        ),
    )

with col_right:
    rerun_twitter = _render_scraper_panel(
        title="Twitter / X Scraper",
        state_prefix="twitter",
        runner_script=RUNNERS_DIR / "twitter_runner.py",
        default_tickers=["AAPL", "MSFT", "NVDA"],
        extra_notes=(
            "Opens a Chrome browser and scrapes tweets for each ticker. "
            "Credentials are read from the .env file automatically. "
            "Requires an active internet connection and Chrome installed. "
            "Keep this browser tab focused while scraping is in progress "
            "for maximum performance; background tabs may be throttled by the OS."
        ),
    )

# ---------------------------------------------------------------------------
# Sentiment Classification panel
# ---------------------------------------------------------------------------
st.divider()
st.header("Sentiment Classification")
st.caption(
    "Run FIN-RoBERTa (alasteirho/FIN-RoBERTa-Custom) on raw scraped data to "
    "produce daily sentiment CSVs in Processed_Data/. The model scores each "
    "headline or tweet as P(positive) - P(negative), then aggregates by NYSE "
    "trading session (16:00 ET cutoff)."
)

sent_running = _is_running("sentiment")

col_source, col_tickers_sent = st.columns([1, 2])
with col_source:
    source_option = st.selectbox(
        "Data source",
        options=["both", "news", "tweets"],
        format_func=lambda x: {"both": "News + Tweets", "news": "News only", "tweets": "Tweets only"}[x],
        key="sentiment_source",
        disabled=sent_running,
    )
with col_tickers_sent:
    sent_tickers = st.multiselect(
        "Tickers", options=TICKERS, default=TICKERS,
        key="sentiment_tickers", disabled=sent_running,
    )

btn1, btn2, status = st.columns([1, 1, 4])
with btn1:
    sent_start = st.button(
        "Classify", key="sentiment_start_btn",
        disabled=sent_running or not sent_tickers,
        type="primary",
    )
with btn2:
    sent_stop = st.button(
        "Stop", key="sentiment_stop_btn", disabled=not sent_running,
    )
with status:
    if sent_running:
        st.warning("Classification running...")
    else:
        proc = st.session_state.get("sentiment_proc")
        if proc is not None:
            rc = proc.poll()
            if rc == 0:
                st.success("Classification complete")
            else:
                st.error(f"Exited with code {rc}")
        else:
            st.info("Idle")

rerun_sentiment = False

if sent_start and not sent_running:
    args = ["--source", source_option, "--tickers", *sent_tickers]
    _start_scraper(RUNNERS_DIR / "sentiment_runner.py", args, "sentiment")
    rerun_sentiment = True

if sent_stop and sent_running:
    _stop_scraper("sentiment")
    st.session_state["sentiment_log"].append("[Stopped by user]")
    rerun_sentiment = True

_drain_queue("sentiment")

sent_log_lines = st.session_state["sentiment_log"]
sent_log_text = "\n".join(sent_log_lines) if sent_log_lines else "No output yet."
st.session_state["sentiment_log_area"] = sent_log_text
st.text_area(
    "Log output", height=300,
    key="sentiment_log_area", disabled=True, label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Single rerun point AFTER all panels have rendered
# ---------------------------------------------------------------------------
if rerun_gdelt or rerun_twitter or rerun_sentiment:
    st.rerun()

if _is_running("gdelt") or _is_running("twitter") or _is_running("sentiment"):
    time.sleep(0.5)
    st.rerun()
