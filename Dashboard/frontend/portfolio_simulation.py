"""Portfolio Simulation page.

Loads data, trains the model, runs backtests, and displays interactive results.
A Play button animates the simulation day by day (1 second per trading day).
Rebalance markers appear as the simulation progresses.  Stop freezes in place.
"""

import contextlib
import io
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.config import (
    BACKTEST_END,
    BACKTEST_START,
    DEVICE,
    DEVICE_NAME,
    FACTOR_COLS,
    REBALANCE_DAYS,
    RETRAIN_EVERY,
    STOP_LOSS_PCT,
    TOP_N_STOCKS,
    USE_AMP,
)
from backend.data import load_all_data
from backend.features import build_features
from backend.model import load_or_train
from backend.backtest import (
    run_backtest,
    run_equal_weight,
    run_spy_bah,
)

st.title("Portfolio Simulation")

# ---------------------------------------------------------------------------
# Modern button CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stButton > button {
        border-radius: 2rem !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.6rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.25) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    .stDownloadButton > button {
        border-radius: 2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chart heights & colour palette
# ---------------------------------------------------------------------------
CHART_HEIGHT_LARGE = 500
CHART_HEIGHT_MED   = 400
CHART_HEIGHT_SMALL = 320

PALETTE = [
    "#6C63FF",   # primary purple
    "#48C9B0",   # teal
    "#F39C12",   # amber
    "#E74C3C",   # red
    "#3498DB",   # blue
    "#1ABC9C",   # mint
    "#9B59B6",   # violet
    "#E67E22",   # orange
]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.6)",
    font=dict(family="Inter, sans-serif", color="#E0E0E0"),
    colorway=PALETTE,
    xaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.15)"),
    yaxis=dict(gridcolor="rgba(108,99,255,0.08)", zerolinecolor="rgba(108,99,255,0.15)"),
)


# ---------------------------------------------------------------------------
# Stdout capture helper
# ---------------------------------------------------------------------------
class _StreamCapture(io.StringIO):
    def __init__(self, original):
        super().__init__()
        self._original = original

    def write(self, s):
        self._original.write(s)
        return super().write(s)

    def flush(self):
        self._original.flush()
        super().flush()


@contextlib.contextmanager
def capture_stdout():
    import sys
    buf = _StreamCapture(sys.stdout)
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics_from_nav(nav):
    """Compute strategy metrics from a NAV series."""
    if len(nav) < 2:
        return {}
    rets = nav.pct_change().dropna()
    if len(rets) < 1:
        return {}
    ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    return {
        "Sharpe Ratio":      round(sharpe, 4),
        "Annualised Return": round(ann_ret, 4),
        "Annualised Vol":    round(ann_vol, 4),
        "Max Drawdown":      round(max_dd, 4),
        "Calmar Ratio":      round(calmar, 4),
        "Total Return":      round(total_ret, 4),
    }


def _format_metrics_df(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("Strategy")
    fmt = {
        "Sharpe Ratio": "{:.4f}",
        "Annualised Return": "{:.2%}",
        "Annualised Vol": "{:.2%}",
        "Max Drawdown": "{:.2%}",
        "Calmar Ratio": "{:.4f}",
        "Total Return": "{:.2%}",
    }
    display = df.copy()
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x, fmt=f: fmt.format(x) if pd.notna(x) else "--"
            )
    return display


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

strategy_options = {
    "Price-Only (no sentiment)":    {"use_sentiment": False, "use_adaptive": False},
    "Static-Fusion (equal weight)": {"use_sentiment": True,  "use_adaptive": False},
    "Adaptive Fixed":               {"use_sentiment": True,  "use_adaptive": True, "retrain": 0},
    "Adaptive Walk-Forward":        {"use_sentiment": True,  "use_adaptive": True, "retrain": RETRAIN_EVERY},
}

selected_strategies = st.sidebar.multiselect(
    "Strategies to run",
    options=list(strategy_options.keys()),
    default=["Adaptive Fixed"],
)

run_benchmarks = st.sidebar.checkbox("Include SPY & Equal-Weight benchmarks", value=True)

# ---- Stop-loss controls --------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Stop-Loss")
use_stop_loss = st.sidebar.toggle(
    "Enable stop-loss",
    value=False,
    help=(
        "When enabled, any held position whose intraday Low falls below "
        "entry_price * (1 - threshold) is liquidated at the stop price, "
        "with cash held until the next rebalance."
    ),
)
stop_loss_pct_input = st.sidebar.number_input(
    "Stop-loss threshold (%)",
    min_value=0.1,
    max_value=50.0,
    value=float(STOP_LOSS_PCT * 100),
    step=0.1,
    disabled=not use_stop_loss,
    help="Per-position drawdown at which a holding is liquidated.",
)
stop_loss_strategies = st.sidebar.multiselect(
    "Apply stop-loss to",
    options=selected_strategies,
    default=selected_strategies if use_stop_loss else [],
    disabled=not use_stop_loss,
    help="Choose which strategies use the stop-loss rule.",
)
stop_loss_pct_value = (stop_loss_pct_input / 100.0) if use_stop_loss else None

# ---- Date range controls -------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Backtest Period")
_default_start = pd.Timestamp(BACKTEST_START).date()
_default_end = pd.Timestamp(BACKTEST_END).date()
backtest_start_date = st.sidebar.date_input(
    "Start date",
    value=_default_start,
    help="First trading day of the backtest window. Default is the project's backtest start.",
)
backtest_end_date = st.sidebar.date_input(
    "End date",
    value=_default_end,
    help="Last trading day of the backtest window. Default is the latest available data.",
)
if backtest_end_date <= backtest_start_date:
    st.sidebar.error("End date must be after start date.")

backtest_start_str = backtest_start_date.isoformat()
backtest_end_str = backtest_end_date.isoformat()

st.sidebar.divider()
if use_stop_loss and stop_loss_strategies:
    _stop_loss_label = f"{stop_loss_pct_input:.1f}% ({len(stop_loss_strategies)} strategies)"
else:
    _stop_loss_label = "Off"
st.sidebar.markdown(
    f"**Config snapshot**\n"
    f"- Backtest: {backtest_start_str} to {backtest_end_str}\n"
    f"- Top N stocks: {TOP_N_STOCKS}\n"
    f"- Rebalance: every {REBALANCE_DAYS} days\n"
    f"- Stop-loss: {_stop_loss_label}\n"
    f"- Device: {DEVICE_NAME}\n"
    f"- Mixed precision: {USE_AMP}"
)

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

if not selected_strategies:
    st.info("Select at least one strategy from the sidebar.")
    st.stop()

if backtest_end_date <= backtest_start_date:
    st.info("Fix the backtest date range in the sidebar before running.")
    st.stop()

run_btn = st.button("Run Simulation", type="primary", width="stretch")

if run_btn:
    # Reset playback state for fresh run
    st.session_state["playing"] = False
    st.session_state.pop("play_idx", None)
    st.session_state.pop("inspect_date", None)
    try:
        # ---- Step 1: Load data ----
        with st.status("Loading data...", expanded=True) as status:
            with capture_stdout() as buf:
                price_data, sentiment_data, master_data, spy_returns = load_all_data()
                feature_data = build_features(master_data)
            st.code(buf.getvalue(), language="text")
            status.update(
                label=f"Data loaded: {len(feature_data)} tickers, {len(FACTOR_COLS)} factors",
                state="complete",
            )

        # ---- Step 2: Train model if needed ----
        needs_model = any(
            strategy_options[s].get("use_adaptive", False)
            for s in selected_strategies
        )

        model = None
        if needs_model:
            with st.status("Training Adaptive Fusion Network...", expanded=True) as status:
                train_bar = st.progress(0, text="Initialising...")
                train_log = st.empty()

                def _train_progress(epoch, train_ic, val_ic, total):
                    pct = min(epoch / total, 1.0)
                    train_bar.progress(pct, text=f"Epoch {epoch}/{total}")
                    train_log.text(
                        f"Train IC: {train_ic:.4f} | Val IC: {val_ic:.4f} | "
                        f"Gap: {train_ic - val_ic:+.4f}"
                    )

                with capture_stdout() as buf:
                    model, train_hist, val_hist = load_or_train(
                        feature_data,
                        force_retrain=True,
                        progress_callback=_train_progress,
                    )
                train_bar.progress(1.0, text="Training complete")
                st.code(buf.getvalue(), language="text")
                status.update(label="Training complete", state="complete")

                if train_hist:
                    fig_train = go.Figure()
                    fig_train.add_trace(go.Scatter(
                        y=[-v for v in train_hist], name="Train IC", mode="lines",
                    ))
                    fig_train.add_trace(go.Scatter(
                        y=[-v for v in val_hist], name="Val IC", mode="lines",
                    ))
                    fig_train.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Training Curves (IC per epoch)",
                        xaxis_title="Epoch", yaxis_title="Information Coefficient",
                        height=CHART_HEIGHT_SMALL, autosize=True,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_train, width="stretch", key="train_curve")

        # ---- Step 3: Run strategies ----
        with st.status("Running backtests...", expanded=True) as status:
            bt_bar = st.progress(0, text="Starting...")
            completed = {}
            n_total = len(selected_strategies)

            for i, strat_name in enumerate(selected_strategies):
                opts = strategy_options[strat_name]

                def _bt_progress(day_idx, total, dt, nav,
                                 nav_history, weight_records, trade_records,
                                 _i=i, _name=strat_name):
                    inner_pct = day_idx / max(total, 1)
                    overall = (_i + inner_pct) / n_total
                    bt_bar.progress(
                        min(overall, 1.0),
                        text=f"{_name}: day {day_idx}/{total}  |  NAV: ${nav:,.2f}",
                    )

                with capture_stdout() as buf:
                    result = run_backtest(
                        name=strat_name,
                        feature_data=feature_data,
                        price_data=price_data,
                        model=model if opts.get("use_adaptive") else None,
                        start=backtest_start_str,
                        end=backtest_end_str,
                        use_sentiment=opts["use_sentiment"],
                        use_adaptive=opts.get("use_adaptive", False),
                        retrain_every=opts.get("retrain", 0),
                        stop_loss_pct=stop_loss_pct_value if strat_name in stop_loss_strategies else None,
                        progress_callback=_bt_progress,
                    )
                st.code(buf.getvalue(), language="text")
                completed[strat_name] = result

            if run_benchmarks:
                bt_bar.progress(0.95, text="Running benchmarks...")
                with capture_stdout() as buf:
                    completed["SPY Buy-and-Hold"] = run_spy_bah(
                        spy_returns,
                        start=backtest_start_str,
                        end=backtest_end_str,
                    )
                    completed["Equal-Weight"] = run_equal_weight(
                        price_data,
                        start=backtest_start_str,
                        end=backtest_end_str,
                    )
                st.code(buf.getvalue(), language="text")

            bt_bar.progress(1.0, text="All backtests complete")
            status.update(label="All backtests complete", state="complete")

        st.session_state["sim_results"] = completed
        st.session_state["sim_metrics"] = {
            name: res.metrics for name, res in completed.items()
        }

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.code(traceback.format_exc(), language="text")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "sim_results" not in st.session_state:
    st.stop()

results = st.session_state["sim_results"]

st.divider()
st.header("Results")

# ---- Collect all trading days ----
all_days = sorted(set().union(*(
    r.nav_series.index.tolist()
    for r in results.values() if len(r.nav_series) > 0
)))

if not all_days:
    st.warning("No trading days found in results.")
    st.stop()

# ---- Pre-compute fixed axis ranges ----
_all_navs = [r.nav_series for r in results.values() if len(r.nav_series) > 0]
_y_min = min(n.min() for n in _all_navs) * 0.97
_y_max = max(n.max() for n in _all_navs) * 1.03

# ---- Playback state ----
# States:
#   idle      : play_idx absent           -> empty chart
#   playing   : play_idx present, playing  -> animating
#   paused    : play_idx present, !playing -> frozen at position
#   complete  : play_idx == last, !playing -> full data, markers clickable

is_playing = st.session_state.get("playing", False)
show_empty = "play_idx" not in st.session_state

if show_empty:
    cutoff_idx = 0
    cutoff = all_days[0]
else:
    play_idx = st.session_state["play_idx"]
    cutoff_idx = min(play_idx, len(all_days) - 1)
    cutoff = all_days[cutoff_idx]

is_complete = (not show_empty
               and not is_playing
               and cutoff_idx >= len(all_days) - 1)


def _cut(obj):
    """Slice a Series or DataFrame index to the playback cutoff."""
    if show_empty:
        if isinstance(obj, pd.Series):
            return obj.iloc[:0]
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:0]
        return obj
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj[obj.index <= cutoff]
    return obj


# ---- Performance metrics (hidden when idle) ----
if not show_empty:
    st.subheader("Performance Metrics")
    cached_metrics = st.session_state.get("sim_metrics", {})
    metric_rows = []
    for name, res in results.items():
        if is_complete and name in cached_metrics:
            m = cached_metrics[name]
        else:
            nav_slice = _cut(res.nav_series)
            m = _metrics_from_nav(nav_slice) if len(nav_slice) >= 2 else {}
        if m:
            metric_rows.append({"Strategy": name, **m})
    mdf = _format_metrics_df(metric_rows)
    if mdf is not None:
        st.dataframe(mdf, width="stretch")

# ---- NAV chart header + playback controls ----
st.subheader("NAV Comparison")

play_col, stop_col, info_col = st.columns([1, 1, 6])
with play_col:
    if st.button("\u25B6  Play", key="play_btn"):
        st.session_state["playing"] = True
        st.session_state["play_idx"] = 0
        st.session_state.pop("inspect_date", None)
        st.rerun()
with stop_col:
    if st.button("\u25A0  Stop", key="stop_btn"):
        st.session_state["playing"] = False
        # play_idx is kept so we freeze at the current position
        st.rerun()

# Re-read state after button callbacks
is_playing = st.session_state.get("playing", False)
show_empty = "play_idx" not in st.session_state
if not show_empty:
    play_idx = st.session_state["play_idx"]
    cutoff_idx = min(play_idx, len(all_days) - 1)
    cutoff = all_days[cutoff_idx]
is_complete = (not show_empty
               and not is_playing
               and cutoff_idx >= len(all_days) - 1)

if is_playing:
    progress_frac = (cutoff_idx + 1) / len(all_days)
    with info_col:
        st.caption(f"Day {cutoff_idx + 1} / {len(all_days)}  |  {cutoff.date()}")
    st.progress(progress_frac)
elif not show_empty and not is_complete:
    with info_col:
        st.caption(f"Paused at day {cutoff_idx + 1} / {len(all_days)}  |  {cutoff.date()}")
elif not show_empty:
    st.caption("Click a rebalance marker to inspect allocation and metrics at that point.")

# ---- Build NAV chart ----
fig_nav = go.Figure()

if show_empty:
    # Empty frame with centered annotation
    fig_nav.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        xaxis_range=[all_days[0], all_days[-1]],
        yaxis_range=[_y_min, _y_max],
        height=CHART_HEIGHT_LARGE, autosize=True,
        margin=dict(l=50, r=20, t=30, b=40),
        annotations=[dict(
            text="Press Play to begin simulation",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="rgba(108,99,255,0.5)"),
        )],
    )
    st.plotly_chart(fig_nav, width="stretch", key="nav_empty")

else:
    color_idx = 0
    for name, res in results.items():
        nav = _cut(res.nav_series)
        if len(nav) == 0:
            continue
        color = PALETTE[color_idx % len(PALETTE)]

        # Continuous line
        fig_nav.add_trace(go.Scatter(
            x=nav.index, y=nav.values,
            name=name, mode="lines",
            line=dict(color=color),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>" + name + "</extra>",
        ))

        # Rebalance markers (always shown)
        if res.rebalance_dates:
            mask = nav.index.isin(res.rebalance_dates)
            rebal_nav = nav[mask]
            if len(rebal_nav) > 0:
                can_click = not is_playing
                fig_nav.add_trace(go.Scatter(
                    x=rebal_nav.index,
                    y=rebal_nav.values,
                    mode="markers",
                    marker=dict(
                        size=9, color=color,
                        line=dict(width=1.5, color="white"),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "%{x|%Y-%m-%d}<br>"
                        "NAV: $%{y:,.2f}<br>"
                        + ("<i>Click to inspect</i>" if can_click else "Rebalance")
                        + "<extra></extra>"
                    ),
                ))

        color_idx += 1

    nav_layout = dict(
        **PLOTLY_LAYOUT,
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        xaxis_range=[all_days[0], all_days[-1]],
        yaxis_range=[_y_min, _y_max],
        height=CHART_HEIGHT_LARGE, autosize=True,
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    event = None
    if is_playing:
        fig_nav.update_layout(**nav_layout)
        st.plotly_chart(fig_nav, width="stretch", key="nav_play")
    else:
        nav_layout["clickmode"] = "event+select"
        nav_layout["dragmode"] = False
        fig_nav.update_layout(**nav_layout)
        event = st.plotly_chart(
            fig_nav, on_select="rerun", selection_mode=("points",),
            width="stretch", key="nav_select",
        )

    # ---- Inspection / allocation panel ----
    inspect_date = None
    if is_playing:
        inspect_date = cutoff
    elif event is not None and event.selection.points:
        raw_x = event.selection.points[0]["x"]
        inspect_date = pd.Timestamp(raw_x)
        st.session_state["inspect_date"] = inspect_date
    elif "inspect_date" in st.session_state:
        inspect_date = st.session_state["inspect_date"]
    else:
        # Paused or complete: default to current position / last day
        inspect_date = cutoff

    if inspect_date is not None:
        with st.container(border=True):
            if is_playing:
                st.subheader(f"Day {cutoff_idx + 1}: {inspect_date.date()}")
            else:
                st.subheader(f"Snapshot: {inspect_date.date()}")

                # Point-in-time metrics (manual inspection only)
                snap_rows = []
                for name, res in results.items():
                    nav_slice = res.nav_series[res.nav_series.index <= inspect_date]
                    if len(nav_slice) >= 2:
                        snap_rows.append({"Strategy": name, **_metrics_from_nav(nav_slice)})
                snap_mdf = _format_metrics_df(snap_rows)
                if snap_mdf is not None:
                    st.dataframe(snap_mdf, width="stretch")

            # Pie charts
            pie_strats = {}
            for name, res in results.items():
                if res.weight_history.empty:
                    continue
                wh = res.weight_history[res.weight_history.index <= inspect_date]
                if not wh.empty:
                    pie_strats[name] = wh.iloc[-1]

            if pie_strats:
                pie_cols = st.columns(max(len(pie_strats), 1))
                for col, (name, weights) in zip(pie_cols, pie_strats.items()):
                    with col:
                        w = weights[weights > 0].sort_values(ascending=False)
                        if w.empty:
                            continue
                        rebal_date = weights.name
                        rebal_label = (
                            rebal_date.date()
                            if hasattr(rebal_date, "date") else rebal_date
                        )
                        fig_pie = go.Figure(go.Pie(
                            labels=w.index, values=w.values, hole=0.4,
                            marker=dict(colors=PALETTE[:len(w)]),
                            textinfo="label+percent", textposition="outside",
                        ))
                        fig_pie.update_layout(
                            **PLOTLY_LAYOUT,
                            title=f"{name}<br><sup>Rebalance: {rebal_label}</sup>",
                            height=CHART_HEIGHT_SMALL, autosize=True,
                            margin=dict(l=10, r=10, t=60, b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(
                            fig_pie, width="stretch",
                            key=f"snap_pie_{name}",
                        )
            elif is_playing:
                st.caption("No rebalance has occurred yet.")

    # ---- Drawdown ----
    st.subheader("Drawdown Comparison")
    fig_dd = go.Figure()
    for name, res in results.items():
        nav = _cut(res.nav_series)
        if len(nav) > 0:
            dd = (nav - nav.cummax()) / nav.cummax() * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values, name=name, mode="lines", fill="tozeroy",
            ))
    dd_layout = dict(
        **PLOTLY_LAYOUT,
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        xaxis_range=[all_days[0], all_days[-1]],
        height=CHART_HEIGHT_MED, autosize=True,
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_dd.update_layout(**dd_layout)
    st.plotly_chart(fig_dd, width="stretch", key="dd_chart")

    # ---- Weight evolution ----
    strategies_with_weights = {
        n: r for n, r in results.items() if not r.weight_history.empty
    }
    if strategies_with_weights:
        st.subheader("Weight Evolution")
        weight_tabs = st.tabs(list(strategies_with_weights.keys()))
        for tab, (name, res) in zip(weight_tabs, strategies_with_weights.items()):
            with tab:
                wh = _cut(res.weight_history)
                if wh is None or wh.empty:
                    st.info("No rebalance data yet for this date.")
                    continue
                fig_w = go.Figure()
                for col_name in wh.columns:
                    fig_w.add_trace(go.Scatter(
                        x=wh.index, y=wh[col_name],
                        name=col_name, mode="lines", stackgroup="one",
                    ))
                w_layout = dict(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Date", yaxis_title="Weight",
                    xaxis_range=[all_days[0], all_days[-1]],
                    height=CHART_HEIGHT_MED, autosize=True,
                    margin=dict(l=50, r=20, t=30, b=40),
                    hovermode="x unified",
                )
                fig_w.update_layout(**w_layout)
                st.plotly_chart(fig_w, width="stretch", key=f"wt_{name}")

    # ---- Attention weights ----
    strategies_with_attention = {
        n: r for n, r in results.items() if r.attention_history
    }
    if strategies_with_attention:
        st.subheader("Attention Weights Over Time")
        attn_tabs = st.tabs(list(strategies_with_attention.keys()))
        for tab, (name, res) in zip(attn_tabs, strategies_with_attention.items()):
            with tab:
                attn_df = pd.DataFrame(res.attention_history).set_index("date")
                attn_df = _cut(attn_df)
                rename = {
                    "z_news_sentiment": "News", "z_social_sentiment": "Social",
                    "z_rsi": "RSI", "z_momentum": "Momentum",
                    "z_reversal": "Reversal", "z_abnormal_volume": "Volume",
                    "z_idiovol": "Idiovol", "z_52w_high_ratio": "52w High",
                }
                attn_df = attn_df.rename(columns=rename)
                if attn_df.empty:
                    st.info("No attention data yet for this date.")
                    continue
                fig_a = go.Figure()
                for col_name in attn_df.columns:
                    fig_a.add_trace(go.Scatter(
                        x=attn_df.index, y=attn_df[col_name],
                        name=col_name, mode="lines", stackgroup="one",
                    ))
                a_layout = dict(
                    **PLOTLY_LAYOUT,
                    xaxis_title="Date", yaxis_title="Attention Weight",
                    xaxis_range=[all_days[0], all_days[-1]],
                    height=CHART_HEIGHT_MED, autosize=True,
                    margin=dict(l=50, r=20, t=30, b=40),
                    hovermode="x unified",
                )
                fig_a.update_layout(**a_layout)
                st.plotly_chart(fig_a, width="stretch", key=f"attn_{name}")

                mean_attn = attn_df.mean()
                fig_bar = go.Figure(go.Bar(
                    x=mean_attn.index, y=mean_attn.values,
                    marker_color=PALETTE[:len(mean_attn)],
                ))
                fig_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Mean Attention Weights",
                    yaxis_title="Weight", height=CHART_HEIGHT_SMALL,
                    autosize=True, margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_bar, width="stretch", key=f"attn_bar_{name}")

    # ---- Trade log ----
    strategies_with_trades = {
        n: r for n, r in results.items() if r.trade_log
    }
    if strategies_with_trades:
        st.subheader("Trade Log")
        trade_tabs = st.tabs(list(strategies_with_trades.keys()))
        for tab, (name, res) in zip(trade_tabs, strategies_with_trades.items()):
            with tab:
                trade_df = pd.DataFrame(res.trade_log)
                trade_df = trade_df[pd.to_datetime(trade_df["date"]) <= cutoff]
                if trade_df.empty:
                    st.info("No trades executed yet.")
                    continue
                trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.date
                trade_df["value"] = trade_df["value"].round(2)
                trade_df["price"] = trade_df["price"].round(2)
                trade_df["shares"] = trade_df["shares"].round(4)

                col_summary, col_download = st.columns([3, 1])
                with col_summary:
                    n_buys = (trade_df["action"] == "BUY").sum()
                    n_sells = (trade_df["action"] == "SELL").sum()
                    n_stops = (trade_df["action"] == "STOP").sum()
                    summary_bits = [
                        f"**{n_buys}** buys",
                        f"**{n_sells}** sells",
                    ]
                    if n_stops:
                        summary_bits.append(f"**{n_stops}** stop-outs")
                    st.markdown(
                        f"**{len(trade_df)}** trades total: "
                        + ", ".join(summary_bits)
                    )
                with col_download:
                    csv_data = trade_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV", csv_data,
                        file_name=f"{name.replace(' ', '_')}_trades.csv",
                        mime="text/csv",
                        key=f"dl_{name}",
                    )

                st.dataframe(trade_df, width="stretch", height=400)

    # ---- Return distribution ----
    st.subheader("Daily Return Distribution")
    fig_hist = go.Figure()
    for name, res in results.items():
        rets = _cut(res.returns_series).dropna() * 100
        if len(rets) > 0:
            fig_hist.add_trace(go.Histogram(
                x=rets.values, name=name, opacity=0.6, nbinsx=60,
            ))
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        barmode="overlay", height=CHART_HEIGHT_SMALL, autosize=True,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, width="stretch", key="hist_chart")

# ---------------------------------------------------------------------------
# Auto-advance playback (must be at the very end)
# ---------------------------------------------------------------------------
if is_playing and not show_empty:
    if play_idx < len(all_days) - 1:
        time.sleep(1)
        st.session_state["play_idx"] = play_idx + 1
        st.rerun()
    else:
        # Animation finished: stay at last frame
        st.session_state["playing"] = False
        st.rerun()
