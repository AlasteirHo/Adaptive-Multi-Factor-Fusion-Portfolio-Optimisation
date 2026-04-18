"""Adaptive Fusion POC - Demo Application.

Entry point for the Streamlit multi-page app.
Run with:
    streamlit run main.py
from the Dashboard/ directory.
"""

import sys
from pathlib import Path

# Make backend and frontend importable from all pages
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Adaptive Fusion POC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Responsive CSS: scale elements to fill the viewport
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Layout ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

/* ── Metric cards: glass-morphism style ── */
[data-testid="stMetric"] {
    width: 100%;
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.20);
    border-radius: 12px;
    padding: 14px 18px;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(108, 99, 255, 0.50);
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #6C63FF !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9CA3AF !important;
}

/* ── Section headers ── */
h1 {
    background: linear-gradient(90deg, #6C63FF 0%, #48C9B0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
h2, [data-testid="stSubheader"] {
    color: #B0B8C8 !important;
    border-bottom: 2px solid rgba(108, 99, 255, 0.25);
    padding-bottom: 0.3rem;
}

/* ── Sidebar polish ── */
[data-testid="stSidebar"] {
    background: #12161F !important;
    border-right: 1px solid rgba(108, 99, 255, 0.15);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    background: none !important;
    -webkit-text-fill-color: #E0E0E0 !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #5A54D6) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    transition: box-shadow 0.2s ease;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 18px rgba(108, 99, 255, 0.45) !important;
}
.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid rgba(108, 99, 255, 0.30) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    width: 100%;
}
[data-testid="stTabs"] button[role="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #6C63FF !important;
    color: #6C63FF !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { width: 100% !important; }
[data-testid="stDataFrame"] > div { width: 100% !important; }

/* ── Text areas (scraper logs) ── */
[data-testid="stTextArea"] textarea {
    width: 100% !important;
    resize: vertical;
    background: #0D1017 !important;
    border: 1px solid rgba(108, 99, 255, 0.15) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Plotly charts ── */
[data-testid="stPlotlyChart"] { width: 100% !important; }
[data-testid="stPlotlyChart"] > div { width: 100% !important; }

/* ── Dividers ── */
hr {
    border-color: rgba(108, 99, 255, 0.15) !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
}
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("frontend/about.py",                  title="About",                default=True),
    st.Page("frontend/data_collection.py",        title="Data Collection"),
    st.Page("frontend/portfolio_simulation.py",    title="Portfolio Simulation"),
])
pg.run()
