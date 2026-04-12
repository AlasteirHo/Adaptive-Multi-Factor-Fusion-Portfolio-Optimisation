"""
Constants and hyperparameters for the Adaptive Fusion pipeline.
Mirrors the notebook (Adaptive_Fusion_POC.ipynb) cell 2.
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import pandas as pd
import torch

# ---- Paths (relative to FYP root) ----
PRODUCT_DIR          = Path(__file__).resolve().parents[1]
FYP_DIR              = PRODUCT_DIR.parent

# ---- Conda environment ----
CONDA_ENV_NAME = "fyp-gpu"
_conda_root = Path(os.environ["CONDA_EXE"]).parents[1] if "CONDA_EXE" in os.environ else None
_conda_env_python = _conda_root / "envs" / CONDA_ENV_NAME / "python.exe" if _conda_root else None
PYTHON_EXE = str(_conda_env_python) if _conda_env_python and _conda_env_python.exists() else sys.executable
NEWS_SENTIMENT_DIR   = FYP_DIR / "Processed_Data" / "news_sentiment_daily"
SOCIAL_SENTIMENT_DIR = FYP_DIR / "Processed_Data" / "tweets_sentiment_daily"
OPTIMIZER_DIR        = FYP_DIR / "portfolio_optimizer"
OUTPUT_DIR           = OPTIMIZER_DIR / "outputs"
MODEL_PATH           = OPTIMIZER_DIR / "fusion_network.pt"
SCRAPERS_DIR         = FYP_DIR / "scrapers"
RAW_NEWS_DIR         = FYP_DIR / "Raw_Data" / "gdelt_news_data"
RAW_TWEETS_DIR       = FYP_DIR / "Raw_Data" / "Tweets"
RUNNERS_DIR          = PRODUCT_DIR / "runners"
TRADE_LOG_PATH       = OUTPUT_DIR / "adaptive_fusion_trade_log.csv"
METRICS_PATH         = OUTPUT_DIR / "metrics_summary.csv"

# ---- General ----
INITIAL_NAV = 10_000.0

DATE_COL            = "date"
SENTIMENT_SCORE_COL = "avg_sentiment"
NEWS_CSV_SUFFIX     = "_news_sentiment_daily.csv"
SOCIAL_CSV_SUFFIX   = "_tweets_sentiment_daily.csv"

TICKERS = [
    "AAPL", "AMZN", "AVGO", "BRK-B", "GOOGL",
    "HD",   "JNJ",  "JPM",  "LLY",   "MA",
    "META", "MSFT", "NVDA", "ORCL",  "PG",
    "TSLA", "UNH",  "V",    "WMT",   "XOM",
]

SECTOR_MAP = {
    "AAPL": "Technology",  "AVGO": "Technology",  "MSFT": "Technology",
    "NVDA": "Technology",  "ORCL": "Technology",
    "AMZN": "ConsumerDiscretionary", "HD": "ConsumerDiscretionary", "TSLA": "ConsumerDiscretionary",
    "GOOGL": "Communication", "META": "Communication",
    "BRK-B": "Financials", "JPM": "Financials", "MA": "Financials", "V": "Financials",
    "JNJ": "Healthcare",  "LLY": "Healthcare",  "UNH": "Healthcare",
    "PG": "ConsumerStaples", "WMT": "ConsumerStaples",
    "XOM": "Energy",
}
SECTORS = sorted(set(SECTOR_MAP.values()))

# ---- Date ranges ----
BACKTEST_START = "2025-01-01"

def _detect_data_bounds(*dirs: Path):
    """Scan sentiment CSVs and return (earliest, latest) available dates."""
    earliest = pd.Timestamp.today()
    latest   = pd.Timestamp(BACKTEST_START)
    for d in dirs:
        if not d.exists():
            continue
        for csv in d.glob("*.csv"):
            try:
                dates = pd.read_csv(csv, usecols=["date"], parse_dates=["date"])["date"]
                if not dates.empty:
                    earliest = min(earliest, dates.min())
                    latest   = max(latest,   dates.max())
            except Exception:
                continue
    return str(earliest.normalize().date()), str(latest.normalize().date())

DATA_START, BACKTEST_END = _detect_data_bounds(
    FYP_DIR / "Processed_Data" / "news_sentiment_daily",
    FYP_DIR / "Processed_Data" / "tweets_sentiment_daily",
)
TRAIN_START    = DATA_START   # use all available history (20-stock universe needs maximum data)
TRAIN_END      = BACKTEST_START

# ---- Portfolio ----
TOP_N_STOCKS   = 5
MIN_WEIGHT     = 0.05
MAX_WEIGHT     = 0.40
REBALANCE_DAYS = 10

# ---- Transaction costs ----
SLIPPAGE_BPS      = 5
SEC_FEE_RATE      = 0.0000278
FINRA_TAF_PER_SH  = 0.000166

# ---- Stop-loss ----
# Default per-position stop-loss threshold. Liquidate a held position if its
# intraday Low falls below entry_price * (1 - STOP_LOSS_PCT). The 1% default is
# tighter than a typical 10% because the 10-day rebalance interval and low
# daily volatility of the large-cap universe would otherwise rarely trigger.
STOP_LOSS_PCT     = 0.01

# ---- Technical indicator parameters ----
RSI_PERIOD        = 14
MOMENTUM_PERIOD   = 20
REVERSAL_PERIOD   = 5
VOLUME_AVG_WINDOW = 20
IDIOVOL_WINDOW    = 20
HIGH_52W_WINDOW   = 252
VOLATILITY_WINDOW = 20

# ---- Neural network ----
N_FACTORS      = 8
CONTEXT_DIM    = 32
HIDDEN_DIM     = 16
LEARNING_RATE  = 5e-4
TRAIN_EPOCHS   = 100
BATCH_SIZE     = 64
DROPOUT_RATE   = 0.2
WEIGHT_DECAY   = 1e-4
SOFTMAX_TEMP   = 1.0
ENTROPY_LAMBDA = 0.0
FWD_HORIZON    = 10
RANDOM_SEED    = 42
STATIC_WEIGHTS = [1 / N_FACTORS] * N_FACTORS
RETRAIN_EVERY  = REBALANCE_DAYS  # retrain at every rebalance for walk-forward
ROLLING_WINDOW = 63           # ~3 months of trading days for rolling training window

# ---- Black-Litterman ----
BL_TAU         = 0.5
BL_DELTA       = 2.5
BL_OMEGA_SCALE = 1.0

# ---- Device ----
if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
USE_AMP      = DEVICE == "cuda"
USE_COMPILE  = False
PIN_MEMORY   = DEVICE == "cuda"
BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE   = 0.0

FACTOR_COLS = [
    "z_news_sentiment", "z_social_sentiment",
    "z_rsi", "z_momentum", "z_reversal", "z_abnormal_volume",
    "z_idiovol", "z_52w_high_ratio",
]
CONTEXT_COLS = (
    ["volatility_regime", "news_intensity", "social_intensity"]
    + [f"sector_{s}" for s in SECTORS]
)
