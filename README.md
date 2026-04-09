# Adaptive Multi-source Sentiment Fusion For Portfolio Optimisation

A context-conditioned attention network for adaptive multi-source sentiment fusion in Black-Litterman portfolio optimisation. The system dynamically weights eight factor signals (news sentiment, social sentiment, and six technical indicators) based on volatility regime, data availability, and sector characteristics.

**Author:** Alasteir Ho Zhen Wei
**Student ID:** 001295065
**Institution:** University of Greenwich (COMP1682 Final Year Project)
**Supervisor:** Dr Ali Jazdarreh
**Date:** April 2026

## Key Results

| Strategy | Sharpe | Ann. Return | Max Drawdown | Total Return |
|----------|--------|-------------|-------------|-------------|
| **Adaptive (WF)** | **1.51** | 29.29% | -10.70% | 28.76% |
| **Adaptive (Fixed)** | **1.51** | 29.29% | -10.70% | 28.76% |
| Equal-Weight | 1.12 | 21.76% | -18.44% | 21.47% |
| Price-Only | 1.02 | 20.70% | -19.56% | 20.34% |
| SPY Buy-and-Hold | 0.75 | 14.84% | -18.76% | 14.54% |
| Static-Fusion | 0.47 | 9.99% | -20.02% | 9.83% |

Both Adaptive variants outperformed all benchmarks (Sharpe 1.51 vs SPY 0.75, Equal-Weight 1.12). The Walk-Forward and Fixed variants produced identical results, demonstrating that the pre-trained attention weights generalised robustly beyond the training period.

## Overview

The system performs the following pipeline:

1. **Data Collection** -- Scrapes financial news headlines from the GDELT API and tweets from X/Twitter via Selenium-based browser automation for 20 S&P 500 stocks across 7 sectors
2. **Sentiment Model** -- Fine-tunes a custom FIN-RoBERTa classifier from RoBERTa-base on four financial sentiment datasets (99.3% accuracy on Financial PhraseBank)
3. **Preprocessing** -- Cleans, deduplicates, and labels data using the custom model, then aggregates into daily sentiment scores per ticker
4. **Adaptive Fusion & Portfolio Optimisation** -- A PyTorch attention network produces context-dependent factor weights over 8 signals, integrated into a Black-Litterman framework and optimised via Sharpe ratio maximisation
5. **Walk-Forward Backtest** -- Evaluates four progressive strategies over ~252 trading days (Jan 2025 to Jan 2026) with realistic transaction costs (SEC/FINRA fees + 5 bps slippage)
6. **Streamlit Web Application** -- Three-page interactive demo with data collection, model training, and portfolio simulation with animated playback

## Supported Stocks

20 large-cap S&P 500 stocks across 7 GICS sectors:

| Sector | Tickers |
|--------|---------|
| Technology | NVDA, AAPL, MSFT, AVGO, ORCL |
| Communication Services | GOOGL, META |
| Consumer Discretionary | AMZN, TSLA, HD |
| Financial Services | BRK-B, JPM, V, MA |
| Healthcare | JNJ, LLY, UNH |
| Consumer Staples | WMT, PG |
| Energy | XOM |

## Technology Stack

- **Language:** Python 3.13
- **Deep Learning / NLP:** PyTorch, Transformers (RoBERTa), scikit-learn
- **Portfolio Optimisation:** SciPy (SLSQP), Black-Litterman framework
- **Data Handling:** pandas, NumPy, yfinance, exchange-calendars
- **Web Scraping:** Selenium, undetected-chromedriver, gdeltdoc (GDELT API)
- **Language Detection:** lingua-language-detector
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Demo App:** Streamlit
- **Testing:** pytest (50 unit tests)

## Project Structure

```
Adaptive-Fusion-For-Stock-Portfolio-Optimization/
|-- scrapers/
|   |-- GDELTscraper.py              # GDELT news headline collection
|   +-- twitter_scraper.py           # Selenium-based tweet collection
|-- preprocessing/
|   |-- news_eda.ipynb                # News exploratory data analysis
|   |-- news_preprocessing_labelling.ipynb
|   |-- tweets_eda.ipynb              # Tweet exploratory data analysis
|   +-- tweets_preprocessing_labelling.ipynb
|-- Sentiment_Model/
|   |-- RoBERTa-Train/                # Fine-tuning scripts and training data
|   +-- model_evaluation.ipynb        # FIN-RoBERTa benchmark evaluation
|-- portfolio_optimizer/
|   |-- Adaptive_Fusion_POC.ipynb     # Main research notebook
|   |-- fusion_network.pt             # Pre-trained model weights (Fixed)
|   |-- fusion_network_dynamic.pt     # Walk-forward model weights
|   +-- outputs/                      # Figures, CSV results, trade logs
|-- Product/
|   |-- backend/                      # Python package (9 modules)
|   |   |-- config.py                 #   Centralised hyperparameters
|   |   |-- model.py                  #   Attention network and training
|   |   |-- backtest.py               #   Walk-forward backtest engine
|   |   |-- optimizer.py              #   Black-Litterman and Sharpe MVO
|   |   |-- sentiment.py              #   FIN-RoBERTa inference wrapper
|   |   |-- features.py               #   Technical factor engineering
|   |   +-- data.py                   #   Data loading utilities
|   |-- frontend/                     # Streamlit web application
|   |   |-- about.py                  #   System status dashboard
|   |   |-- data_collection.py        #   Data acquisition interface
|   |   +-- portfolio_simulation.py   #   Portfolio simulation page
|   |-- runners/                      # Subprocess managers
|   |   |-- gdelt_runner.py           #   News scraper execution
|   |   |-- twitter_runner.py         #   Tweet scraper execution
|   |   +-- sentiment_runner.py       #   Sentiment scoring execution
|   |-- tests/                        # Unit test suite (pytest)
|   |   |-- test_features.py          #   20 tests for technical indicators
|   |   |-- test_model.py             #   13 tests for attention network
|   |   +-- test_optimizer.py         #   17 tests for BL and MVO
|   +-- main.py                       # Streamlit application entry point
|-- Raw_Data/                         # Unprocessed scraper outputs
|-- Processed_Data/                   # Daily sentiment CSV files
|-- literature/                       # Source PDFs for referenced papers
|-- Report/                           # LaTeX dissertation
|-- requirements.txt
+-- README.md
```

## Installation

### Prerequisites

- Python 3.13+
- Conda (recommended) or venv
- Chrome browser (for Twitter scraping)
- NVIDIA GPU with CUDA support (recommended for sentiment inference and NN training)

### Setup

1. **Create conda environment:**
   ```bash
   conda create -n fyp-gpu python=3.13
   conda activate fyp-gpu
   ```

2. **Install PyTorch with CUDA support (recommended):**
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   Required to download model weights. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

5. **Configure environment variables (for scraping only):**
   Create a `.env` file:
   ```
   TWITTER_USERNAME=your_username
   TWITTER_PASSWORD=your_password
   ```

## Usage

### Step 1: Collect News Data
```bash
python scrapers/GDELTscraper.py
```
Collects news headlines via the GDELT Document API. Outputs to `Raw_Data/gdelt_news_data/`.

### Step 2: Collect Twitter Data
```bash
python scrapers/twitter_scraper.py
```
Scrapes cashtag-mentioning tweets via Selenium browser automation. Outputs to `Raw_Data/Tweets/`.

### Step 3: Preprocess and Label Data
Run the Jupyter notebooks in `preprocessing/`:
- `news_preprocessing_labelling.ipynb` -- Clean, label with FIN-RoBERTa, aggregate to daily scores
- `tweets_preprocessing_labelling.ipynb` -- Multi-stage filtering, label, aggregate to daily scores

### Step 4: Portfolio Optimisation and Backtest
Run the main notebook:
- `portfolio_optimizer/Adaptive_Fusion_POC.ipynb` -- Trains the attention network, runs walk-forward backtest for all strategies, generates results and visualisations

### Step 5: Run Tests
```bash
cd Product
pytest tests/ -v
```

### Step 6: Streamlit GUI (optional)
```bash
cd Product
streamlit run main.py
```

## Models

### Custom FIN-RoBERTa Sentiment Classifier

A RoBERTa-base model (125M parameters) fine-tuned for three-class financial sentiment classification:

- **Hugging Face:** [alasteirho/FIN-RoBERTa-Custom](https://huggingface.co/alasteirho/FIN-RoBERTa-Custom)
- **Labels:** negative, neutral, positive
- **Training data:** Financial PhraseBank, Twitter Financial News Sentiment, FiQA 2018, SemEval 2017 Task 5
- **Benchmark:** 99.3% accuracy on Financial PhraseBank (sentences_allagree, 2,264 samples)

## Architecture

### Attention Network

The context-conditioned attention network takes 10 context inputs (volatility regime tercile, news/social intensity, 7 sector one-hots) and produces 8 factor weights via softmax:

```
Context (10-dim) --> Fully Connected (32) --> ReLU --> Dropout (0.2)
                 --> Fully Connected (16) --> ReLU
                 --> Fully Connected (8)  --> Softmax --> Factor Weights

8 Z-scored Factors x Factor Weights --> Composite Alpha Score
```

### Eight Factor Signals

| Factor | Source |
|--------|--------|
| News sentiment | FIN-RoBERTa on GDELT headlines |
| Social sentiment | FIN-RoBERTa on X/Twitter posts |
| RSI (14-day) | Close prices |
| Momentum (20-day) | Close prices |
| 5-day reversal | Close prices |
| Abnormal volume | Volume data |
| Idiosyncratic volatility | Close prices (20-day rolling std) |
| 52-week high ratio | Close prices |

### Portfolio Construction

Composite alpha scores are integrated into a **Black-Litterman** framework (tau=0.5, delta=2.5) and optimised via **Sharpe ratio maximisation** (SLSQP) with weight bounds [5%, 40%] and top-5 stock selection per rebalance.

## Key Design Decisions

- **No look-ahead bias:** Signals use T-1 data; trades execute at T open price. Day 0 of the backtest is skipped to enforce strict T-1 discipline.
- **Walk-forward retraining:** The WF variant retrains the attention network at every rebalance (every 10 trading days) using a 3-month (63 trading day) rolling window with warm-starting.
- **No sentiment forward-fill:** Missing sentiment days default to neutral (0) rather than carrying stale values forward; the attention network learns to rely on technical factors when sentiment coverage is sparse.
- **Realistic costs:** SEC fee (0.278 bps on sells), FINRA TAF ($0.000166/share, capped at $8.30), and 5 bps one-way slippage on all trades.
- **Statistical robustness:** 2,000 stationary bootstrap resamples with geometric block length of 10 days for confidence intervals.

## Limitations

- **Historical data only:** The system operated entirely on historical Yahoo Finance data. Live deployment was not feasible as paper-trading platforms (such as Alpaca) do not support backtesting on historical data with custom models, and a data-source mismatch exists between Yahoo Finance prices and live execution feeds.
- **Open price execution is unrealistic:** In real markets, the official open price is determined retrospectively by the opening auction; actual fill prices differ from the published open.
- **Limited sentiment contribution:** Sentiment signals received only 12.1% of the attention budget; the majority of alpha came from adaptive weighting of technical factors. Sentiment may be more effective for smaller, less-covered stocks.
- **Single backtest period:** The 252-day evaluation window (Jan 2025 to Jan 2026) captures limited market conditions. Bootstrap confidence intervals overlap across strategies.
- **Narrow universe:** 20 large-cap S&P 500 stocks only. Performance may differ for small-cap, international, or less liquid equities.

## Disclaimer

This project is an academic prototype developed for the COMP1682 Final Year Project at the University of Greenwich. It does not constitute financial advice. Past backtest performance does not guarantee future returns. The system is not suitable for live trading.

## License

This project is part of the COMP1682 Final Year Project at the University of Greenwich.
