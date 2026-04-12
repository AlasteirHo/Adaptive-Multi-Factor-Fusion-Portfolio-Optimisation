# Adaptive Multi-source Sentiment Fusion For Portfolio Optimisation

A context-conditioned attention network for adaptive multi-source sentiment fusion in Black-Litterman portfolio optimisation. The system dynamically weights eight factor signals (news sentiment, social sentiment, and six technical indicators) based on volatility regime, data availability, and sector characteristics.

**Author:** Alasteir Ho Zhen Wei
**Student ID:** 001295065
**Institution:** University of Greenwich (COMP1682 Final Year Project)
**Supervisor:** Dr Ali Jazdarreh
**Date:** April 2026

## Key Results

### Base Backtest (Jan 2025 to Jan 2026, ~252 trading days)

| Strategy | Sharpe | Ann. Return | Ann. Vol | Max Drawdown | Calmar | Total Return |
|----------|--------|-------------|----------|-------------|--------|-------------|
| **Adaptive (WF)** | **1.51** | 29.29% | 19.45% | -10.70% | 2.74 | 28.76% |
| **Adaptive (Fixed)** | **1.51** | 29.29% | 19.45% | -10.70% | 2.74 | 28.76% |
| Equal-Weight | 1.12 | 21.76% | 19.46% | -18.44% | 1.18 | 21.47% |
| Price-Only | 1.02 | 20.70% | 20.38% | -19.56% | 1.06 | 20.34% |
| SPY Buy-and-Hold | 0.75 | 14.84% | 19.84% | -18.76% | 0.79 | 14.54% |
| Static-Fusion | 0.47 | 9.99% | 21.21% | -20.02% | 0.50 | 9.83% |

### Stop-Loss Ablation (Dec 2024 to Dec 2025)

| Strategy | Sharpe | Ann. Return | Ann. Vol | Max Drawdown | Calmar | Total Return |
|----------|--------|-------------|----------|-------------|--------|-------------|
| **Adaptive (Fixed+Stop)** | **3.35** | 31.83% | 9.51% | **-5.30%** | **6.01** | 31.26% |
| Adaptive (Fixed) | 1.51 | 29.31% | 19.45% | -10.70% | 2.74 | 28.79% |

Adding a 1% per-position stop-loss (based on Han, Zhou & Zhu, 2016) to the Adaptive strategy more than doubled the Sharpe ratio (1.51 to 3.35) by cutting annualised volatility in half (19.45% to 9.51%) and halving the maximum drawdown (-10.70% to -5.30%), while slightly improving total return. 86 stop-outs were triggered during the backtest period.

Both Adaptive variants (Walk-Forward and Fixed) produced identical base results, demonstrating that the pre-trained attention weights generalised robustly beyond the training period.

### Portfolio NAV Performance

<p align="center">
  <img src="diagrams/1_nav_comparison.png" alt="Portfolio NAV comparison across all six strategies" width="800"/>
</p>

### Drawdown Comparison

<p align="center">
  <img src="diagrams/2_drawdown_comparison.png" alt="Drawdown profiles for all strategies" width="800"/>
</p>

## Overview

The system performs the following pipeline:

1. **Data Collection** -- Scrapes financial news headlines from the GDELT API and tweets from X/Twitter via Selenium-based browser automation for 20 S&P 500 stocks across 7 sectors
2. **Sentiment Model** -- Fine-tunes a custom FIN-RoBERTa classifier from RoBERTa-base on four financial sentiment datasets (99.3% accuracy on Financial PhraseBank)
3. **Preprocessing** -- Cleans, deduplicates, and labels data using the custom model, then aggregates into daily sentiment scores per ticker
4. **Adaptive Fusion & Portfolio Optimisation** -- A PyTorch attention network produces context-dependent factor weights over 8 signals, integrated into a Black-Litterman framework and optimised via Sharpe ratio maximisation
5. **Walk-Forward Backtest** -- Evaluates strategies over ~252 trading days with realistic transaction costs (SEC/FINRA fees + 5 bps slippage)
6. **Stop-Loss Ablation** -- Tests the impact of a 1% per-position stop-loss rule on the best-performing adaptive strategy
7. **Streamlit Web Application** -- Three-page interactive demo with data collection, model training, and portfolio simulation with animated playback

## Architecture

### Portfolio Optimisation Pipeline

<p align="center">
  <img src="diagrams/Portfolio_Pipeline.png" alt="Portfolio optimisation pipeline" width="800"/>
</p>

Close prices feed into technical indicator computation and a shrunk 60-day covariance matrix. News and tweet sentiment scores join the technical factors to form 8 Z-scored signals. The attention network, conditioned on volatility regime and sector context, produces adaptive factor weights that fuse the signals into a composite alpha score. These alpha scores become views in the Black-Litterman framework, which combines market equilibrium priors with the model's views to produce posterior expected returns. A Sharpe ratio maximiser (SLSQP) then determines optimal portfolio weights.

### Adaptive Fusion Network

<p align="center">
  <img src="diagrams/AdaptiveFusionArchitecture.png" alt="Attention network architecture" width="500"/>
</p>

The context-conditioned attention network takes 10 context inputs (volatility regime tercile, news/social intensity, 7 sector one-hots) and produces 8 factor weights via softmax:

```
Context (10-dim) --> FC(32) --> ReLU --> Dropout(0.2)
                 --> FC(16) --> ReLU
                 --> FC(8)  --> Softmax --> Factor Weights

8 Z-scored Factors x Factor Weights --> Composite Alpha Score
```

The network is trained by maximising the information coefficient (IC) between predicted alpha scores and next-day cross-sectional returns.

### Strategy Hierarchy

<p align="center">
  <img src="diagrams/Strategies.png" alt="Strategy hierarchy from S1 (SPY) to S5 (Adaptive NN Fusion)" width="700"/>
</p>

Five strategies of increasing complexity are compared to isolate the contribution of each component: market benchmark (S1), equal-weight diversification (S2), technical-only factors (S3), static sentiment fusion (S4), and adaptive NN-based fusion (S5).

### Factor Attribution

<p align="center">
  <img src="diagrams/5_factor_attribution.png" alt="Average attention weights assigned to each factor" width="700"/>
</p>

The attention network allocates the majority of weight to risk-based technical factors: idiosyncratic volatility (32.0%), 5-day reversal (20.8%), and abnormal volume (15.5%) account for 68.3% of the attention budget. News sentiment (8.7%) and social sentiment (4.6%) receive a combined 13.3%, indicating that sentiment serves as a complementary signal rather than the primary alpha driver for this large-cap universe.

### Volatility Regime Adaptation

<p align="center">
  <img src="diagrams/7_attention_by_vol_regime.png" alt="Attention weights by volatility regime" width="800"/>
</p>

The model adapts its factor weighting based on the prevailing volatility regime. In high-volatility periods, the network increases weight on idiosyncratic volatility and RSI (mean-reversion signals), while in low-volatility periods it allocates relatively more to sentiment and momentum signals. This regime-conditional behaviour is learned end-to-end without explicit rules.

### Bootstrap Confidence Intervals

<p align="center">
  <img src="diagrams/10_bootstrap_ci.png" alt="Bootstrap 95% confidence intervals for Sharpe, CAGR, and max drawdown" width="800"/>
</p>

Statistical robustness is assessed via 2,000 stationary bootstrap resamples with geometric block length of 10 days. While the Adaptive strategies show higher point estimates across all metrics, the 95% confidence intervals overlap across strategies, reflecting the inherent uncertainty of a single 252-day evaluation window.

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

## Eight Factor Signals

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

## Portfolio Construction

Composite alpha scores are integrated into a **Black-Litterman** framework (tau=0.5, delta=2.5) and optimised via **Sharpe ratio maximisation** (SLSQP) with weight bounds [5%, 40%] and top-5 stock selection per rebalance (every 10 trading days).

**Stop-loss extension:** A 1% per-position stop-loss monitors intraday lows against entry prices. When triggered, the position is liquidated at the stop price with one-way slippage and regulatory fees; proceeds are held as cash (earning Fed Funds Effective minus 25 bps) until the next scheduled rebalance.

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
|   |-- Adaptive_Fusion_POC.ipynb     # Main research notebook (base backtest)
|   |-- Adaptive_Fusion_POC_StopLoss.ipynb  # Stop-loss ablation notebook
|   |-- fusion_network.pt             # Pre-trained model weights (Fixed)
|   |-- fusion_network_dynamic.pt     # Walk-forward model weights
|   |-- outputs/                      # Figures and CSV results (base)
|   +-- outputs2/                     # Figures and CSV results (stop-loss)
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
|-- Report/                           # LaTeX dissertation and figures
|-- literature/                       # Source PDFs for referenced papers
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
Run the main notebooks:
- `portfolio_optimizer/Adaptive_Fusion_POC.ipynb` -- Trains the attention network, runs walk-forward backtest for all strategies, generates results and visualisations
- `portfolio_optimizer/Adaptive_Fusion_POC_StopLoss.ipynb` -- Stop-loss ablation study on the best-performing adaptive strategy

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

## Key Design Decisions

- **No look-ahead bias:** Signals use T-1 data; trades execute at T open price. Day 0 of the backtest is skipped to enforce strict T-1 discipline.
- **Walk-forward retraining:** The WF variant retrains the attention network at every rebalance (every 10 trading days) using a 3-month (63 trading day) rolling window with warm-starting.
- **No sentiment forward-fill:** Missing sentiment days default to neutral (0) rather than carrying stale values forward; the attention network learns to rely on technical factors when sentiment coverage is sparse.
- **Realistic costs:** SEC fee (0.278 bps on sells), FINRA TAF ($0.000166/share, capped at $8.30), and 5 bps one-way slippage on all trades.
- **Stop-loss rule:** 1% per-position stop based on intraday low vs. entry price, informed by Han, Zhou & Zhu (2016) and Kaminski & Lo (2014).
- **Statistical robustness:** 2,000 stationary bootstrap resamples with geometric block length of 10 days for confidence intervals.

## Limitations

- **Historical data only:** The system operated entirely on historical Yahoo Finance data. Live deployment was not feasible as paper-trading platforms (such as Alpaca) do not support backtesting on historical data with custom models, and a data-source mismatch exists between Yahoo Finance prices and live execution feeds.
- **Open price execution is unrealistic:** In real markets, the official open price is determined retrospectively by the opening auction; actual fill prices differ from the published open.
- **Limited sentiment contribution:** Sentiment signals received only 13.3% of the attention budget; the majority of alpha came from adaptive weighting of technical factors. Sentiment may be more effective for smaller, less-covered stocks.
- **Single backtest period:** The ~252-day evaluation window captures limited market conditions. Bootstrap confidence intervals overlap across strategies.
- **Narrow universe:** 20 large-cap S&P 500 stocks only. Performance may differ for small-cap, international, or less liquid equities.
- **Stop-loss assumes intraday data availability:** The stop-loss rule checks intraday lows, which are available historically from Yahoo Finance but would require real-time data feeds in a live setting.

## Disclaimer

This project is an academic prototype developed for the COMP1682 Final Year Project at the University of Greenwich. It does not constitute financial advice. Past backtest performance does not guarantee future returns. The system is not suitable for live trading.

## License

This project is part of the COMP1682 Final Year Project at the University of Greenwich.
