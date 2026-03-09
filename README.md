# Adaptive Multi-source Sentiment Fusion For Portfolio Optimisation

A context-conditioned attention network for adaptive multi-source sentiment fusion in Black-Litterman portfolio optimisation. The system learns to dynamically weight eight factor signals (news sentiment, social sentiment, and six technical indicators) based on volatility regime, data availability, and sector characteristics.

**Author:** Alasteir Ho
**Institution:** University of Greenwich (COMP1682 Final Year Project)
**Supervisor:** Dr Ali Jazdarreh

## Overview

The system performs the following pipeline:

1. **Data Collection** -- Scrapes financial news headlines from the GDELT API and tweets from X/Twitter via Selenium-based browser automation for 20 S&P 500 stocks across 7 sectors
2. **Sentiment Model** -- Fine-tunes a custom FIN-RoBERTa classifier from RoBERTa-base on four financial sentiment datasets (99.3% accuracy on Financial PhraseBank)
3. **Preprocessing** -- Cleans, deduplicates, and labels data using the custom model, then aggregates into daily sentiment scores per ticker
4. **Adaptive Fusion & Portfolio Optimisation** -- A PyTorch attention network produces context-dependent factor weights over 8 signals, which are integrated into a Black-Litterman framework and optimised via Sharpe ratio maximisation
5. **Walk-Forward Backtest** -- Evaluates four progressive strategies over ~252 trading days (Jan 2024 to Jan 2025) with realistic transaction costs (SEC/FINRA fees + 5 bps slippage)

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

- **Language:** Python 3.12
- **Deep Learning / NLP:** PyTorch, Transformers (RoBERTa), scikit-learn
- **Portfolio Optimisation:** SciPy (SLSQP), Black-Litterman framework
- **Data Handling:** pandas, NumPy, yfinance, exchange-calendars
- **Web Scraping:** Selenium, undetected-chromedriver, gdeltdoc (GDELT API)
- **Language Detection:** lingua-language-detector
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Demo App:** Streamlit

## Project Structure

```
Adaptive-Fusion-For-Stock-Portfolio-Optimization/
+-- scrapers/
|   +-- GDELTscraper.py                      # GDELT news API scraper
|   +-- twitter_scraper.py                    # X/Twitter scraper (Selenium)
|
+-- preprocessing/
|   +-- news_eda.ipynb                        # News exploratory data analysis
|   +-- news_preprocessing_labelling.ipynb    # News cleaning, labelling & aggregation
|   +-- tweets_eda.ipynb                      # Tweets exploratory data analysis
|   +-- tweets_preprocessing_labelling.ipynb  # Tweets cleaning, labelling & aggregation
|
+-- Sentiment_Model/
|   +-- model_evaluation.ipynb                # FIN-RoBERTa vs FinBERT comparison
|   +-- RoBERTa-Train/
|       +-- train.ipynb                       # Custom FIN-RoBERTa fine-tuning
|       +-- semeval-2017-task-5-subtask-2/    # SemEval training data
|
+-- portfolio_optimizer/
|   +-- Adaptive_Fusion_POC.ipynb             # Main notebook: training, backtest, analysis
|   +-- fusion_network.pt                     # Trained PyTorch fusion network weights
|   +-- outputs/                              # Backtest results & visualisations
|       +-- 1_nav_comparison.png
|       +-- 2_drawdown_comparison.png
|       +-- 3_weight_evolution.png
|       +-- 4_attention_weights.png
|       +-- 5_factor_attribution.png
|       +-- 6_attention_per_sector_ticker.png
|       +-- 7_attention_by_vol_regime.png
|       +-- 8_oos_ic_hit_rate.png
|       +-- 9_turnover_analysis.png
|       +-- 10_bootstrap_ci.png
|       +-- adaptive_fusion_trade_log.csv
|       +-- bootstrap_ci.csv
|       +-- metrics_summary.csv
|
+-- Product/                                  # Streamlit demo application
|   +-- main.py
|   +-- core/
|   |   +-- paths.py
|   +-- pages/
|   |   +-- 0_About.py
|   |   +-- 1_Data_Collection.py
|   |   +-- 2_Portfolio_Simulation.py
|   +-- runners/
|       +-- gdelt_runner.py
|       +-- twitter_runner.py
|
+-- requirements.txt
+-- .gitignore
```

**Not tracked in git:** `Raw_Data/`, `Processed_Data/`, `Report/`, `.env`, `scrapers/chrome_profile/`

## Installation

### Prerequisites

- Python 3.12
- Conda (recommended) or venv
- Chrome browser (for Twitter scraping)
- NVIDIA GPU with CUDA support (optional; accelerates sentiment inference and NN training)

### Setup

1. **Create conda environment:**
   ```bash
   conda create -n <env_name> python=3.12
   conda activate <env_name>
   ```

2. **Install PyTorch with CUDA support (recommended):**
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
   This installs the CUDA 13.0 build of PyTorch for GPU acceleration. Skip this step if running you have an AMD GPU or CPU-only .

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

   Create a `.env` file with your Twitter credentials:
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

Outputs to `Processed_Data/news_sentiment_daily/` and `Processed_Data/tweets_sentiment_daily/`.

### Step 4: Portfolio Optimisation and Backtest
Run the main notebook in `portfolio_optimizer/`:
- `Adaptive_Fusion_POC.ipynb` -- Trains the attention network, runs the walk-forward backtest for all strategies, and generates all results and visualisations

### Step 5: Demo App (optional)
```bash
cd Product
streamlit run main.py
```

## Models

### Custom FIN-RoBERTa Sentiment Classifier

A RoBERTa-base model (125M parameters) fine-tuned for three-class financial sentiment classification:

- **Hugging Face:** [alasteirho/FIN-RoBERTa-Custom](https://huggingface.co/alasteirho/FIN-RoBERTa-Custom)
- **Labels:** negative, neutral, positive
- **Benchmark:** 99.3% accuracy on Financial PhraseBank (sentences_allagree subset, 2,264 samples)

#### Training Datasets

| Dataset | Description |
|---------|-------------|
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | Financial news sentences (sentences_allagree, 80:20 split) |
| [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Twitter financial news sentiment |
| [FiQA 2018](https://huggingface.co/datasets/pauri32/fiqa-2018) | Financial opinion mining and QA |
| [SemEval-2017 Task 5 Subtask 2](https://alt.qcri.org/semeval2017/task5/) | Fine-grained sentiment on financial news headlines |

## Architecture

### Attention Network

The context-conditioned attention network takes 10 context inputs (volatility regime tercile, news/social intensity, 7 sector one-hots) and produces 8 factor weights via softmax. The weighted sum of z-scored factors yields a composite alpha score per ticker.

```
Context (10-dim) --> Linear(32) --> ReLU --> Dropout(0.2)
                 --> Linear(16) --> ReLU
                 --> Linear(8)  --> Softmax --> Factor Weights

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
| Idiosyncratic volatility | Close prices (20-day rolling) |
| 52-week high ratio | Close prices |

### Portfolio Construction

Composite alpha scores are integrated into a **Black-Litterman** framework (tau=0.5, delta=2.5) and optimised via **Sharpe ratio maximisation** (SLSQP) with weight bounds [5%, 20%] and top-5 stock selection per rebalance.

## Key Design Decisions

- **No look-ahead bias:** Signals use T-1 data; trades execute at T's open price. Sentiment is shifted +1 day before feature construction.
- **Walk-forward retraining:** The WF variant retrains the attention network every 60 trading days using an expanding window with warm-starting.
- **Realistic costs:** SEC fee (0.278 bps on sells), FINRA TAF ($0.000166/share), and 5 bps one-way slippage on all trades.
- **Statistical robustness:** 2,000 stationary bootstrap resamples with geometric block length of 10 days for confidence intervals.

## License

This project is part of the COMP1682 Final Year Project at the University of Greenwich.
