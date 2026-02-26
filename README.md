# Adaptive Fusion for Multi-Source Sentiment Analysis

An adaptive fusion approach to financial sentiment analysis that integrates multiple data sources (news articles and social media) with a custom-trained NLP model to predict stock price direction and optimize an investment portfolio.

**Author:** Alasteir Ho
**Institution:** University of Greenwich (Final Year Project)

## Overview

This system performs the following workflow:

1. **Data Collection** - Scrapes news articles from the GDELT API and tweets from Twitter/X for major S&P 500 stocks
2. **Preprocessing & Labelling** - Cleans, deduplicates, and labels data using a custom FIN-RoBERTa sentiment model
3. **Sentiment Aggregation** - Aggregates labelled data into daily sentiment scores per ticker
4. **Portfolio Optimization** - Implements an adaptive fusion strategy that combines news and tweet sentiment signals via a learned attention mechanism, with full backtesting

## Supported Stocks

20 major S&P 500 stocks across 7 sectors:

| Sector | Tickers |
|--------|---------|
| Technology | NVDA, AAPL, MSFT, AVGO, ORCL |
| Communication Services | GOOGL, META |
| Consumer Discretionary | AMZN, TSLA, HD |
| Financial Services | BRK-B, JPM, V, MA |
| Healthcare | JNJ, LLY, UNH |
| Consumer Staples | WMT, PG |
| Energy | XOM |

**Data period:** October 2, 2023 - October 9, 2025 (~502 trading days)

## Technology Stack

- **Languages:** Python 3.13+
- **ML/NLP:** PyTorch, Transformers (FIN-RoBERTa), scikit-learn
- **Portfolio Optimisation:** cvxpy
- **Data Handling:** Pandas, NumPy, yfinance
- **Web Scraping:** Selenium, undetected-chromedriver, GDELT API
- **Visualisation:** Matplotlib, Jupyter Notebook

## Project Structure

```
FYP/
├── scrapers/                                # Data collection scripts
│   ├── GDELTscraper.py                      # GDELT news API scraper
│   └── twitter_scraper.py                   # Twitter/X scraper (Selenium)
│
├── preprocessing/                           # Data preprocessing, EDA & labelling
│   ├── news_eda.ipynb                       # News exploratory data analysis
│   ├── news_preprocessing_labelling.ipynb   # News cleaning, labelling & aggregation
│   ├── tweets_eda.ipynb                     # Tweets exploratory data analysis
│   └── tweets_preprocessing_labelling.ipynb # Tweets cleaning, labelling & aggregation
│   
│
├── Sentiment_Model/                         # Sentiment model training & evaluation
│   ├── model_evaluation.ipynb               # FIN-RoBERTa vs FinBERT comparison
│   └── RoBERTa-Train/
│       ├── train.ipynb                      # Custom FIN-RoBERTa fine-tuning
│       └── semeval-2017-task-5-subtask-2/   # SemEval training data
│
│
├── portfolio_optimizer/                     # Adaptive Fusion portfolio strategy
│   ├── Adaptive_Fusion_POC.ipynb            # Main adaptive fusion notebook (Proof Of Concept)
│   ├── fusion_network.pt                    # Trained PyTorch fusion network weights 
│   └── outputs/                             # Backtest results & plots
│       ├── 1_nav_comparison.png
│       ├── 2_drawdown_comparison.png
│       ├── 3_weight_evolution.png
│       ├── 4_attention_weights.png
│       ├── 5_factor_attribution.png
│       ├── adaptive_fusion_trade_log.csv
│       └── metrics_summary.csv
│
├── Raw_Data/                                # Raw scraped data
│   ├── gdelt_news_data/                     # Raw GDELT news per ticker
│   │   └── <TICKER>_news.csv
│   └── Tweets/                              # Raw scraped tweets per ticker
│       └── tweets_<TICKER>.csv
│
├── Processed_Data/                          # Aggregated daily sentiment data
│   ├── news_sentiment_daily/                # Daily news sentiment scores per ticker
│   │   └── <TICKER>_news_sentiment_daily.csv
│   └── tweets_sentiment_daily/              # Daily tweet sentiment scores per ticker
│       └── <TICKER>_tweets_sentiment_daily.csv
│
├── requirements.txt                         # Python dependencies
├── .env                                     # Environment variables
└── .gitignore
```

### Processed Data Schema

**`news_sentiment_daily/`** columns: `date, open_price, close_price, avg_sentiment`

**`tweets_sentiment_daily/`** columns: `date, avg_sentiment`

Ticker from filename: `{TICKER}_{source}_sentiment_daily.csv`

## Installation

### Prerequisites

- Python 3.13+
- Anaconda or Miniconda (recommended)
- Chrome browser (for Twitter scraping)
- Nvidia GPU with 8 GB+ VRAM and Tensor Core (optional, for faster training and inference)

### Setup

1. **Create conda environment (recommended):**
   ```bash
   conda create -n fyp python=3.13
   conda activate fyp
   ```

   Alternatively, use venv:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate with Hugging Face:**
   ```bash
   huggingface-cli login
   ```
   Required to download model weights. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

4. **Configure environment variables:**

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
Outputs to `Raw_Data/gdelt_news_data/`.

### Step 2: Collect Twitter Data
```bash
python scrapers/twitter_scraper.py
```
Outputs to `Raw_Data/Tweets/`.

### Step 3: Preprocess & Label Data
Run the Jupyter notebooks in `preprocessing/`:
- `news_preprocessing_labelling.ipynb` - Clean and label news data, aggregate to daily scores
- `tweets_preprocessing_labelling.ipynb` - Clean and label tweet data, aggregate to daily scores

### Step 4: Portfolio Optimization
Run the notebook in `portfolio_optimizer/`:
- `Adaptive_Fusion_POC.ipynb` - Adaptive fusion strategy combining news and tweet sentiment via learned attention, with full backtesting

Early strategy iterations (news-aware, price-aware) are in `Strategy/Initial_S1toS3.ipynb`.

## Models

### Custom FIN-RoBERTa Model

A RoBERTa-based model fine-tuned for financial sentiment analysis on domain-specific corpora:

- **Model:** [alasteirho/FIN-RoBERTa-Custom](https://huggingface.co/alasteirho/FIN-RoBERTa-Custom)
- **Labels:** negative, neutral, positive

#### Training Datasets

| Dataset | Description |
|---------|-------------|
| [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | Financial news sentences with sentiment labels (sentences_allagree, 80:20 split) |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Twitter financial news sentiment dataset |
| [pauri32/fiqa-2018](https://huggingface.co/datasets/pauri32/fiqa-2018) | Financial Opinion Mining and Question Answering dataset |
| [SemEval-2017 Task 5 Subtask 2](https://alt.qcri.org/semeval2017/task5/) | Fine-grained sentiment analysis on financial news headlines |

#### Evaluation Dataset

Evaluated on the [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset (`sentences_allagree` subset) - 2,264 sentences with 100% annotator agreement.

#### Training Statistics

| Metric | Value |
|--------|-------|
| GPU | NVIDIA GeForce RTX 5080 |
| GPU Memory | 15.92 GB |
| Training time | 166.4 s (2.77 min) |
| VRAM used for training | 5.26 GB |
| VRAM used for inference | 4.04 GB |
| Evaluation time | 1.80 s |

### Model Evaluation

Model comparison notebooks are in `Sentiment_Model/`:
- `model_evaluation.ipynb` - FIN-RoBERTa vs FinBERT evaluation on Financial PhraseBank
- `RoBERTa-Train/train.ipynb` - Fine-tuning notebook

## Key Design Decisions

- **Data Leakage Prevention:** Only t-1 sentiment data is used to generate portfolio weights for day t
- **Multiple Data Sources:** Financial news (GDELT) and social media (Twitter/X)
- **Adaptive Fusion:** A PyTorch attention network learns to weight news vs tweet sentiment dynamically
- **Backtesting:** Daily mark-to-market with benchmark comparison and drawdown analysis

## License

This project is part of my Final Year Project at the University of Greenwich.
