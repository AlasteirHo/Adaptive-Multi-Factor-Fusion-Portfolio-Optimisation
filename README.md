# Adaptive Fusion for Multi-Source Stock Portfolio Optimization

An adaptive fusion approach to financial sentiment analysis that integrates multiple data sources (news articles and social media) and technical indicators to predict stock price direction and optimize an investment portfolio.

**Author:** Alasteir Ho
**Institution:** University of Greenwich (Final Year Project)

## Overview

This system performs the following workflow:

1. **Data Collection** - Scrapes news articles from GDELT API and Twitter/X posts for major stocks
2. **Preprocessing & Labelling** - Cleans and labels data for sentiment analysis
3. **Portfolio Optimization** - Implements news-aware and price-aware portfolio strategies with backtesting

## Supported Stocks

Major S&P 500 stocks across 7 sectors:

| Sector | Tickers |
|--------|---------|
| Technology | NVDA, AAPL, MSFT, AVGO, ORCL |
| Communication Services | GOOGL, META |
| Consumer Discretionary | AMZN, TSLA, HD |
| Financial Services | BRK.B, JPM, V, MA |
| Healthcare | JNJ, LLY, UNH |
| Consumer Staples | WMT, PG |
| Energy | XOM |

## Technology Stack

- **Languages:** Python 3.12+
- **ML/NLP:** scikit-learn, Transformers (FinBERT/FinRoBERTa)
- **Data handling:** Pandas, NumPy, yfinance
- **Web Scraping:** Selenium, undetected-chromedriver, GDELT API
- **Visualization:** Matplotlib, Jupyter Notebook

## Project Structure*
*Not final: Proposed structure

```
FYP/
├── Pipeline/                              # Main analysis and modeling
│   ├── News_aware_PO.ipynb               # News-aware portfolio optimization
│   └── Price_aware_PO copy.ipynb         # Price-aware portfolio optimization
│
├── scrapers/                              # Data collection scripts
│   ├── GDELTscraper.py                   # GDELT news API scraper
│   ├── twitter_scraper.py                # Twitter/X scraper (primary)
│   └── twitter_scraper2.py               # Twitter/X scraper (Second X account after rate limited)
│
├── preprocessing/                         # Data preprocessing & labelling
│   ├── news_preprocessing_labelling.ipynb    # News data preprocessing
│   └── tweets_preprocessing_labelling.ipynb  # Tweets preprocessing & labelling
│
├── output/                                # Back-testing results
│   ├── performance_metrics.csv           # Strategy performance metrics
│   ├── portfolio_performance.csv         # Portfolio daily performance
│   ├── price_based_performance_metrics.csv
│   ├── price_based_portfolio_performance.csv
│   ├── price_based_trade_log.csv
│   └── trade_log.csv                     # Trading history
│
├── tweets_labelled/                       # Labelled tweet data
├── tweets_*.csv                           # Raw scraped tweets per ticker
├── .env                                   # Environment variables
└── .gitignore                             # Git ignore file
```

## Installation

### Prerequisites

- Python 3.12+
- Anaconda or Miniconda (recommended)
- Chrome browser (for Twitter scraping)
- CUDA-capable GPU with 8GB+ VRAM (optional, for faster inference)

### Setup

1. **Create conda environment (recommended):**
   ```bash
   conda create -n project python=<python_version>
   conda activate project
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

3. **Configure environment variables:**

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

### Step 2: Collect Twitter Data
```bash
python scrapers/twitter_scraper.py
```

### Step 3: Preprocess Data
Run the Jupyter notebooks in `preprocessing/`:
- `news_preprocessing_labelling.ipynb` - Process and label news data
- `tweets_preprocessing_labelling.ipynb` - Process and label tweet data

### Step 4: Portfolio Optimization
Run the Jupyter notebooks in `Pipeline/`:
- `News_aware_PO.ipynb` - News sentiment-aware portfolio optimization
- `Price_aware_PO copy.ipynb` - Price-based portfolio optimization

## Output Files

| File | Description |
|------|-------------|
| `performance_metrics.csv` | Strategy performance summary metrics |
| `portfolio_performance.csv` | Daily portfolio value tracking |
| `strategy_trade_log.csv` | Complete trading history with buy/sell actions |
| `price_based_trade_log*.csv` | Price-based strategy buy/sell |

## Models

### Custom FIN-RoBERTa Model

We trained a custom RoBERTa-based model for financial sentiment analysis:

- **Model:** [alasteirho/FIN-RoBERTa-Custom](https://huggingface.co/alasteirho/FIN-RoBERTa-Custom)
- **Labels:** negative, neutral, positive

#### Training Datasets

| Dataset | Description |
|---------|-------------|
| [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | Financial news sentences with sentiment labels (sentences_allagree - 50% annotator agreement, 80:20 split) |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | Twitter financial news sentiment dataset |
| [pauri32/fiqa-2018](https://huggingface.co/datasets/pauri32/fiqa-2018) | Financial Opinion Mining and Question Answering dataset |
| [SemEval-2017 Task 5 Subtask 2](https://alt.qcri.org/semeval2017/task5/) | Fine-grained sentiment analysis on financial news headlines |

#### Evaluation Dataset

The model was evaluated on the [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) dataset (sentences_allagree subset) which contains 2,264 sentences with 100% annotator agreement, ensuring high-quality ground truth labels.

### Model Evaluation

The model evaluation notebooks are located in `model/`:
- `finbert.ipynb` - FinBERT evaluation
- `finroberta.ipynb` - Custom FIN-RoBERTa evaluation

## Features

- **Data Leakage Prevention:** Uses only t-1 data to predict future returns
- **Multiple Data Sources:** Financial news (GDELT) and social media (Twitter/X)
- **Custom FIN-RoBERTa Model:** Domain-specific transformer model fine-tuned on financial text
- **Backtesting Framework:** Daily mark-to-market with benchmark comparison
- **Current Portfolio Strategies:** News-aware and price-aware optimization approaches

## License

This project is part of my Final Year Project at the University of Greenwich.




