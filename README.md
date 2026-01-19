# Meta Learning for Multi-Source Sentiment Analysis

A meta-learning approach to financial sentiment analysis that integrates multiple data sources (news articles and social media) to predict stock price movements and optimize investment portfolios.

**Author:** Alasteir Ho
**Institution:** University of Greenwich (Final Year Project)

## Overview

This system performs the following workflow:

1. **Data Collection** - Scrapes news articles from GDELT API and Twitter/X posts for major stocks
2. **Preprocessing & Labelling** - Cleans and labels data for sentiment analysis
3. **Portfolio Optimization** - Implements news-aware and price-aware portfolio strategies with backtesting

## Supported Stocks

Major S&P 500 stocks across multiple sectors:

| Sector | Tickers |
|--------|---------|
| Technology | NVDA, AAPL, MSFT, AVGO, ORCL, GOOGL, META, AMZN, TSLA |
| Finance | BRK.B, JPM, V, MA |
| Energy & Consumer | XOM, HD |

## Technology Stack

- **Languages:** Python 3.8+
- **ML/NLP:** scikit-learn, Transformers (FinBERT/FinVADER)
- **Data handling:** Pandas, NumPy, yfinance
- **Web Scraping:** Selenium, undetected-chromedriver, GDELT API
- **Visualization:** Matplotlib, Jupyter Notebook

## Project Structure

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
├── .env                                   # Environment variables (Twitter credentials)
└── .gitignore                             # Git ignore file
```

## Installation

### Prerequisites

- Python 3.12+
- Chrome browser (for Twitter scraping)
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn torch transformers
   pip install matplotlib jupyter yfinance
   pip install selenium undetected-chromedriver
   pip install gdelt python-dotenv
   ```

3. **Configure environment variables:**

   Create a `.env` file with Twitter credentials:
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
| `trade_log.csv` | Complete trading history with buy/sell actions |
| `price_based_*.csv` | Price-based strategy outputs |

## Features

- **Data Leakage Prevention:** Uses only t-1 data to predict future returns
- **Multiple Data Sources:** Financial news (GDELT) and social media (Twitter/X)
- **FinBERT Sentiment Analysis:** Domain-specific NLP model for financial text
- **Backtesting Framework:** Daily mark-to-market with benchmark comparison
- **Portfolio Strategies:** News-aware and price-aware optimization approaches

## License

This project is part of a Final Year Project at the University of Greenwich.
