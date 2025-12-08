"""
GDELT News Scraper for Stock Sentiment Analysis
Scrapes news headlines related to specified stock tickers using the GDELT API.
Date range: 2023-10-10 to 2025-10-10
By Alasteir Ho

"""

import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta
import time
import os

# Variables and Configuration
# ============================================
# Note: The GDELT API requires keywords to be at least 4 characters
TICKERS = {
    # Tech
    "NVDA": ["Nvidia", "Nvidia stock", "Jensen Huang", "Nvidia GPU"],
    "AAPL": ["Apple Inc", "Apple stock", "iPhone Apple", "Tim Cook Apple"],
    "MSFT": ["Microsoft", "Microsoft stock", "Satya Nadella", "Azure Microsoft"],
    "AVGO": ["Broadcom", "Broadcom stock", "Broadcom chip", "AVGO stock"],
    "ORCL": ["Oracle Corporation", "Oracle stock", "Oracle cloud", "Larry Ellison Oracle"],
    
    # Tech / Comms
    "GOOGL": ["Google", "Alphabet Inc", "Sundar Pichai", "Google stock"],
    "META": ["Meta Platforms", "Facebook", "Mark Zuckerberg", "Instagram Meta"],
    
    # Tech / Consumer
    "AMZN": ["Amazon", "Amazon stock", "Jeff Bezos Amazon", "AWS Amazon"],
    "TSLA": ["Tesla", "Tesla stock", "Elon Musk Tesla", "Tesla EV"],
    
    # Finance
    "BRK-B": ["Berkshire Hathaway", "Warren Buffett", "Berkshire stock"],
    "JPM": ["JPMorgan Chase", "JPMorgan stock", "Jamie Dimon"],
    "V": ["Visa Inc", "Visa stock", "Visa payments"],
    "MA": ["Mastercard", "Mastercard stock", "Mastercard payments"],
    
    # Healthcare
    "LLY": ["Eli Lilly", "Lilly stock", "Eli Lilly pharma"],
    "JNJ": ["Johnson Johnson", "JNJ stock", "Johnson pharma"],
    "UNH": ["UnitedHealth", "UnitedHealth stock", "UnitedHealth Group"],
    
    # Energy
    "XOM": ["Exxon Mobil", "ExxonMobil stock", "Exxon oil"],
    
    # Consumer Staples
    "WMT": ["Walmart", "Walmart stock", "Walmart retail"],
    "PG": ["Procter Gamble", "Procter stock", "P&G consumer"],
    
    # Consumer Discretionary
    "HD": ["Home Depot", "Home Depot stock", "Home Depot retail"],
}

START_DATE = datetime(2023, 10, 10)
END_DATE = datetime(2025, 10, 10)
OUTPUT_DIR = "gdelt_news_data"

# GDELT limits queries, chunk into smaller periods
CHUNK_DAYS = 14  # 2-week chunks work well
MAX_RECORDS = 250  # Max per query (API limit)
RATE_LIMIT_DELAY = 3  # Seconds between requests (increased to avoid recursion errors)
MAX_RETRIES = 3  # Number of retries for failed requests

# Reputable financial news sources to filter by
REPUTABLE_SOURCES = {
    # Major Financial News
    'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com', 'cnbc.com',
    'marketwatch.com', 'barrons.com', 'investors.com', 'finance.yahoo.com',
    'seekingalpha.com', 'benzinga.com', 'thestreet.com', 'investopedia.com',
    
    # Business news sources
    'forbes.com', 'businessinsider.com', 'fortune.com', 'economist.com',
    
    # General News with business sections
    'nytimes.com', 'washingtonpost.com', 'bbc.com', 'bbc.co.uk', 'theguardian.com',
    'apnews.com', 'npr.org', 'cnn.com', 'nbcnews.com', 'abcnews.go.com',
    
    # Tech related News
    'techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com', 'zdnet.com',
    'cnet.com', 'engadget.com', 'venturebeat.com',
    
    # Wire Services
    'prnewswire.com', 'globenewswire.com', 'businesswire.com',
}



# SCRAPING FUNCTIONS
# ============================================
"""Load existing CSV data for a ticker and return (DataFrame, set of scraped dates)."""
def load_existing_data(ticker: str, output_dir: str) -> tuple:
    filepath = os.path.join(output_dir, f"{ticker}_news.csv")
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Gets unique dates that have been scraped
                scraped_dates = set(df['date'].dt.date.dropna())
                print(f"  [LOADED] Found {len(df)} existing articles ({len(scraped_dates)} unique dates)")
                return df, scraped_dates
        except Exception as e:
            print(f"  [WARNING] Could not load existing CSV data: {e}")
    
    return pd.DataFrame(), set()

"""Filter out date chunks that have already been fully scraped."""
def get_unscraped_chunks(date_chunks: list, scraped_dates: set) -> list:
    unscraped = []
    for chunk_start, chunk_end in date_chunks:
        # Check if any date in this chunk range hasn't been scraped
        chunk_dates = set()
        current = chunk_start
        while current <= chunk_end:
            chunk_dates.add(current.date())
            current += timedelta(days=1)
        
        # If there are dates in this chunk not in scraped_dates, include it
        if not chunk_dates.issubset(scraped_dates):
            unscraped.append((chunk_start, chunk_end))
    
    return unscraped

"""Fetch news articles for a single ticker within the date range."""
def fetch_news_for_ticker(ticker: str, keywords: list, start: datetime, end: datetime) -> pd.DataFrame:
    gd = GdeltDoc()
    all_articles = []
    
    for keyword in keywords:
        try:
            f = Filters(
                keyword=keyword,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                num_records=MAX_RECORDS,
                language="English"
            )
            articles = gd.article_search(f)
            
            if articles is not None and len(articles) > 0:
                articles['search_keyword'] = keyword
                articles['ticker'] = ticker
                all_articles.append(articles)
                print(f"{keyword}: {len(articles)} articles")
            else:
                print(f"{keyword}:contains 0 articles")
                
        except Exception as e:
            print(f"Error fetching '{keyword}': {e}")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    if all_articles:
        df = pd.concat(all_articles, ignore_index=True)
        # Filter to only reputable sources
        df = filter_reputable_sources(df)
        return df
    return pd.DataFrame()

""" To filter articles and only include those from mentioned reputable sources."""
def filter_reputable_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # Check which column contains the domain/source
    source_col = 'domain' if 'domain' in df.columns else 'source' if 'source' in df.columns else None
    
    if source_col is None:
        print("    [WARNING] No source column found, skipping filter")
        return df
    
    original_count = len(df)
    
    # Filter by checking if source contains any reputable domain
    def is_reputable(source):
        if pd.isna(source):
            return False
        source_lower = str(source).lower()
        return any(rep_source in source_lower for rep_source in REPUTABLE_SOURCES)
    
    df = df[df[source_col].apply(is_reputable)]
    
    filtered_count = len(df)
    print(f"    [FILTER] {original_count} → {filtered_count} articles (reputable sources only)")
    
    return df

"""Generate list of (start, end) date tuples."""
def generate_date_chunks(start: datetime, end: datetime, chunk_days: int) -> list:
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks

"""Scrape all news for a single ticker across all date chunks, skipping already scraped data."""
def scrape_ticker(ticker: str, keywords: list, date_chunks: list, output_dir: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"Processing: {ticker}")
    print(f"Keywords: {keywords}")
    print(f"{'='*50}")
    
    # Load existing data
    existing_df, scraped_dates = load_existing_data(ticker, output_dir)
    
    # Filter out already scraped chunks
    unscraped_chunks = get_unscraped_chunks(date_chunks, scraped_dates)
    
    if not unscraped_chunks:
        print(f"  [SKIPPING] All the date chunks already scraped for {ticker}")
        return existing_df
    
    print(f"  [INFO] {len(unscraped_chunks)}/{len(date_chunks)} chunks need scraping")
    
    ticker_data = []
    
    for i, (chunk_start, chunk_end) in enumerate(unscraped_chunks):
        print(f"\n  Chunk {i+1}/{len(unscraped_chunks)}: {chunk_start.date()} → {chunk_end.date()}")
        
        chunk_df = fetch_news_for_ticker(ticker, keywords, chunk_start, chunk_end)
        if len(chunk_df) > 0:
            ticker_data.append(chunk_df)
        
        time.sleep(RATE_LIMIT_DELAY)
    
    # Combine new data with the existing
    if ticker_data:
        new_df = pd.concat(ticker_data, ignore_index=True)
        if not existing_df.empty:
            # Merge existing and new data
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates by URL and headline
            final_df = final_df.drop_duplicates(subset=['url'], keep='first')
            headline_col = 'title' if 'title' in final_df.columns else 'headline'
            if headline_col in final_df.columns:
                final_df = final_df.drop_duplicates(subset=[headline_col], keep='first')
            print(f"  [MERGED] {len(existing_df)} existing + {len(new_df)} new = {len(final_df)} total (deduplicated)")
        else:
            final_df = new_df.drop_duplicates(subset=['url'], keep='first')
            headline_col = 'title' if 'title' in final_df.columns else 'headline'
            if headline_col in final_df.columns:
                final_df = final_df.drop_duplicates(subset=[headline_col], keep='first')
        return final_df
    
    return existing_df if not existing_df.empty else pd.DataFrame()

"""Clean dataframe and save to CSV."""
def clean_and_save(df: pd.DataFrame, ticker: str, output_dir: str) -> None:
    if df.empty:
        print(f"No articles found for {ticker}")
        return
    
    # Rename columns for consistency (only if they exist and haven't been renamed)
    rename_map = {}
    if 'title' in df.columns and 'headline' not in df.columns:
        rename_map['title'] = 'headline'
    if 'seendate' in df.columns and 'date' not in df.columns:
        rename_map['seendate'] = 'date'
    if 'domain' in df.columns and 'source' not in df.columns:
        rename_map['domain'] = 'source'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Drop duplicate columns if they exist
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Parse and sort by date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')
    
    # Select relevant columns
    keep_cols = ['ticker', 'date', 'headline', 'source', 'url', 'language', 'sourcecountry', 'search_keyword']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Remove duplicate headlines
    if 'headline' in df.columns:
        before_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        if before_count != len(df):
            print(f"  [DEDUP] Removed {before_count - len(df)} duplicate headlines")
    
    # Save
    filepath = os.path.join(output_dir, f"{ticker}_news.csv")
    df.to_csv(filepath, index=False)
    print(f"\n  ✓ Saved {len(df)} unique articles → {filepath}")


def main():
    # User friendly formatting for console output
    print("="*60)
    print("GDELT based Stock News Scraper")
    print("="*60)
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Tickers: {list(TICKERS.keys())}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_chunks = generate_date_chunks(START_DATE, END_DATE, CHUNK_DAYS)
    print(f"Generated {len(date_chunks)} date chunks ({CHUNK_DAYS}-day intervals)")
    
    # Collect news for each ticker
    all_data = []
    for ticker, keywords in TICKERS.items():
        df = scrape_ticker(ticker, keywords, date_chunks, OUTPUT_DIR)
        clean_and_save(df, ticker, OUTPUT_DIR)
        if not df.empty:
            all_data.append(df)
    
    # Creates a combined dataset for all tickers for collective analysis
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['url'], keep='first')
        combined = combined.sort_values('seendate' if 'seendate' in combined.columns else 'date')
        
        combined_path = os.path.join(OUTPUT_DIR, "all_tickers_news.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n{'='*60}")
        print(f"✓ Combined dataset: {len(combined)} total articles → {combined_path}")
    
    print(f"\n{'='*60}")
    print("Scraping has been completed")
    print("="*60)


if __name__ == "__main__":
    main()