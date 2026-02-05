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

# Search terms used for each ticker
# Important Consideration: GDELT API requires search keywords to be at least 4 characters
TICKERS = {
    # Tech / Hardware
    "NVDA": ["Nvidia", "Nvidia stock", "Jensen Huang", "Nvidia GPU"],
    "AAPL": ["Apple Inc", "Apple stock", "iPhone Apple", "Tim Cook Apple"],
    
    # Tech / Software
    "GOOGL": ["Google", "Alphabet Inc", "Sundar Pichai", "Google stock"],
    "META": ["Meta Platforms", "Facebook", "Mark Zuckerberg", "Instagram Meta"],
    "MSFT": ["Microsoft", "Microsoft stock", "Satya Nadella", "Azure Microsoft"],
    "AVGO": ["Broadcom", "Broadcom stock", "Broadcom chip", "AVGO stock"],
    "ORCL": ["Oracle Corporation", "Oracle stock", "Oracle cloud", "Larry Ellison Oracle"],
    
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

START_DATE = datetime(2023, 10, 10, 0, 0, 0)  # 12:00 AM on 10/10/2023
END_DATE = datetime(2025, 10, 10, 23, 59, 59)  # 11:59 PM on 10/10/2025
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Raw_Data", "gdelt_news_data")

# GDELT API limits
MAX_RECORDS = 250  # Max per query (limit to avoid overwhelming the API)
RATE_LIMIT_DELAY = 3  # Seconds between requests (Prevent rate limiting)
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


# Functions for Scraping
# Load existing CSV data for a ticker and return (DataFrame, set of scraped dates).
def load_existing_data(ticker: str, output_dir: str) -> tuple:
    filepath = os.path.join(output_dir, f"{ticker}_news.csv")
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if 'date' in df.columns and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                # Remove rows with invalid/missing dates (corrupted rows)
                original_count = len(df)
                df = df.dropna(subset=['date'])

                # Also remove rows with missing critical fields (headline, source)
                if 'headline' in df.columns:
                    df = df.dropna(subset=['headline'])
                if 'source' in df.columns:
                    df = df.dropna(subset=['source'])

                cleaned_count = len(df)
                if original_count != cleaned_count:
                    print(f"[CLEANED] Removed {original_count - cleaned_count} corrupted/incomplete rows")
                    # Save the cleaned data back to CSV
                    df.to_csv(filepath, index=False)
                    print(f"[SAVED] Cleaned CSV saved with {cleaned_count} valid rows")

                # Gets unique dates that have been scraped
                scraped_dates = set(df['date'].dt.date.dropna())
                print(f" [LOADED] Found {len(df)} existing articles ({len(scraped_dates)} unique dates)")
                return df, scraped_dates
        except Exception as e:
            print(f" [ERROR] Could not load existing CSV data: {e}")

    return pd.DataFrame(), set()

# Find all gap ranges that need to be scraped (dates without any articles).
def get_gap_ranges(start_date: datetime, end_date: datetime, scraped_dates: set) -> list:
    if not scraped_dates:
        # No existing data, return full range
        return [(start_date, end_date)]

    # Generate all expected dates in range
    all_dates = set()
    current = start_date.date()
    end = end_date.date()
    while current <= end:
        all_dates.add(current)
        current += timedelta(days=1)

    # Find missing dates
    missing_dates = sorted(all_dates - scraped_dates)

    if not missing_dates:
        return []

    # Group consecutive missing dates into ranges
    gap_ranges = []
    gap_start = missing_dates[0]
    gap_end = missing_dates[0]

    for date in missing_dates[1:]:
        if date == gap_end + timedelta(days=1):
            gap_end = date
        else:
            # Convert to datetime for consistency
            gap_ranges.append((
                datetime.combine(gap_start, datetime.min.time()),
                datetime.combine(gap_end, datetime.max.time())
            ))
            gap_start = date
            gap_end = date

    # Add the last gap
    gap_ranges.append((
        datetime.combine(gap_start, datetime.min.time()),
        datetime.combine(gap_end, datetime.max.time())
    ))

    print(f"[GAPS] Found {len(gap_ranges)} gap range(s) totaling {len(missing_dates)} days")
    return gap_ranges

# Fetch news articles for a single ticker within the date range.
def fetch_news_for_ticker(ticker: str, keywords: list, start: datetime, end: datetime) -> pd.DataFrame:
    gd = GdeltDoc()
    all_articles = []

    # GDELT API end_date is exclusive, so add 1 day to include the end date
    api_end_date = end + timedelta(days=1)

    for keyword in keywords:
        try:
            f = Filters(
                keyword=keyword,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=api_end_date.strftime("%Y-%m-%d"),
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

# Filter articles and only include those from mentioned reputable sources.
def filter_reputable_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    # Check which column contains the domain/source
    source_col = 'domain' if 'domain' in df.columns else 'source' if 'source' in df.columns else None
    
    if source_col is None:
        print("[WARNING] No source column found, skipping filter")
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
    print(f"[FILTER] {original_count} to {filtered_count} articles (reputable sources only)")
    
    return df

# Scrape all news for a single ticker, filling in any gaps in the data.
def scrape_ticker(ticker: str, keywords: list, start_date: datetime, end_date: datetime, output_dir: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"Processing: {ticker}")
    print(f"Keywords: {keywords}")
    print(f"{'='*50}")

    # Load existing data
    existing_df, scraped_dates = load_existing_data(ticker, output_dir)

    # Find all gap ranges that need scraping
    gap_ranges = get_gap_ranges(start_date, end_date, scraped_dates)

    if not gap_ranges:
        print(f" [COMPLETE] No gaps found for {ticker}")
        return existing_df

    print(f" [INFO] {len(gap_ranges)} gap range(s) to scrape")
    
    # Start with existing data and track existing URLs/headlines
    all_data = existing_df.copy() if not existing_df.empty else pd.DataFrame()
    
    # Build sets of existing URLs and headlines to check against
    existing_urls = set()
    existing_headlines = set()
    if not all_data.empty:
        if 'url' in all_data.columns:
            existing_urls = set(all_data['url'].dropna().str.strip())
        headline_col = 'title' if 'title' in all_data.columns else 'headline'
        if headline_col in all_data.columns:
            existing_headlines = set(all_data[headline_col].dropna().str.strip())
    
    for i, (gap_start, gap_end) in enumerate(gap_ranges):
        # Display the actual API search dates (end_date + 1 day since GDELT is exclusive)
        api_end_date = gap_end + timedelta(days=1)
        print(f"\n Gap {i+1}/{len(gap_ranges)}: {gap_start.date()} to {api_end_date.date()}")

        chunk_df = fetch_news_for_ticker(ticker, keywords, gap_start, gap_end)
        if len(chunk_df) > 0:
            # Filter out articles with duplicate URLs or headlines BEFORE adding
            headline_col = 'title' if 'title' in chunk_df.columns else 'headline'
            original_count = len(chunk_df)
            
            # Filter by URL
            if 'url' in chunk_df.columns:
                chunk_df = chunk_df[~chunk_df['url'].fillna('').str.strip().isin(existing_urls)]
            
            # Filter by headline
            if headline_col in chunk_df.columns:
                chunk_df = chunk_df[~chunk_df[headline_col].fillna('').str.strip().isin(existing_headlines)]
            
            # Also remove duplicates within the chunk itself
            chunk_df = chunk_df.drop_duplicates(subset=['url'], keep='first')
            if headline_col in chunk_df.columns:
                chunk_df = chunk_df.drop_duplicates(subset=[headline_col], keep='first')
            
            filtered_count = len(chunk_df)
            if original_count != filtered_count:
                print(f"[FILTERED] {original_count - filtered_count} duplicates skipped")
            
            if len(chunk_df) > 0:
                # Add new URLs and headlines to tracking sets
                if 'url' in chunk_df.columns:
                    existing_urls.update(chunk_df['url'].dropna().str.strip())
                if headline_col in chunk_df.columns:
                    existing_headlines.update(chunk_df[headline_col].dropna().str.strip())
                
                # Merge chunk with all data
                if not all_data.empty:
                    all_data = pd.concat([all_data, chunk_df], ignore_index=True)
                else:
                    all_data = chunk_df.copy()
                
                # Save after each chunk
                clean_and_save(all_data, ticker, output_dir)
                print(f"[SAVED] {len(all_data)} articles after chunk {i+1}")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    return all_data if not all_data.empty else pd.DataFrame()

# Clean dataframe and save to CSV
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
            print(f"[DEDUP] Removed {before_count - len(df)} duplicate headlines")
    
    # Save
    filepath = os.path.join(output_dir, f"{ticker}_news.csv")
    df.to_csv(filepath, index=False)
    print(f"\nSaved {len(df)} unique articles to {filepath}")


def main():
    # User friendly formatting for console output
    print("="*60)
    print("Beginning of GDELT News Scraping Process")
    print("="*60)
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Tickers: {list(TICKERS.keys())}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Scrape news for each ticker, automatically filling any gaps
    for ticker, keywords in TICKERS.items():
        df = scrape_ticker(ticker, keywords, START_DATE, END_DATE, OUTPUT_DIR)
        clean_and_save(df, ticker, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Scraping has been completed")
    print("="*60)


if __name__ == "__main__":
    main()