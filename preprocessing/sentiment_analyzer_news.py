import os
import re
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import yfinance as yf

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = str(BASE_DIR / "gdelt_news_data")
OUTPUT_DIR = BASE_DIR / "processed_data" / "news_sentiment_daily"

# Load FinBERT model and tokenizer
print("Loading FinBERT model from ProsusAI/finbert...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Move model to nvidia based GPU if available or else use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded onto {device}")

# Preprocess news title for sentiment analysis.
def preprocess_title(title):
    if pd.isna(title) or not isinstance(title, str):
        return None

    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()

    # Remove URLs
    title = re.sub(r'http\S+|www\.\S+', '', title)

    # Remove ticker symbols in format (NASDAQ:XXX) or (NYSE:XXX)
    title = re.sub(r'\(\s*(NASDAQ|NYSE|AMEX)\s*:\s*\w+\s*\)', '', title)

    # Remove standalone ticker mentions like "- AAPL" at end
    title = re.sub(r'\s*-\s*[A-Z]{1,5}\s*$', '', title)

    # Remove special characters but keep basic punctuation
    title = re.sub(r'[^\w\s\.,!?\'\"-]', '', title)

    # Remove extra whitespace again after cleaning
    title = re.sub(r'\s+', ' ', title).strip()

    # Skip if title is too short after preprocessing
    if len(title) < 10:
        return None

    return title


def get_sentiment_score(texts, batch_size=16):
    """Get sentiment scores for a list of texts using FinBERT.

    Returns a score between -1 (negative) and 1 (positive).
    """
    if not texts:
        return []

    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)

        # FinBERT outputs: [negative, neutral, positive]
        # Convert to single score: positive - negative (range: -1 to 1)
        batch_scores = predictions[:, 2] - predictions[:, 0]
        scores.extend(batch_scores.cpu().numpy().tolist())

    return scores


def process_file(input_path, ticker):
    # Process a single CSV file and return sentiment results.
    print(f"\nProcessing: {os.path.basename(input_path)} (Ticker: {ticker})")

    # Read the CSV file
    df = pd.read_csv(input_path)

    # Check if required columns exist
    if 'headline' not in df.columns or 'date' not in df.columns:
        print(f"Skipping: Missing 'headline' or 'date' column")
        return

    # Preprocess titles
    print("Preprocessing titles")
    df['clean_headline'] = df['headline'].apply(preprocess_title)

    # Remove rows with invalid titles
    valid_df = df.dropna(subset=['clean_headline']).copy()

    if len(valid_df) == 0:
        print("No valid headlines found")
        return

    print(f"Processing {len(valid_df)} valid headlines...")

    # Get sentiment scores
    headlines = valid_df['clean_headline'].tolist()
    scores = get_sentiment_score(headlines)
    valid_df['sentiment_score'] = scores

    valid_df['datetime'] = pd.to_datetime(valid_df['date'])
    valid_df['sentiment_score'] = valid_df['sentiment_score'].round(4)
    result = valid_df[['datetime', 'sentiment_score']].copy()
    result['ticker'] = ticker
    result = result.sort_values('datetime')
    return result


def extract_ticker(filename):
    # Extract ticker symbol from filename (e.g., 'AAPL_news.csv' -> 'AAPL').
    # Remove .csv extension and common suffixes
    base = filename.replace('.csv', '')

    # Handle patterns like 'AAPL_news', 'tweets_AMD', etc.
    if '_news' in base:
        return base.replace('_news', '')
    elif base.startswith('tweets_'):
        return base.replace('tweets_', '')
    elif '_' in base:
        # Try to extract ticker from other patterns
        parts = base.split('_')
        # Return the part that looks like a ticker (uppercase, short)
        for part in parts:
            if part.isupper() and 1 <= len(part) <= 5:
                return part

    # Skip files that don't have a clear ticker
    return None


def fetch_price_data(ticker, start_date, end_date):
    # Fetch opening and closing prices for a ticker from yfinance.
    try:
        print(f"  Fetching price data for {ticker}...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if df.empty:
            print(f"  Warning: No price data available for {ticker}")
            return None

        # Ensure Date column exists
        if "Date" not in df.columns:
            print(f"  Warning: Date column not found for {ticker}")
            return None

        # Convert to proper types
        df["Date"] = pd.to_datetime(df["Date"])
        df["date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Extract Open and Close prices
        price_df = df[["date"]].copy()

        if "Open" in df.columns:
            price_df["open_price"] = pd.to_numeric(df["Open"], errors="coerce")
        else:
            price_df["open_price"] = None

        if "Close" in df.columns:
            price_df["close_price"] = pd.to_numeric(df["Close"], errors="coerce")
        else:
            price_df["close_price"] = None

        return price_df

    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return None


def main():
    # Process all CSV files in the input directory.
    # Get list of CSV files
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    results = []
    for filename in tqdm(csv_files, desc="Processing files"):
        input_path = os.path.join(INPUT_DIR, filename)

        # Extract ticker from filename
        ticker = extract_ticker(filename)
        if not ticker:
            print(f"\nSkipping {filename}: Could not extract the ticker symbol")
            continue

        try:
            result = process_file(input_path, ticker)
            if result is not None and not result.empty:
                results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not results:
        print("\nNo sentiment records generated.")
        return

    combined = pd.concat(results, ignore_index=True)
    combined = combined.dropna(subset=["datetime", "ticker", "sentiment_score"])
    combined = combined.sort_values(["datetime", "ticker"])

    # Daily average sentiment per ticker
    combined["date"] = pd.to_datetime(combined["datetime"]).dt.strftime("%Y-%m-%d")
    daily = (
        combined.groupby(["date", "ticker"], as_index=False)["sentiment_score"]
        .mean()
        .rename(columns={"sentiment_score": "avg_sentiment"})
        .sort_values(["ticker", "date"])
    )

    # Get date range for price data fetching
    min_date = daily["date"].min()
    max_date = daily["date"].max()

    print(f"\nFetching price data for date range: {min_date} to {max_date}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = sorted(daily["ticker"].unique().tolist())

    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        ticker_sentiment = daily[daily["ticker"] == ticker][["date", "avg_sentiment"]].copy()

        # Fetch price data for this ticker
        price_data = fetch_price_data(ticker, min_date, max_date)

        if price_data is not None and not price_data.empty:
            # Merge sentiment with price data
            ticker_data = ticker_sentiment.merge(price_data, on="date", how="left")
            print(f"Merged {len(ticker_data)} records with price data")
        else:
            # If no price data available, just save sentiment
            ticker_data = ticker_sentiment.copy()
            ticker_data["open_price"] = None
            ticker_data["close_price"] = None
            print("No price data available, saving sentiment only")

        # Save to CSV
        out_path = OUTPUT_DIR / f"{ticker}_news_sentiment_daily.csv"
        ticker_data.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")

    print(f"\nCompleted processing for {len(tickers)} tickers")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
