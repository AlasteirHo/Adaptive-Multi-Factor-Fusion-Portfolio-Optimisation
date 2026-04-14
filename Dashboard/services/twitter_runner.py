"""CLI wrapper around twitter_scraper.py.

Usage:
    python twitter_runner.py --start 2026-01-01 --end 2026-03-01 --tickers AAPL MSFT NVDA

Credentials are loaded automatically from the .env file in the scrapers directory.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add scrapers directory to path and load .env from there
SCRAPERS_DIR = Path(__file__).resolve().parents[2] / "scrapers"
sys.path.insert(0, str(SCRAPERS_DIR))

from dotenv import load_dotenv
load_dotenv(SCRAPERS_DIR.parent / ".env")

from twitter_scraper import (  # noqa: E402
    TwitterScraper,
    check_ticker_completion,
    get_project_tweets_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Twitter scraper with custom parameters")
    parser.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols to scrape")
    return parser.parse_args()


def main():
    args = parse_args()

    USERNAME = os.getenv("TWITTER_USERNAME")
    PASSWORD = os.getenv("TWITTER_PASSWORD")

    if not USERNAME or not PASSWORD:
        print("[ERROR] TWITTER_USERNAME / TWITTER_PASSWORD not found in .env")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")

    # Twitter scraper uses BRK.B (not BRK-B) for that ticker
    tickers = [t.replace("BRK-B", "BRK.B") for t in args.tickers]

    tweets_dir = get_project_tweets_dir()
    os.makedirs(tweets_dir, exist_ok=True)

    print("=" * 60)
    print("Beginning Twitter/X Scraping Process")
    print("=" * 60)
    print(f"Date range : {start_dt.date()} to {end_dt.date()}")
    print(f"Tickers    : {tickers}")
    print(f"Output dir : {tweets_dir}")
    print("=" * 60)

    # Pre-check for already-completed tickers
    tickers_to_scrape = []
    for ticker in tickers:
        is_complete, latest_date, _ = check_ticker_completion(ticker, start_dt, end_dt, tweets_dir)
        if is_complete:
            print(f"{ticker}: COMPLETED (data up to {latest_date.strftime('%Y-%m-%d')})")
        else:
            if latest_date:
                print(f"{ticker}: Resume from {latest_date.strftime('%Y-%m-%d')}")
            else:
                print(f"{ticker}: Starting fresh")
            tickers_to_scrape.append(ticker)

    if not tickers_to_scrape:
        print("\nAll selected tickers are already completed.")
        return

    print(f"\nTickers to scrape: {tickers_to_scrape}")

    scraper = TwitterScraper(USERNAME, PASSWORD)
    try:
        print("Starting Chrome...")
        scraper.start_driver()
        print("Logging in...")
        scraper.login()
        print(f"Scraping {len(tickers_to_scrape)} ticker(s)...\n")

        for ticker in tickers_to_scrape:
            output_file = os.path.join(tweets_dir, f"tweets_{ticker}.csv")
            print(f"\n{'='*50}")
            print(f"Scraping ${ticker}")
            print(f"{'='*50}")
            scraper.scrape_date_range(ticker, start_dt, end_dt, output_file)
            print(f"Completed ${ticker} -> {output_file}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        scraper.close()

    print("\n" + "=" * 60)
    print("Twitter scraping completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
