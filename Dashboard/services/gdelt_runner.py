"""CLI wrapper around GDELTscraper.py.

Usage:
    python gdelt_runner.py --start 2026-01-01 --end 2026-03-01 --tickers AAPL MSFT NVDA
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Import the bundled scraper from this same folder
SERVICES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICES_DIR))

import GDELTscraper  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser(description="Run GDELT news scraper with custom parameters")
    parser.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols to scrape")
    return parser.parse_args()


def main():
    args = parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    # Filter TICKERS dict to only the requested tickers
    requested = set(args.tickers)
    filtered_tickers = {k: v for k, v in GDELTscraper.TICKERS.items() if k in requested}

    if not filtered_tickers:
        print(f"[ERROR] None of the requested tickers found in scraper config: {args.tickers}")
        sys.exit(1)

    # Override module-level globals (main() reads these at call time)
    GDELTscraper.START_DATE = start_dt
    GDELTscraper.END_DATE   = end_dt
    GDELTscraper.TICKERS    = filtered_tickers

    GDELTscraper.main()


if __name__ == "__main__":
    main()
