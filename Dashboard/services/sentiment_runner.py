"""CLI runner for FIN-RoBERTa sentiment classification.

Usage:
    python sentiment_runner.py --source news --tickers AAPL MSFT NVDA
    python sentiment_runner.py --source tweets --tickers AAPL MSFT
    python sentiment_runner.py --source both --tickers AAPL MSFT NVDA
"""

import argparse
import sys
from pathlib import Path

# Ensure Product/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sentiment import load_model  # noqa: E402
from backend.news_preprocessing import classify_news  # noqa: E402
from backend.tweets_preprocessing import classify_tweets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run FIN-RoBERTa sentiment classification")
    parser.add_argument(
        "--source",
        required=True,
        choices=["news", "tweets", "both"],
        help="Which raw data to classify",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker symbols to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer, model, device = load_model()

    if args.source in ("news", "both"):
        print("\n========== NEWS CLASSIFICATION ==========")
        classify_news(args.tickers, tokenizer, model, device, args.batch_size)

    if args.source in ("tweets", "both"):
        print("\n========== TWEET CLASSIFICATION ==========")
        classify_tweets(args.tickers, tokenizer, model, device, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
