"""Tweet preprocessing and FIN-RoBERTa classification.

Replicates the Wilksch & Abramova (2023) cleaning pipeline from
preprocessing/tweets_preprocessing_labelling.ipynb.

Pipeline per ticker:
    1. Load raw CSV from Raw_Data/Tweets/
    2. DataFrame-level filtering:
       a. Remove messaging-platform spam (WhatsApp, Discord, Telegram)
       b. Remove hyperlinks
       c. Drop duplicates (exact + bot-like repeated texts > 5 words)
       d. Filter tweets with >= 5 cashtags or >= 8 hashtags
       e. Spam ratio filters (cashtag/word, hashtag/word, mention/word <= 0.5)
       f. Remove cryptocurrency posts (<= 2 crypto keywords allowed)
       g. Filter non-English tweets (lingua, if installed)
    3. Text-level cleaning for FIN-RoBERTa:
       - Replace own $TICKER with company name
       - Remove other cashtags
       - Convert #hashtags to readable text
       - Remove emojis, special characters, excessive punctuation
       - Filter very short tweets (< 10 chars)
    4. Score with FIN-RoBERTa  ->  P(positive) - P(negative)
    5. Assign each tweet to a NYSE session (16:00 ET cutoff)
    6. Aggregate to daily avg_sentiment (+ total_replies, total_retweets, total_likes)
    7. Save to Processed_Data/tweets_sentiment_daily/
"""

import os
import re

import pandas as pd

from backend.config import RAW_TWEETS_DIR, SOCIAL_SENTIMENT_DIR
from backend.sentiment import score_texts, assign_market_close_session

# Optional: lingua for non-English tweet filtering
try:
    from lingua import LanguageDetectorBuilder, Language

    _LINGUA_AVAILABLE = True
    _lingua_detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_minimum_relative_distance(0.25)
        .build()
    )
except ImportError:
    _LINGUA_AVAILABLE = False
    _lingua_detector = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKER_TO_NAME = {
    "AAPL": "Apple", "AMZN": "Amazon", "AVGO": "Broadcom",
    "BRK.B": "Berkshire Hathaway", "BRK-B": "Berkshire Hathaway",
    "GOOGL": "Google", "HD": "Home Depot",
    "JNJ": "Johnson and Johnson", "JPM": "JPMorgan", "LLY": "Eli Lilly",
    "MA": "Mastercard", "META": "Meta", "MSFT": "Microsoft",
    "NVDA": "Nvidia", "ORCL": "Oracle", "PG": "Procter and Gamble",
    "TSLA": "Tesla", "UNH": "UnitedHealth", "V": "Visa",
    "WMT": "Walmart", "XOM": "ExxonMobil",
}

CRYPTO_KEYWORDS = {
    "bitcoin", "etherium", "btc", "eth", "nft", "token", "wallet",
    "web3", "airdrop", "wagmi", "solana", "opensea", "cryptopunks",
    "uniswap", "lunar", "hodl", "binance", "coinbase", "cryptocom", "doge",
}

_MSG_SPAM_RE = re.compile(
    r"wa\.me[/\s]|whatsapp|discord\.gg[/\s]|discord|telegram|t\.me[/\s]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DataFrame-level filters
# ---------------------------------------------------------------------------

def _is_messaging_spam(text):
    """Return True if text contains WhatsApp/Discord/Telegram references."""
    if pd.isna(text) or not isinstance(text, str):
        return False
    return bool(_MSG_SPAM_RE.search(text))


def _remove_hyperlinks(text):
    """Strip all URLs / hyperlinks from text."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"bit\.ly/\S+|goo\.gl/\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _count(pattern, text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(re.findall(pattern, text))


def _count_crypto(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & CRYPTO_KEYWORDS)


def _is_english(text, min_alpha_chars=20):
    """Check if text is English using lingua."""
    if not _LINGUA_AVAILABLE:
        return True
    if not text or not isinstance(text, str):
        return False

    sample = re.sub(r"http\S+|www\.\S+", " ", text)
    sample = re.sub(r"@\w+", " ", sample)
    sample = re.sub(r"\$[^\d\s]{1,5}", " ", sample)
    sample = sample.replace("#", " ")
    sample = re.sub(r"[^A-Za-z\s]", " ", sample)
    sample = re.sub(r"\s+", " ", sample).strip()
    if not sample:
        return False

    alpha_chars = len(re.sub(r"[^A-Za-z]", "", sample))
    if alpha_chars < min_alpha_chars:
        return True

    detected = _lingua_detector.detect_language_of(sample)
    return detected == Language.ENGLISH


def filter_dataframe(df):
    """Apply all DataFrame-level filters (steps 1-7).

    Returns the filtered DataFrame (rows may be dropped but no text
    cleaning for the model is done here).
    """
    n_raw = len(df)

    # 1. Remove messaging-platform spam
    df = df[~df["body"].apply(_is_messaging_spam)]

    # 2. Remove hyperlinks (needed for accurate word counts)
    df["body"] = df["body"].apply(_remove_hyperlinks)

    # 3. Drop duplicates
    df = df.drop_duplicates(subset=["body"])
    word_counts = df["body"].apply(
        lambda t: len(t.split()) if isinstance(t, str) else 0,
    )
    long = df[word_counts > 5].drop_duplicates(subset=["body"])
    short = df[word_counts <= 5]
    df = pd.concat([long, short], ignore_index=True)

    # 4. Filter by cashtag / hashtag counts
    df["_ct"] = df["body"].apply(lambda t: _count(r"\$[A-Za-z]{1,5}\b", t))
    df["_ht"] = df["body"].apply(lambda t: _count(r"#\w+", t))
    df = df[(df["_ct"] < 5) & (df["_ht"] < 8)]

    # 5. Spam ratio filters
    df["_mt"] = df["body"].apply(lambda t: _count(r"@\w+", t))
    df["_wc"] = df["body"].apply(
        lambda t: len(t.split()) if isinstance(t, str) else 0,
    )
    safe = df["_wc"] > 0
    df = df[
        ~safe
        | (
            (df["_ct"] / df["_wc"].clip(lower=1) <= 0.5)
            & (df["_ht"] / df["_wc"].clip(lower=1) <= 0.5)
            & (df["_mt"] / df["_wc"].clip(lower=1) <= 0.5)
        )
    ]

    # 6. Remove cryptocurrency posts
    df["_crypto"] = df["body"].apply(_count_crypto)
    df = df[df["_crypto"] <= 2]
    df = df.drop(columns=["_ct", "_ht", "_mt", "_wc", "_crypto"])

    # 7. Filter non-English
    if _LINGUA_AVAILABLE:
        df = df[df["body"].apply(_is_english)]

    n_clean = len(df)
    print(f"  DataFrame filter: {n_raw} raw -> {n_clean} kept")
    return df


# ---------------------------------------------------------------------------
# Text-level cleaning for FIN-RoBERTa
# ---------------------------------------------------------------------------

def clean_tweet_for_model(text, ticker=None):
    """Clean a single tweet for FIN-RoBERTa input.

    - Replaces the file's own $TICKER cashtag with the company name
    - Removes all other $TICKER cashtags
    - Converts #CamelCase hashtags to readable text
    - Strips emojis, special characters, excessive punctuation
    - Returns None if result is shorter than 10 characters
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove leading @mentions (reply chains)
    text = re.sub(r"^(@\w+\s*)+", "", text)

    # Convert #hashtags to readable text
    def _hashtag_to_text(m):
        tag = m.group(1)
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)

    text = re.sub(r"#(\w+)", _hashtag_to_text, text)

    # Replace own $TICKER with company name
    if ticker:
        company = TICKER_TO_NAME.get(ticker, ticker)
        escaped = re.escape(ticker)
        variants = [escaped]
        if "." in ticker:
            variants.append(re.escape(ticker.split(".")[0]))
        if "-" in ticker:
            variants.append(re.escape(ticker.split("-")[0]))
        own_pat = r"\$(?:" + "|".join(variants) + r")\b"
        text = re.sub(own_pat, company, text)

    # Remove remaining cashtags
    text = re.sub(r"\$[^\d\s]{1,5}", "", text)

    # Collapse excessive punctuation
    text = re.sub(r"([!?.])\\1{2,}", r"\1", text)

    # Strip emojis and special characters
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 10:
        return None
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _tweet_ticker(ticker):
    """Convert config ticker (BRK-B) to tweet filename ticker (BRK.B)."""
    return ticker.replace("-", ".")


def classify_tweets(tickers, tokenizer, model, device, batch_size=16):
    """Classify tweets and save daily sentiment CSVs.

    Checks existing processed data and only scores raw rows newer than
    the latest processed date, then appends the new daily aggregates.

    Parameters
    ----------
    tickers : list[str]
        Tickers to process (config format, e.g. BRK-B).
    tokenizer, model, device
        As returned by ``sentiment.load_model()``.
    batch_size : int
        Inference batch size (default 16).
    """
    os.makedirs(SOCIAL_SENTIMENT_DIR, exist_ok=True)

    if not _LINGUA_AVAILABLE:
        print("[WARN] lingua not installed; skipping non-English filter")

    for ticker in tickers:
        tweet_ticker = _tweet_ticker(ticker)
        csv_path = RAW_TWEETS_DIR / f"tweets_{tweet_ticker}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path.name} not found")
            continue

        df = pd.read_csv(csv_path)
        if "body" not in df.columns:
            print(f"[SKIP] {csv_path.name}: missing 'body' column")
            continue

        print(f"\n--- {ticker} tweets ---")
        print(f"  Raw rows: {len(df)}")

        # ---- Check existing processed data ----
        out_path = SOCIAL_SENTIMENT_DIR / f"{tweet_ticker}_tweets_sentiment_daily.csv"
        existing_daily = None
        latest_processed = None

        if out_path.exists():
            existing_daily = pd.read_csv(out_path)
            if "date" in existing_daily.columns and len(existing_daily) > 0:
                latest_processed = pd.to_datetime(existing_daily["date"]).max()
                print(f"  Processed data up to: {latest_processed.date()}")

        # ---- Filter raw data to only new rows ----
        if "post_date" not in df.columns:
            print("  [SKIP] missing 'post_date' column")
            continue

        df["_parsed_date"] = pd.to_datetime(df["post_date"], utc=True, errors="coerce")

        if latest_processed is not None:
            cutoff = pd.Timestamp(latest_processed, tz="UTC")
            df = df[df["_parsed_date"] > cutoff]
            if df.empty:
                print(f"  Already up to date ({len(existing_daily)} daily rows)")
                continue
            print(f"  New rows to process: {len(df)}")

        df = df.drop(columns=["_parsed_date"])

        # DataFrame-level filtering
        df = filter_dataframe(df)

        # Text-level cleaning for model
        df["cleaned_body"] = df["body"].apply(
            lambda t: clean_tweet_for_model(t, ticker=tweet_ticker),
        )
        df = df.dropna(subset=["cleaned_body"])

        if df.empty:
            print("  No tweets left after cleaning")
            continue

        # Score
        print(f"  Scoring {len(df)} tweets...")
        df["sentiment"] = score_texts(
            df["cleaned_body"].tolist(), tokenizer, model, device, batch_size,
        )
        df = df.dropna(subset=["sentiment"])

        # Assign to NYSE session via 16:00 ET cutoff
        df["trade_date"] = assign_market_close_session(df["post_date"])
        df["sentiment"] = df["sentiment"].astype(float).clip(-1, 1)

        # Aggregate to daily
        new_daily = df.groupby("trade_date").agg(**{
            "avg_sentiment": pd.NamedAgg(column="sentiment", aggfunc="mean"),
            **{
                f"total_{c}": pd.NamedAgg(column=c, aggfunc="sum")
                for c in ("replies", "retweets", "likes")
                if c in df.columns
            },
        }).reset_index().rename(columns={"trade_date": "date"})

        new_daily["avg_sentiment"] = new_daily["avg_sentiment"].round(4)

        # ---- Merge with existing processed data ----
        if existing_daily is not None and len(existing_daily) > 0:
            combined = pd.concat([existing_daily, new_daily], ignore_index=True)
            # For overlapping dates (boundary), keep the new computation
            combined = combined.drop_duplicates(subset=["date"], keep="last")
        else:
            combined = new_daily

        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(out_path, index=False)
        print(f"  Saved {len(combined)} daily rows -> {out_path.name} (+{len(new_daily)} new)")
