"""FIN-RoBERTa model loading, batch scoring, and NYSE session utilities.

Shared by news_preprocessing.py and tweets_preprocessing.py.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import exchange_calendars

MODEL_ID = "alasteirho/FIN-RoBERTa-Custom"

_nyse_cal = exchange_calendars.get_calendar("XNYS")


# ---------------------------------------------------------------------------
# NYSE session mapping
# ---------------------------------------------------------------------------

def map_to_next_session(bucket_dates_str):
    """Map each bucket date to the first NYSE session >= that date."""
    bd = np.array(bucket_dates_str, dtype="datetime64[D]")
    d_min = str(bd.min() - np.timedelta64(30, "D"))
    d_max = str(bd.max() + np.timedelta64(30, "D"))
    sessions = _nyse_cal.sessions_in_range(d_min, d_max)
    sess_np = sessions.values.astype("datetime64[D]")
    idx = np.searchsorted(sess_np, bd, side="left")
    idx = np.clip(idx, 0, len(sess_np) - 1)
    return sess_np[idx].astype(str)


def assign_market_close_session(dt_series):
    """Assign UTC timestamps to NYSE sessions using the 16:00 ET cutoff.

    Articles/tweets at or before 16:00 ET belong to that day's session;
    those after 16:00 ET roll to the next session.  Weekends and holidays
    are handled via exchange_calendars (XNYS).
    """
    dt_utc = pd.to_datetime(dt_series, utc=True)
    ny = dt_utc.dt.tz_convert("America/New_York")
    sec = ny.dt.hour * 3600 + ny.dt.minute * 60 + ny.dt.second
    after_close = sec > 16 * 3600

    bucket_date = ny.dt.normalize()
    bucket_date = bucket_date.where(~after_close, bucket_date + pd.Timedelta(days=1))
    bucket_dates_str = bucket_date.dt.strftime("%Y-%m-%d").values
    return map_to_next_session(bucket_dates_str)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device=None):
    """Load FIN-RoBERTa tokenizer and model onto the best available device."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Loading FIN-RoBERTa model ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Batch sentiment scoring
# ---------------------------------------------------------------------------

def score_texts(texts, tokenizer, model, device, batch_size=16):
    """Return sentiment scores (P(positive) - P(negative)) for a list of strings.

    FIN-RoBERTa outputs logits for [negative, neutral, positive].
    """
    if not texts:
        return []

    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        batch_scores = (probs[:, 2] - probs[:, 0]).cpu().numpy().tolist()
        scores.extend(batch_scores)
    return scores
