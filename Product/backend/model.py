"""
Adaptive Fusion Network: architecture, inference, and training.
Mirrors notebook cells 8 and 11.
"""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    BATCH_SIZE,
    CONTEXT_COLS,
    CONTEXT_DIM,
    TRAIN_START,
    DEVICE,
    DROPOUT_RATE,
    ENTROPY_LAMBDA,
    FACTOR_COLS,
    FWD_HORIZON,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_PATH,
    N_FACTORS,
    PIN_MEMORY,
    RANDOM_SEED,
    ROLLING_WINDOW,
    SOFTMAX_TEMP,
    TRAIN_END,
    TRAIN_EPOCHS,
    USE_AMP,
    USE_COMPILE,
    WEIGHT_DECAY,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class AdaptiveFusionNetwork(nn.Module):
    """Context-conditioned attention network that reweights eight z-scored factors."""

    def __init__(self, context_size=None):
        super().__init__()
        context_size = context_size or len(CONTEXT_COLS)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, CONTEXT_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.attention_logits = nn.Sequential(
            nn.Linear(CONTEXT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, N_FACTORS),
        )

    def forward(self, factors, context):
        context_encoded = self.context_encoder(context)
        logits = self.attention_logits(context_encoded)
        attention_weights = torch.softmax(logits / SOFTMAX_TEMP, dim=-1)
        composite_score = (attention_weights * factors).sum(dim=-1)
        return composite_score, attention_weights


def maybe_compile(model):
    if USE_COMPILE:
        try:
            return torch.compile(model)
        except Exception:
            return model
    return model


def get_composite_scores(model, feature_data, date, tickers):
    """
    Run batched inference for all tickers on a single rebalance date.
    Returns (scores dict, mean_attention dict).
    """
    valid_tickers, factor_rows, context_rows = [], [], []
    for ticker in tickers:
        ticker_df = feature_data.get(ticker)
        if ticker_df is None or date not in ticker_df.index:
            continue
        row = ticker_df.loc[date]
        valid_tickers.append(ticker)
        factor_rows.append(row[FACTOR_COLS].values.astype(np.float32))
        context_rows.append(row[CONTEXT_COLS].values.astype(np.float32))

    if not valid_tickers:
        return {}, {}

    factor_batch = torch.tensor(np.stack(factor_rows)).to(DEVICE)
    context_batch = torch.tensor(np.stack(context_rows)).to(DEVICE)

    model.eval()
    with torch.no_grad():
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                all_scores, all_attention = model(factor_batch, context_batch)
        else:
            all_scores, all_attention = model(factor_batch, context_batch)

    scores_np = all_scores.cpu().numpy()
    attention_np = all_attention.cpu().numpy()

    scores = {ticker: float(scores_np[i]) for i, ticker in enumerate(valid_tickers)}
    mean_attention = {
        factor: float(weight)
        for factor, weight in zip(FACTOR_COLS, attention_np.mean(axis=0))
    }
    return scores, mean_attention


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_train_tensors(feature_data, train_end=None, min_rows=20, verbose=True):
    cutoff = pd.Timestamp(train_end) if train_end is not None else pd.Timestamp(TRAIN_END)
    target_col = f"fwd_return_{FWD_HORIZON}d"

    # Push back by FWD_HORIZON business days so no forward-return target
    # uses prices from after the cutoff (prevents look-ahead bias).
    safe_cutoff = cutoff - pd.tseries.offsets.BDay(FWD_HORIZON)

    # Rolling window: use ROLLING_WINDOW trading days (~3 months) of recent
    # data for walk-forward retrains to keep the model adapted to current
    # regime while avoiding stale patterns.
    if train_end is not None:
        rolling_start = safe_cutoff - pd.tseries.offsets.BDay(ROLLING_WINDOW)
    else:
        rolling_start = pd.Timestamp(TRAIN_START)

    all_rows = []
    for ticker, df in feature_data.items():
        mask = (df.index >= rolling_start) & (df.index < safe_cutoff)
        ticker_subset = df[mask].dropna(subset=FACTOR_COLS + CONTEXT_COLS + [target_col])
        if len(ticker_subset) < min_rows:
            continue
        rolling_vol = (
            ticker_subset[target_col]
            .expanding(min_periods=20).std()
            .replace(0, np.nan).ffill()
            .fillna(ticker_subset[target_col].iloc[:20].std())
        )
        adj_returns = (ticker_subset[target_col] / rolling_vol).clip(-3, 3)
        for date, row in ticker_subset.iterrows():
            all_rows.append({
                "date": date,
                "factors": row[FACTOR_COLS].values.astype(np.float32),
                "context": row[CONTEXT_COLS].values.astype(np.float32),
                "target": float(adj_returns.loc[date]),
            })

    all_rows.sort(key=lambda sample: sample["date"])

    if len(all_rows) < 2:
        if verbose:
            print(f"WARNING: insufficient training data ({len(all_rows)} rows). "
                  "Returning empty tensors -- model will keep previous weights.")
        return [], []

    n_val = max(int(len(all_rows) * 0.15), 1)
    n_train = len(all_rows) - n_val

    train_targets = np.array([s["target"] for s in all_rows[:n_train]], dtype=np.float32)
    lower_bound, upper_bound = np.percentile(train_targets, 2), np.percentile(train_targets, 98)
    for sample in all_rows:
        sample["target"] = float(np.clip(sample["target"], lower_bound, upper_bound))

    if verbose:
        cutoff_date = all_rows[n_train]["date"]
        print(f"Training set : {n_train:,} samples")
        print(f"Val set      : {n_val:,} samples (dates >= {cutoff_date.date()})")

    def to_date_groups(rows):
        by_date = {}
        for sample in rows:
            by_date.setdefault(sample["date"], []).append(sample)
        groups = []
        for group_date in sorted(by_date.keys()):
            date_samples = by_date[group_date]
            f = torch.tensor(np.stack([s["factors"] for s in date_samples]))
            c = torch.tensor(np.stack([s["context"] for s in date_samples]))
            t = torch.tensor(np.array([s["target"] for s in date_samples], dtype=np.float32))
            if PIN_MEMORY:
                f, c, t = f.pin_memory(), c.pin_memory(), t.pin_memory()
            groups.append((
                f.to(DEVICE, non_blocking=True),
                c.to(DEVICE, non_blocking=True),
                t.to(DEVICE, non_blocking=True),
            ))
        return groups

    return to_date_groups(all_rows[:n_train]), to_date_groups(all_rows[n_train:])


def pearson_ic_loss(predictions, targets):
    """Negative Pearson IC (cross-sectional)."""
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    return -(pred_centered * target_centered).sum() / (
        pred_centered.norm() * target_centered.norm() + 1e-8
    )


# Keep old name as alias for backward compatibility
rank_ic_loss = pearson_ic_loss


def attention_entropy(weights):
    return -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()


def train_model(feature_data, train_end=None, verbose=True, k_dates=BATCH_SIZE,
                warm_start_state=None, progress_callback=None):
    """
    Train the fusion network using cross-sectional rank-IC loss.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with (epoch, train_ic, val_ic, total_epochs) after each epoch.
    """
    train_groups, val_groups = build_train_tensors(
        feature_data, train_end=train_end, verbose=verbose
    )

    if not train_groups:
        # Insufficient data -- return existing model unchanged
        if warm_start_state is not None:
            model = AdaptiveFusionNetwork().to(DEVICE)
            model.load_state_dict(warm_start_state)
            return model, [], []
        return AdaptiveFusionNetwork().to(DEVICE), [], []

    context_size = train_groups[0][1].shape[1]
    model = AdaptiveFusionNetwork(context_size).to(DEVICE)
    is_finetune = warm_start_state is not None
    if is_finetune:
        model.load_state_dict(warm_start_state)
    model = maybe_compile(model)

    ft_lr = LEARNING_RATE / 5 if is_finetune else LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=ft_lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10 if is_finetune else 20, factor=0.5, min_lr=1e-5
    )

    EARLY_STOP = 15 if is_finetune else 40
    train_ic_history, val_ic_history = [], []

    # For warm-start: evaluate initial model on val first; only accept improvements.
    if is_finetune and val_groups:
        model.eval()
        baseline_ics = []
        with torch.no_grad():
            for f_b, c_b, t_b in val_groups:
                if len(t_b) >= 4:
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        baseline_ics.append(
                            -rank_ic_loss(model(f_b, c_b)[0], t_b).item())
        baseline_val_ic = float(np.mean(baseline_ics)) if baseline_ics else 0.0
        best_val_loss = -baseline_val_ic
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        best_val_loss, best_state, patience_counter = float("inf"), None, 0

    if verbose:
        device_label = torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE
        print(f"Training on {device_label} (cross-sectional rank-IC) "
              f"for up to {TRAIN_EPOCHS} epochs ...")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        permutation = np.random.permutation(len(train_groups))
        train_ic_list = []

        for batch_start in range(0, len(permutation) - k_dates + 1, k_dates):
            optimizer.zero_grad()
            batch_losses = []
            batch_attentions = []
            for group_index in permutation[batch_start:batch_start + k_dates]:
                factor_batch = train_groups[group_index][0]
                context_batch = train_groups[group_index][1]
                target_batch = train_groups[group_index][2]
                if len(target_batch) < 4:
                    continue
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    preds, attn = model(factor_batch, context_batch)
                    date_loss = rank_ic_loss(preds, target_batch)
                batch_losses.append(date_loss)
                batch_attentions.append(attn)
                train_ic_list.append(-date_loss.item())
            if not batch_losses:
                continue
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                ic_loss = torch.stack(batch_losses).mean()
                all_attn = torch.cat(batch_attentions, dim=0)
                entropy_penalty = ENTROPY_LAMBDA * attention_entropy(all_attn)
                total_loss = ic_loss + entropy_penalty
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_ic_list = []
        with torch.no_grad():
            for factor_batch, context_batch, target_batch in val_groups:
                if len(target_batch) >= 4:
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        val_ic_list.append(
                            -rank_ic_loss(
                                model(factor_batch, context_batch)[0], target_batch
                            ).item()
                        )

        train_ic = float(np.mean(train_ic_list)) if train_ic_list else 0.0
        val_ic = float(np.mean(val_ic_list)) if val_ic_list else 0.0
        train_ic_history.append(-train_ic)
        val_ic_history.append(-val_ic)

        scheduler.step(-val_ic)

        if -val_ic < best_val_loss:
            best_val_loss, patience_counter = -val_ic, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1

        if progress_callback:
            progress_callback(epoch, train_ic, val_ic, TRAIN_EPOCHS)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | train IC={train_ic:.4f} | val IC={val_ic:.4f} "
                  f"| gap={train_ic - val_ic:+.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= EARLY_STOP:
            if verbose:
                print(f"  Early stop at epoch {epoch} (best val IC: {-best_val_loss:.4f})")
            break

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(best_state)
    raw_model.eval()
    train_ic_final = float(np.mean(train_ic_list)) if train_ic_list else 0.0
    val_ic_final   = -best_val_loss
    if verbose:
        print(f"  Training complete. Best val IC: {val_ic_final:.4f}")
        print(f"  Final gap  : {train_ic_final - val_ic_final:+.4f}  "
              f"{'(overfit)' if train_ic_final - val_ic_final > 0.10 else '(healthy)'}")
    return raw_model, train_ic_history, val_ic_history


def load_or_train(feature_data, force_retrain=False, progress_callback=None):
    """Load saved model or train a new one."""
    if not force_retrain and MODEL_PATH.exists():
        model = AdaptiveFusionNetwork(len(CONTEXT_COLS))
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        model.to(DEVICE).eval()
        return model, [], []
    model, train_hist, val_hist = train_model(
        feature_data, progress_callback=progress_callback
    )
    torch.save(model.state_dict(), MODEL_PATH)
    return model, train_hist, val_hist
