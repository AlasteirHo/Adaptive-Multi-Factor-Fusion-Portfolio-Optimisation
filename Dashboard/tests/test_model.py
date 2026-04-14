"""Unit tests for backend.model -- AdaptiveFusionNetwork and training utilities."""

import numpy as np
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.model import AdaptiveFusionNetwork, pearson_ic_loss, rank_ic_loss, attention_entropy


# ── AdaptiveFusionNetwork ────────────────────────────────────────────

class TestAdaptiveFusionNetwork:
    def test_output_shapes(self):
        model = AdaptiveFusionNetwork(context_size=10)
        factors = torch.randn(20, 8)   # 20 tickers, 8 factors
        context = torch.randn(20, 10)  # 20 tickers, 10 context features
        scores, attn = model(factors, context)
        assert scores.shape == (20,)
        assert attn.shape == (20, 8)

    def test_attention_sums_to_one(self):
        model = AdaptiveFusionNetwork(context_size=10)
        factors = torch.randn(5, 8)
        context = torch.randn(5, 10)
        _, attn = model(factors, context)
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_attention_non_negative(self):
        model = AdaptiveFusionNetwork(context_size=10)
        factors = torch.randn(5, 8)
        context = torch.randn(5, 10)
        _, attn = model(factors, context)
        assert (attn >= 0).all()

    def test_deterministic_with_same_input(self):
        model = AdaptiveFusionNetwork(context_size=10)
        model.eval()
        factors = torch.randn(3, 8)
        context = torch.randn(3, 10)
        s1, a1 = model(factors, context)
        s2, a2 = model(factors, context)
        torch.testing.assert_close(s1, s2)
        torch.testing.assert_close(a1, a2)

    def test_composite_score_is_weighted_sum(self):
        model = AdaptiveFusionNetwork(context_size=10)
        model.eval()
        factors = torch.randn(3, 8)
        context = torch.randn(3, 10)
        scores, attn = model(factors, context)
        expected = (attn * factors).sum(dim=-1)
        torch.testing.assert_close(scores, expected, atol=1e-5, rtol=1e-5)

    def test_different_context_produces_different_weights(self):
        model = AdaptiveFusionNetwork(context_size=10)
        model.eval()
        factors = torch.randn(1, 8)
        ctx1 = torch.zeros(1, 10)
        ctx2 = torch.ones(1, 10) * 5.0
        _, a1 = model(factors, ctx1)
        _, a2 = model(factors, ctx2)
        assert not torch.allclose(a1, a2)


# ── Pearson IC Loss ──────────────────────────────────────────────────

class TestPearsonICLoss:
    def test_perfect_correlation_gives_minus_one(self):
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = pearson_ic_loss(preds, targets)
        assert abs(loss.item() + 1.0) < 1e-5  # -IC = -1.0

    def test_perfect_negative_gives_plus_one(self):
        preds = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = pearson_ic_loss(preds, targets)
        assert abs(loss.item() - 1.0) < 1e-5  # -IC = +1.0

    def test_zero_correlation(self):
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, -1.0, 1.0, -1.0])
        loss = pearson_ic_loss(preds, targets)
        assert abs(loss.item()) < 0.5  # near zero

    def test_alias_works(self):
        # rank_ic_loss should be same function
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        assert pearson_ic_loss(preds, targets) == rank_ic_loss(preds, targets)


# ── Attention Entropy ────────────────────────────────────────────────

class TestAttentionEntropy:
    def test_uniform_gives_max_entropy(self):
        uniform = torch.ones(1, 8) / 8
        ent = attention_entropy(uniform)
        # Entropy = -(w * log(w)).sum() = log(8) for uniform
        assert abs(ent.item() - np.log(8)) < 1e-4

    def test_concentrated_gives_low_entropy(self):
        concentrated = torch.zeros(1, 8) + 1e-8
        concentrated[0, 0] = 1.0
        ent = attention_entropy(concentrated)
        assert ent.item() < 0.1  # near zero (most concentrated)

    def test_uniform_higher_than_concentrated(self):
        uniform = torch.ones(1, 8) / 8
        concentrated = torch.zeros(1, 8) + 1e-8
        concentrated[0, 0] = 1.0
        assert attention_entropy(uniform).item() > attention_entropy(concentrated).item()
