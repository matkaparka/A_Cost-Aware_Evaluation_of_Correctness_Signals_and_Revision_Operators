"""Exp 5 answer loss (src/ttt/update.answer_loss): values and gradients, including confident tokens (P2 fix)."""
import math

import pytest

torch = pytest.importorskip("torch")

from src.ttt.update import answer_loss  # noqa: E402


def test_nll_matches_log_softmax():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
    t = torch.tensor([0, 2])
    ref = -torch.log_softmax(logits, -1)[[0, 1], [0, 2]].mean()
    assert torch.allclose(answer_loss(logits, t, True), ref)


def test_unlikelihood_value():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    p = torch.softmax(logits, -1)[0, 0].item()
    assert answer_loss(logits, torch.tensor([0]), False).item() == pytest.approx(-math.log(1 - p), rel=1e-5)


def test_unlikelihood_gradient_survives_confident_tokens():
    # p(target) = 1 - 1e-9 in fp32: the old clamp(max=-1e-6) form had exactly zero gradient here.
    logits = torch.tensor([[30.0, 9.3, 9.3]], requires_grad=True)
    assert torch.softmax(logits, -1)[0, 0].item() > 1 - 1e-6
    loss = answer_loss(logits, torch.tensor([0]), False)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 13.8
    assert logits.grad.abs().sum().item() > 0.5
    assert logits.grad[0, 0].item() > 0          # descending the loss lowers the target logit


def test_old_form_was_flat():
    logits = torch.tensor([[30.0, 9.3, 9.3]], requires_grad=True)
    logp = torch.log_softmax(logits, -1)[0, 0]
    (-torch.log1p(-torch.exp(logp.clamp(max=-1e-6)))).backward()
    assert logits.grad.abs().sum().item() == 0.0
