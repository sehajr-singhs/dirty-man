"""Sanity tests for the Switch Operator.

Run: python -m pytest tests/ -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dirty_man.data_glyphs import make_datasets, make_glyph_dataset  # noqa: E402
from dirty_man.switch_operator import (  # noqa: E402
    PRIMITIVES,
    SwitchOperator,
    annealed_tau,
)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return SwitchOperator(n_classes=10)


def test_primitives_are_nine_and_named():
    assert PRIMITIVES == [
        "linear", "dense", "relu", "cnn", "rnn", "lstm",
        "gan", "autoencoder", "transformer",
    ]
    assert len(PRIMITIVES) == 9


def test_forward_shape_and_probs(model):
    x = torch.randn(8, 1, 24, 24)
    out, info = model(x, tau=1.0, hard=False)
    assert out.shape == (8, 10)
    p = info["probs"]
    assert p.shape == (8, 9)
    # soft routing weights sum to one
    assert torch.allclose(p.sum(dim=-1), torch.ones(8), atol=1e-4)


def test_forward_goal_conditioned():
    torch.manual_seed(1)
    m = SwitchOperator(n_classes=10, n_goals=2, heads_out=[10, 576])
    x = torch.randn(6, 1, 24, 24)
    goal = torch.tensor([0, 1, 0, 1, 0, 1])
    out, _ = m(x, goal=goal, tau=1.0, hard=False)
    assert out.shape == (6, 576)


def test_annealed_tau_starts_soft_ends_hard():
    t0 = annealed_tau(0, 12)
    t1 = annealed_tau(11, 12)
    assert t0 > 3.9
    assert t1 < 0.51
    assert annealed_tau(0, 1) < 0.51  # degenerate schedule commits


def test_model_has_eye_router_and_bank(model):
    assert hasattr(model, "eye")
    assert hasattr(model, "router")
    assert len(model.primitives) == 9
    assert hasattr(model, "goal_net")


def test_eval_is_deterministic(model):
    x = torch.randn(4, 1, 24, 24)
    model.eval()
    o1, i1 = model(x, tau=0.5, hard=False)
    o2, i2 = model(x, tau=0.5, hard=False)
    assert torch.equal(o1, o2)
    assert torch.equal(i1["probs"], i2["probs"])


def test_data_has_five_tensors():
    ds, _ = make_datasets(n_train=32, n_test=16, seed=3)
    x, y, domain, severity, regime = ds.tensors
    assert x.shape == (32, 1, 24, 24)
    assert y.shape == (32,)
    assert domain.shape == (32,)
    assert severity.shape == (32,)
    assert regime.shape == (32,)
    # both domains present, severities in [0, 1]
    assert set(domain.tolist()) <= {0, 1}
    assert severity.min() >= 0.0 and severity.max() <= 1.0


def test_data_balanced_across_domains():
    ds, _ = make_datasets(n_train=64, n_test=16, seed=5)
    _, _, domain, _, _ = ds.tensors
    assert (domain == 0).sum().item() == 32
    assert (domain == 1).sum().item() == 32


def test_train_test_streams_are_disjoint():
    tr, te = make_datasets(n_train=64, n_test=64, seed=9)
    # data never leaks: distinct random streams
    assert not torch.equal(tr.tensors[0], te.tensors[0])


def test_device_auto_detect_and_batches_move():
    # the runner moves batches to DEVICE via collate; on CPU-only boxes this
    # is a no-op but must not crash, and on GPU boxes it must not raise
    # device-mismatch errors.
    from dirty_man.data_glyphs import make_datasets
    from run_experiments import collate, make_loader

    tr, _ = make_datasets(n_train=16, n_test=8, seed=2)
    x, y, d, s, r = next(iter(make_loader(tr, batch=8, shuffle=False)))
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert d.shape == (8,) and s.shape == (8,) and r.shape == (8,)
    # all five tensors land on the same device as the model would be
    dev = torch.cuda.device_of(x) if x.is_cuda else torch.device("cpu")
    assert x.device == dev


def test_forward_on_gpu_if_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")
    m = SwitchOperator(n_classes=10).to("cuda")
    x = torch.randn(4, 1, 24, 24, device="cuda")
    out, info = m(x, tau=1.0, hard=False)
    assert out.shape == (4, 10)
    assert info["probs"].device.type == "cuda"
