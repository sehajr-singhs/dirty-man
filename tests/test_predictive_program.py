import torch

from predictive_program import OPS, PredictiveProgram, balanced_assignments


def test_balanced_assignments_cover_batch_without_collapse():
    cost = torch.tensor([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    labels = balanced_assignments(cost)
    counts = torch.bincount(labels, minlength=len(OPS))
    assert labels.shape == (6,)
    assert int(counts.sum()) == 6
    assert counts.tolist() == [2, 2, 2]


def test_predictive_program_reports_balanced_counterfactual_targets():
    torch.manual_seed(4)
    model = PredictiveProgram(dim=8)
    view_a = torch.randn(12, 1, 16, 16)
    view_b = torch.randn(12, 1, 16, 16)
    result = model(view_a, view_b, tau=1.0, hard=False)
    counts = result["pseudo_utilization"] * view_a.size(0)
    assert torch.all(counts >= 3.0 - 1e-5)
    assert torch.isfinite(result["loss"])
    assert result["counterfactual"].shape == (12, len(OPS))
