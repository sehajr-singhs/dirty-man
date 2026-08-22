import numpy as np
import torch

import sarcos_routing as sr


def test_speed_stratification_is_deterministic_and_covers_samples():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(32, 21)).astype(np.float32)
    y = rng.normal(size=(32, 7)).astype(np.float32)
    model = sr.StaticDynamics(path=("linear", "linear"))

    bins = sr.evaluate_by_speed(model, x, y, n_bins=4)

    assert sum(row["n"] for row in bins) == len(x)
    assert all(row["speed_lo"] <= row["speed_hi"] for row in bins)
    assert all(np.isfinite(row["mse"]) for row in bins)


def test_lens_profile_reports_speed_quartiles():
    torch.manual_seed(3)
    x = np.random.default_rng(3).normal(size=(32, 21)).astype(np.float32)
    y = np.zeros((32, 7), dtype=np.float32)
    model = sr.RoutedDynamics()

    profile = sr.lens_profile(model, x, y)

    assert len(profile["d1_shares_by_speed_quartile"]) == 4
    assert sum(row["n"] for row in profile["d1_shares_by_speed_quartile"]) == len(x)
    for row in profile["d1_shares_by_speed_quartile"]:
        assert abs(sum(row["shares"].values()) - 1.0) < 0.002
