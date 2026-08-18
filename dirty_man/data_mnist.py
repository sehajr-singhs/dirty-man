"""Real handwritten digits (MNIST) as the genuinely-real test domain.

The glyph benchmark's "real" domain is still *synthetic* (procedural renders
with hand-tuned corruptions). This module adds the real thing: MNIST
handwritten digits, downloaded once and cached locally. Zero overlap with the
procedural renderer — the operator must transfer to digits it has never seen
in any form.

The dataset is emitted in the exact (x, y, domain, severity, regime) tensor
format used everywhere else, so every existing training/eval function works
unchanged:

    x        (B, 1, 24, 24)  float32, resized from 28x28, roughly [0, 1]
    y        (B,)            long, digit label
    domain   (B,)            long, all 1 (real)
    severity (B,)            float32, all 0.5 (handwriting is "dirty" but not
                             adversarially corrupted)
    regime   (B,)            long, all 1 (spatial: real handwriting has
                             natural stroke variation)

The regime choice is deliberate: real handwriting is a *spatial* perturbation
of the clean-stroke ideal (natural wobble, pressure, slant), so a spatial lens
(CNN / autoencoder) should be the right tool — the router's job on real digits
is to *identify that feature* from visual cues it learned only on synthetic
glyphs.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

SIZE = 24          # the model's native resolution; MNIST is resized to match
REAL_REGIME = 1    # spatial
REAL_SEVERITY = 0.5


def _load_mnist(root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and return MNIST as raw numpy arrays.

    The download is retried with exponential backoff: sandboxed runners
    (Kaggle, CI) often fail the first S3 lookup with a transient DNS error
    (`Temporary failure in name resolution`) even when internet is enabled.
    """
    from torchvision import datasets
    last: Exception | None = None
    for attempt in range(5):
        try:
            ds = datasets.MNIST(root=root, train=train, download=True)
            x = ds.data.numpy().astype(np.float32)  # (N, 28, 28) in [0, 255]
            y = ds.targets.numpy().astype(np.int64)
            return x, y
        except Exception as exc:                     # noqa: BLE001 - retry any dl failure
            last = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"MNIST download failed after 5 attempts: {last}")


def _resize(x: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Box-downscale MNIST's 28x28 to `size` using torch's interpolate
    (bilinear). Deterministic; no random cropping."""
    t = torch.from_numpy(x).unsqueeze(1)            # (N, 1, 28, 28)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return t.squeeze(1).numpy()


def make_mnist_dataset(root: str = os.path.join(os.path.dirname(
                           os.path.abspath(__file__)), "..", "data", "mnist"),
                       n: int | None = None, train: bool = True,
                       size: int = SIZE, seed: int = 0) -> dict:
    """Load real MNIST as a dict of tensors in the standard schema.

    `n` caps the number of samples (for tiny experiments); the full 60k/10k
    split is used when n is None.
    """
    os.makedirs(root, exist_ok=True)
    x, y = _load_mnist(root, train=train)
    if n is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))[:n]
        x, y = x[idx], y[idx]
    x = _resize(x, size=size)
    x = x / 255.0                                   # [0, 1]
    n = x.shape[0]
    return {
        "x": torch.tensor(x, dtype=torch.float32).unsqueeze(1),
        "y": torch.tensor(y, dtype=torch.long),
        "domain": torch.ones(n, dtype=torch.long),
        "severity": torch.full((n,), REAL_SEVERITY, dtype=torch.float32),
        "regime": torch.full((n,), REAL_REGIME, dtype=torch.long),
    }


def make_mnist_datasets(n_train: int | None = None, n_test: int | None = None,
                        root: str | None = None, size: int = SIZE) -> tuple[TensorDataset, TensorDataset]:
    """Train/test TensorDatasets in the standard 5-tensor schema."""
    root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "mnist")
    tr = make_mnist_dataset(root=root, n=n_train, train=True, size=size)
    te = make_mnist_dataset(root=root, n=n_test, train=False, size=size)
    train_ds = TensorDataset(tr["x"], tr["y"], tr["domain"], tr["severity"], tr["regime"])
    test_ds = TensorDataset(te["x"], te["y"], te["domain"], te["severity"], te["regime"])
    return train_ds, test_ds


if __name__ == "__main__":
    tr, te = make_mnist_datasets()
    print(f"train {len(tr)}  test {len(te)}")
    x = tr.tensors[0]
    print("x", tuple(x.shape), f"range [{float(x.min()):.2f}, {float(x.max()):.2f}]")
    print("domain", set(tr.tensors[2].tolist()), "regime", set(tr.tensors[4].tolist()))
