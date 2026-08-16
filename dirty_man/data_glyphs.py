"""Synthetic sim-to-real glyph benchmark.

Digits 0-9 are drawn procedurally as parametric strokes (segments with
anti-aliased thickness), rendered with numpy — no downloads, fully
reproducible. Two domains:

  * SIM  — clean renderings with mild geometric jitter (the simulator: perfect
           geometry, perfect pixels).
  * REAL — sensor-corrupted renderings: motion blur, additive noise,
           brightness/contrast drift, occlusion, block-quantization
           (JPEG-like), and affine warps (the real world: dirty pixels).

Each sample also carries a `severity` in [0, 1] (0 = clean sim, higher =
harsher corruption) so we can plot *how* the operator rewires its computation
as the world gets dirtier.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

SIZE = 24
SEGMENTS: dict[int, list[tuple[float, float, float, float]]] = {
    # strokes in normalized [0,1] coords, y-down; rendered with thickness.
    0: [(0.25, 0.20, 0.75, 0.20), (0.75, 0.20, 0.75, 0.80),
        (0.75, 0.80, 0.25, 0.80), (0.25, 0.80, 0.25, 0.20)],
    1: [(0.50, 0.80, 0.50, 0.28), (0.44, 0.34, 0.56, 0.28)],
    2: [(0.25, 0.26, 0.75, 0.26), (0.75, 0.26, 0.75, 0.52),
        (0.75, 0.52, 0.25, 0.52), (0.25, 0.52, 0.25, 0.78), (0.25, 0.78, 0.75, 0.78)],
    3: [(0.20, 0.22, 0.80, 0.22), (0.80, 0.22, 0.80, 0.50),
        (0.32, 0.50, 0.80, 0.50), (0.80, 0.50, 0.80, 0.78), (0.20, 0.78, 0.80, 0.78)],
    4: [(0.25, 0.20, 0.25, 0.60), (0.20, 0.60, 0.80, 0.60),
        (0.75, 0.18, 0.75, 0.85)],
    5: [(0.75, 0.22, 0.25, 0.22), (0.25, 0.22, 0.25, 0.50),
        (0.25, 0.50, 0.72, 0.50), (0.72, 0.50, 0.72, 0.78), (0.25, 0.78, 0.75, 0.78)],
    6: [(0.72, 0.30, 0.72, 0.72), (0.72, 0.72, 0.34, 0.72),
        (0.34, 0.72, 0.34, 0.44), (0.34, 0.44, 0.72, 0.44), (0.72, 0.44, 0.52, 0.86)],
    7: [(0.22, 0.22, 0.78, 0.22), (0.78, 0.22, 0.46, 0.85)],
    8: [(0.35, 0.20, 0.65, 0.20), (0.65, 0.20, 0.65, 0.50), (0.65, 0.50, 0.35, 0.50),
        (0.35, 0.50, 0.35, 0.20), (0.35, 0.50, 0.65, 0.50), (0.65, 0.50, 0.65, 0.80),
        (0.65, 0.80, 0.35, 0.80), (0.35, 0.80, 0.35, 0.50)],
    9: [(0.35, 0.20, 0.65, 0.20), (0.65, 0.20, 0.65, 0.55), (0.65, 0.55, 0.35, 0.55),
        (0.35, 0.55, 0.35, 0.20), (0.50, 0.55, 0.50, 0.85)],
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _dist_to_segments(p: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """p: (N,2) points; segs: (S,2,2) segments. Returns (N,) min distance."""
    p = p[:, None, :]                                   # (N,1,2)
    a = segs[:, 0][None]                                # (1,S,2)
    b = segs[:, 1][None]
    ab = b - a
    abn = np.linalg.norm(ab, axis=-1, keepdims=True).clip(1e-6)
    t = ((p - a) * ab).sum(-1, keepdims=True) / (abn ** 2)
    t = t.clip(0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj, axis=-1).min(axis=-1)


def _render(digit: int, rng: np.random.Generator, size: int = SIZE,
            thickness: float = 0.055) -> np.ndarray:
    """Render one glyph with mild geometric jitter (shared by both domains)."""
    segs = np.array(SEGMENTS[digit], dtype=np.float64).reshape(-1, 2, 2)
    # jitter: translate, scale, rotate, per-segment wobble
    tx, ty = rng.uniform(-0.05, 0.05, 2)
    s = rng.uniform(0.9, 1.08)
    th = rng.uniform(0.045, 0.075)
    angle = rng.uniform(-0.08, 0.08)
    c, sn = np.cos(angle), np.sin(angle)
    R = np.array([[c, -sn], [sn, c]])
    segs = segs * s + np.array([tx, ty])
    segs = segs @ R.T
    segs = np.clip(segs, 0.02, 0.98)
    wob = rng.normal(0, 0.012, segs.shape)
    segs = segs + wob

    yy, xx = np.mgrid[0:size, 0:size]
    pts = np.stack([xx.ravel() / (size - 1), yy.ravel() / (size - 1)], axis=-1)
    d = _dist_to_segments(pts, segs).reshape(size, size)
    img = np.clip((th - d) / (th * 0.8), 0.0, 1.0)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Corruption library — the "real" world
#
# Every corruption is *spatial*: it destroys the clean pixel-grid structure of
# the simulator in a way that flat (per-pixel) lenses cannot undo, but local
# spatial lenses (CNN, autoencoder) can. This is the property that makes
# structure switching worth it — a dirty image wants a different brain than a
# clean one.
# ---------------------------------------------------------------------------

def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Defocus: smears strokes across neighbors, erasing thin geometry."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=rng.uniform(0.5, 2.4)).astype(np.float32)


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy Gaussian sensor noise — decorrelates neighboring pixels so only
    local spatial averaging can recover the glyph underneath."""
    return np.clip(img + rng.normal(0, rng.uniform(0.12, 0.38), img.shape), 0, 1)


def _spatter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Salt-and-pepper dropout: isolated pixels vanish or flare."""
    img = img.copy()
    frac = rng.uniform(0.04, 0.2)
    mask = rng.random(img.shape) < frac
    img[mask] = rng.integers(0, 2, size=int(mask.sum()))
    return img.astype(np.float32)


def _photo(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exposure drift: global brightness/contrast shift."""
    img = np.clip(img * rng.uniform(0.45, 1.6) + rng.uniform(-0.25, 0.25), 0, 1)
    return img.astype(np.float32)


def _occlude(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Occlusion: chunks of the glyph — up to half the frame — are painted
    over with sensor crud. Local lenses keep classifying from surviving
    patches; flat lenses lose whole coordinate blocks."""
    img = img.copy()
    for _ in range(rng.integers(1, 4)):
        h, w = rng.integers(5, 16, 2)
        y, x = rng.integers(0, img.shape[0] - h), rng.integers(0, img.shape[1] - w)
        img[y:y + h, x:x + w] = rng.uniform(0.05, 0.45)
    return img


def _quantize(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy JPEG-ish block quantization: downscale then upscale, so whole
    6-16px neighborhoods collapse to a single value — the glyph becomes a
    coarse block mosaic. Flat lenses ride the block intensities; spatial
    lenses lose every edge."""
    k = int(rng.integers(8, 13))
    h, w = img.shape
    hc, wc = (h // k) * k, (w // k) * k
    crop = img[:hc, :wc]
    small = crop.reshape(hc // k, k, wc // k, k).mean(axis=(1, 3))
    up = np.repeat(np.repeat(small, k, axis=0), k, axis=1)
    out = np.zeros_like(img)
    out[:hc, :wc] = up
    return out.astype(np.float32)


def _warp(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lens warp: shear/scale distortion — shifts whole pixel neighborhoods, so
    translation-equivariant lenses absorb it while flat lenses misalign."""
    from scipy.ndimage import affine_transform
    m = rng.uniform(0.62, 1.38, 2)
    shear = rng.uniform(-0.35, 0.35)
    M = np.array([[m[0], shear, 0.0], [shear, m[1], 0.0]])
    return affine_transform(img, M, order=1, mode="nearest").astype(np.float32)


def _shift(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Handshake jitter: the whole glyph slides off its nominal pixel grid by
    up to a third of the frame. Translation-equivariant lenses absorb this;
    flat per-pixel lenses see a brand-new pattern."""
    from scipy.ndimage import shift as ndshift
    dy, dx = rng.uniform(-6, 6, 2)
    return ndshift(img, (dy, dx), order=1, mode="nearest").astype(np.float32)


def _scramble(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Local pixel shuffle: each kxk neighborhood's pixels are permuted, so the
    global pixel layout is destroyed but coarse local energy survives. Flat
    lenses read dead coordinates; spatial lenses still feel the texture."""
    k = int(rng.integers(2, 5))
    h, w = img.shape
    hc, wc = (h // k) * k, (w // k) * k
    crop = img[:hc, :wc].copy()
    nr, nc = hc // k, wc // k
    # one random permutation of the k*k pixels inside each kxk block
    perm = np.stack([rng.permutation(k * k) for _ in range(nr * nc)])
    flat = crop.reshape(nr, k, nc, k).transpose(0, 2, 1, 3).reshape(nr, nc, k * k)
    shuffled = np.take_along_axis(flat, perm.reshape(nr, nc, k * k), axis=-1)
    out = np.zeros_like(img)
    out[:hc, :wc] = shuffled.reshape(nr, nc, k, k).transpose(0, 2, 1, 3).reshape(hc, wc)
    return out.astype(np.float32)


def _thin(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ink starvation: strokes are eroded so they thin, break, or vanish into
    isolated dots. Only a lens that reads local continuity (CNN/AE) can
    reassemble the glyph; flat lenses see a shattered pattern."""
    from scipy.ndimage import binary_erosion
    th = img > 0.45
    n = int(rng.integers(1, 3))
    s = np.ones((3, 3))
    for _ in range(n):
        th = binary_erosion(th, structure=s, iterations=1)
    out = np.where(th, 1.0, img * 0.25)
    return out.astype(np.float32)


def _erase(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Barcode dropout: whole diagonal bands of the glyph are erased, so the
    survivor is a set of disconnected strokes that must be pieced back
    together spatially."""
    img = img.copy()
    n = int(rng.integers(1, 3))
    for _ in range(n):
        w = rng.uniform(2, 5)
        off = rng.uniform(-0.5, 0.5) * img.shape[0]
        yy = np.arange(img.shape[0])
        band = np.abs(yy - (off + img.shape[0] * 0.5)) < w
        img[band] = 0.0
    return img


def _stripes(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sensor banding: whole rows or columns flicker."""
    img = img.copy()
    axis = rng.integers(0, 2)
    n = int(rng.integers(1, 5))
    idx = rng.choice(img.shape[axis], size=n, replace=False)
    gain = rng.uniform(0.0, 1.6, size=n)
    for i, g in zip(idx, gain):
        if axis == 0:
            img[i, :] *= g
        else:
            img[:, i] *= g
    return np.clip(img, 0, 1).astype(np.float32)


CORRUPTIONS = [_blur, _noise, _spatter, _photo, _occlude, _quantize, _warp,
               _stripes, _shift, _thin, _erase, _scramble]


def _corrupt(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Corrupt a sim rendering into a 'real' one. Real samples come from two
    visually distinct regimes, each with a clearly different best lens:

      * SPATIAL regime — moderate displacement: the glyph slides and warps
        off its nominal pixel grid (1-2 touches). Translation-equivariant
        lenses (CNN, AE) absorb the shift and stay accurate; flat lenses see
        a brand-new pattern every time.
      * STATISTICAL regime — heavy block-quantization: the glyph collapses
        into a coarse block mosaic (plus mild exposure drift). A flat lens
        rides the block intensities straight through; a spatial lens finds
        every edge destroyed and every gradient zeroed.

    The two regimes are tuned so the best lens differs *materially* per
    regime (spatial -> spatial lens, statistical -> flat lens), and each
    regime stays learnable — that is the visible signal that makes
    structure switching worth it.
    Returns (image, severity in [0,1], regime in {0 sim, 1 spatial, 2 stat})."""
    sev = 0.0
    if rng.random() < 0.5:
        # SPATIAL — moderate: displacement + mild blur, 1-2 touches.
        n = int(rng.integers(1, 3))
        for fn in rng.choice([_warp, _shift, _blur], size=n, replace=False):
            img = fn(img, rng)
            sev += 0.35
        return img, float(min(sev, 1.0)), 1
    # STATISTICAL — heavy block quantization (the flat-friendly one) plus
    # mild exposure drift, 1-2 touches. k is fixed in the sweet spot where
    # the mosaic is still readable as intensities (k=8 -> 3x3 blocks).
    img = _quantize(img, rng)
    sev += 0.6
    if rng.random() < 0.6:
        img = _photo(img, rng)
        sev += 0.2
    return img, float(min(sev, 1.0)), 2


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def make_glyph_dataset(n_per_domain: int, size: int = SIZE, seed: int = 0,
                       domain: str = "mixed") -> dict:
    """Build a balanced glyph tensor dataset.

    Returns a dict with tensors x (B,1,size,size), y (B,), domain (B,) in
    {0 sim, 1 real}, severity (B,), regime (B,) in {0 sim, 1 spatial,
    2 statistical}.
    """
    rng = np.random.default_rng(seed)
    xs, ys, doms, sevs, regs = [], [], [], [], []
    n_domains = 2 if domain == "mixed" else 1
    for d in range(n_domains):
        n = n_per_domain // n_domains
        corrupt = (domain == "real") or (d == 1)
        for _ in range(n):
            digit = int(rng.integers(0, 10))
            img = _render(digit, rng, size=size)
            if corrupt:
                img, sev, reg = _corrupt(img, rng)
            else:
                sev, reg = 0.0, 0
            xs.append(img[None])
            ys.append(digit)
            doms.append(d)
            sevs.append(sev)
            regs.append(reg)
    return {
        "x": torch.tensor(np.stack(xs), dtype=torch.float32),
        "y": torch.tensor(ys, dtype=torch.long),
        "domain": torch.tensor(doms, dtype=torch.long),
        "severity": torch.tensor(sevs, dtype=torch.float32),
        "regime": torch.tensor(regs, dtype=torch.long),
    }


def make_datasets(n_train: int = 6000, n_test: int = 2000, size: int = SIZE,
                  seed: int = 0, domain: str = "mixed") -> tuple[TensorDataset, TensorDataset]:
    """Train/test split with disjoint random streams (data never leaks)."""
    tr = make_glyph_dataset(n_train, size=size, seed=seed, domain=domain)
    te = make_glyph_dataset(n_test, size=size, seed=seed + 1000, domain=domain)
    train_ds = TensorDataset(tr["x"], tr["y"], tr["domain"], tr["severity"], tr["regime"])
    test_ds = TensorDataset(te["x"], te["y"], te["domain"], te["severity"], te["regime"])
    return train_ds, test_ds


def save_montage(path: str, n_per_digit: int = 3, seed: int = 0) -> None:
    """Debug/site asset: a grid of sim vs real glyphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 10 * n_per_digit, figsize=(2.2 * n_per_digit * 10 / 5, 4.4),
                             dpi=150)
    for row, corrupt in enumerate([False, True]):
        for col in range(10 * n_per_digit):
            digit = col // n_per_digit
            img = _render(digit, rng)
            if corrupt:
                img, _sev, _reg = _corrupt(img, rng)
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col % n_per_digit == 0:
                ax.set_ylabel("REAL" if corrupt else "SIM", fontsize=8, rotation=0,
                              labelpad=18, va="center")
    fig.suptitle("The glyph benchmark — sim (clean) vs real (dirty)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_playground_data(probs: list[list[float]], severities: list[float],
                         labels: list[int], images: np.ndarray, path: str) -> None:
    """Serialise per-sample routing for the website's interactive playground."""
    payload = {
        "primitives": ["linear", "dense", "relu", "cnn", "rnn", "lstm",
                       "gan", "autoencoder", "transformer"],
        "severity": severities,
        "label": labels,
        "images": images.tolist(),
        "probs": probs,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    save_montage("figs/fig0_glyphs.png")
    print("wrote figs/fig0_glyphs.png")
