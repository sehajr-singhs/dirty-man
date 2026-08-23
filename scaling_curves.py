"""Scaling curves: Dirty Man advantage grows with problem heterogeneity.

The key NMI-level claim: non-static computation pays when heterogeneity
exceeds the cost of learning the policy. We test this by varying the
number of corruption types (2, 3, 4, 5) and measuring:
  - Static CNN accuracy (fixed architecture, must handle all corruptions)
  - Dirty Man accuracy (feature-conditioned routing)
  - Routing policy diversity (entropy of per-corruption routing)
  - Gap between Dirty Man and static baseline

If the gap grows with more corruption types, it supports the scaling
theorem (Theorem 5): a single map's error is bounded below by a constant
independent of the number of laws, while routing's error decays with
router accuracy.

Output: results/scaling_curves.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Corruptions (same as corruption_routing_benchmark.py)
# ---------------------------------------------------------------------------

def gaussian_noise(x, severity=0.3):
    return x + severity * torch.randn_like(x)

def salt_and_pepper(x, severity=0.15):
    mask = torch.rand_like(x) < severity
    noise = torch.rand_like(x) * 2 - 1
    return torch.where(mask, noise, x)

def rotate(x, severity=30.0):
    angle = (torch.rand(1).item() * 2 - 1) * severity
    theta = torch.tensor([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0],
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(x.unsqueeze(0), grid, align_corners=False,
                         padding_mode='border').squeeze(0)

def block_occlusion(x, severity=8):
    out = x.clone()
    _, h, w = x.shape
    s = int(severity)
    for _ in range(3):
        y0 = torch.randint(0, h - s, (1,)).item()
        x0 = torch.randint(0, w - s, (1,)).item()
        out[:, y0:y0 + s, x0:x0 + s] = 0.0
    return out

def contrast_change(x, severity=0.4):
    """Multiplicative contrast change — favors robust linear processing."""
    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * severity
    return x * factor

ALL_CORRUPTIONS = [
    ("clean", lambda x: x),
    ("gaussian", lambda x: gaussian_noise(x, 0.3)),
    ("saltpepper", lambda x: salt_and_pepper(x, 0.15)),
    ("rotation", lambda x: rotate(x, 30.0)),
    ("occlusion", lambda x: block_occlusion(x, 8)),
    ("contrast", lambda x: contrast_change(x, 0.4)),
]


class CorruptionDataset(Dataset):
    def __init__(self, base, corruptions, n=None, seed=0):
        self.base = base
        self.corruptions = corruptions
        g = torch.Generator().manual_seed(seed)
        n = n or len(base)
        n_each = n // len(corruptions)
        assignments = []
        for ci in range(len(corruptions)):
            assignments.extend([ci] * n_each)
        while len(assignments) < n:
            assignments.append(len(assignments) % len(corruptions))
        perm = torch.randperm(n, generator=g)
        self.assignments = [assignments[p.item()] for p in perm][:n]

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, idx):
        ci = self.assignments[idx]
        sample_idx = idx % len(self.base)
        x, y = self.base[sample_idx]
        name, fn = self.corruptions[ci]
        return fn(x), y, ci


def load_data(n_train=6000, n_test=1500, n_corruptions=4, seed=0):
    from torchvision import datasets, transforms
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    base_train = datasets.FashionMNIST("data_fm", train=True, download=False, transform=tr)
    base_test = datasets.FashionMNIST("data_fm", train=False, download=False, transform=tr)

    corruptions = ALL_CORRUPTIONS[:n_corruptions]

    g = torch.Generator().manual_seed(seed)
    te_idx = torch.randperm(len(base_test), generator=g)[:n_test].tolist()
    per_corruption_test = {}
    for ci, (name, fn) in enumerate(corruptions):
        xs, ys = [], []
        for idx in te_idx:
            x, y = base_test[idx]
            xs.append(fn(x))
            ys.append(y)
        per_corruption_test[name] = TensorDataset(
            torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    train_ds = CorruptionDataset(base_train, corruptions, n_train, seed)
    return train_ds, per_corruption_test, corruptions


# ---------------------------------------------------------------------------
# Models (compact versions for scaling experiment)
# ---------------------------------------------------------------------------

class StaticCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, n_classes),
        )
    def forward(self, x):
        return self.net(x)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class DirtyManScaling(nn.Module):
    """Compact Dirty Man for scaling experiments."""
    LENS_WIDTH = 64

    def __init__(self, n_classes=10, n_lenses=4):
        super().__init__()
        self.n_lenses = n_lenses
        # Eye
        self.eye = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(24 * 7 * 7, 64), nn.ReLU(),
        )
        self.corruption_head = nn.Linear(64, n_lenses)
        # Lenses: linear, relu, cnn, gated
        self.lens0 = nn.Sequential(nn.Flatten(), nn.Linear(28*28, self.LENS_WIDTH))
        self.lens1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                    nn.Linear(8*4*4, self.LENS_WIDTH))
        self.lens2 = nn.Sequential(nn.Conv2d(1, 12, 5, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(2), nn.Conv2d(12, 12, 3, padding=1),
                                    nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                                    nn.Linear(12*4*4, self.LENS_WIDTH))
        self.lens3_gate = nn.Sequential(nn.Flatten(), nn.Linear(28*28, self.LENS_WIDTH))
        self.lens3_cell = nn.Sequential(nn.Flatten(), nn.Linear(28*28, self.LENS_WIDTH))
        self.router = nn.Linear(64, n_lenses)
        self.head = nn.Linear(self.LENS_WIDTH, n_classes)

    def forward(self, x, tau=1.0, hard=False, record=None):
        cues = self.eye(x)
        logits = self.router(cues)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1) if self.training \
                else F.softmax(logits / max(tau, 1e-3), dim=-1)
        all_lens = [self.lens0(x), self.lens1(x), self.lens2(x),
                     self.lens3_gate(x).sigmoid() * self.lens3_cell(x).tanh()]
        lens_outs = all_lens[:self.n_lenses]
        mixture = sum(probs[:, i].unsqueeze(-1) * out for i, out in enumerate(lens_outs))
        out = self.head(mixture)
        if record is not None:
            record["probs"] = probs.detach().cpu()
            record["corr_logits"] = self.corruption_head(cues).detach().cpu()
        return out

    def forward_corruption(self, x):
        return self.corruption_head(self.eye(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def anneal_tau(epoch, total_epochs, start=2.0, end=0.3):
    if total_epochs <= 1:
        return end
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * epoch / (total_epochs - 1)))


def train_and_eval(model, train_ds, per_corruption_test, corruptions,
                   epochs, model_type, batch, device, seed):
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_corruptions = len(corruptions)

    # Dirty Man warm-up
    if model_type == "dirtyman":
        warm_epochs = max(3, epochs // 3)
        corr_opt = torch.optim.AdamW(
            list(model.eye.parameters()) + list(model.corruption_head.parameters()),
            lr=1e-3, weight_decay=1e-4)
        for ep in range(warm_epochs):
            model.train()
            for x, y, ci in loader:
                x, ci = x.to(device), ci.to(device)
                corr_opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.forward_corruption(x), ci)
                loss.backward()
                corr_opt.step()
        # Initialize router from corruption head
        with torch.no_grad():
            src = model.corruption_head.weight.data
            for i in range(min(n_corruptions, model.n_lenses)):
                model.router.weight.data[i] += src[i].clone() * 0.5
            model.router.bias.data.zero_()

    # Main training
    for epoch in range(epochs):
        model.train()
        tau = anneal_tau(epoch, epochs) if model_type == "dirtyman" else 1.0
        for x, y, ci in loader:
            x, y, ci = x.to(device), y.to(device), ci.to(device)
            opt.zero_grad(set_to_none=True)

            if model_type == "dirtyman":
                out = model(x, tau=tau, hard=False)
                task_loss = F.cross_entropy(out, y)
                probs = F.softmax(model.router(model.eye(x)), dim=-1)
                frac = probs.mean(0)
                target = torch.ones(model.n_lenses, device=device) / model.n_lenses
                bal = ((frac - target) ** 2).sum()
                # Pairwise diversity
                corr_means = []
                for c in range(n_corruptions):
                    mask = (ci == c)
                    if mask.sum() > 0:
                        corr_means.append(probs[mask].mean(0))
                div = torch.zeros((), device=device)
                for i in range(len(corr_means)):
                    for j in range(i+1, len(corr_means)):
                        div = div + F.cosine_similarity(
                            corr_means[i].unsqueeze(0), corr_means[j].unsqueeze(0))
                loss = task_loss + 0.3 * bal + 0.1 * div
            else:
                out = model(x)
                loss = F.cross_entropy(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Evaluate
    model.eval()
    results = {"per_corruption": {}, "routing_policy": {}}
    with torch.no_grad():
        for cname, cdata in per_corruption_test.items():
            dl = DataLoader(cdata, batch_size=batch, num_workers=0)
            correct = total = 0
            all_probs = []
            for cx, cy in dl:
                cx, cy = cx.to(device), cy.to(device)
                if model_type == "dirtyman":
                    rec = {}
                    out = model(cx, tau=0.1, hard=True, record=rec)
                    all_probs.append(rec["probs"])
                else:
                    out = model(cx)
                _, pred = out.max(1)
                correct += (pred == cy).sum().item()
                total += cx.size(0)
            results["per_corruption"][cname] = round(correct / max(total, 1), 4)
            if all_probs:
                avg = torch.cat(all_probs, dim=0).mean(0)
                results["routing_policy"][cname] = {
                    f"lens_{i}": round(float(avg[i]), 4)
                    for i in range(model.n_lenses)
                }

    mean_acc = sum(results["per_corruption"].values()) / max(len(results["per_corruption"]), 1)
    results["mean_acc"] = round(mean_acc, 4)
    results["params"] = model.n_params()

    # Compute routing diversity (entropy of mean routing)
    if results["routing_policy"]:
        all_routing = torch.zeros(model.n_lenses)
        for cname in results["routing_policy"]:
            for i in range(model.n_lenses):
                all_routing[i] += results["routing_policy"][cname][f"lens_{i}"]
        all_routing /= len(results["routing_policy"])
        entropy = -(all_routing * (all_routing + 1e-8).log()).sum().item()
        results["routing_entropy"] = round(entropy, 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_train, args.n_test, args.epochs, args.batch = 2000, 500, 3, 64
    device = args.device

    print("=== Scaling Curves: Heterogeneity vs. Routing Advantage ===", flush=True)

    all_results = {"experiment": "scaling_curves", "config": vars(args),
                   "n_corruption_range": [], "results": {}}

    for n_corr in [2, 3, 4, 5]:
        print(f"\n--- {n_corr} corruption types ---", flush=True)
        train_ds, pct, corruptions = load_data(
            args.n_train, args.n_test, n_corr, args.seed)

        # Static CNN
        t0 = time.time()
        m_cnn = StaticCNN().to(device)
        r_cnn = train_and_eval(m_cnn, train_ds, pct, corruptions,
                                args.epochs, "static", args.batch, device, args.seed)
        t_cnn = round(time.time() - t0, 1)
        print(f"  Static CNN: {r_cnn['mean_acc']:.3f} ({t_cnn:.0f}s)", flush=True)

        # Dirty Man
        t0 = time.time()
        n_lenses = min(n_corr, 4)  # max 4 lenses
        m_dm = DirtyManScaling(n_lenses=n_lenses).to(device)
        r_dm = train_and_eval(m_dm, train_ds, pct, corruptions,
                               args.epochs, "dirtyman", args.batch, device, args.seed)
        t_dm = round(time.time() - t0, 1)
        print(f"  Dirty Man:  {r_dm['mean_acc']:.3f} ({t_dm:.0f}s) "
              f"entropy={r_dm.get('routing_entropy', '?')}", flush=True)

        gap = round(r_dm['mean_acc'] - r_cnn['mean_acc'], 4)
        print(f"  Gap: {gap:+.4f}", flush=True)

        all_results["n_corruption_range"].append(n_corr)
        all_results["results"][str(n_corr)] = {
            "n_corruptions": n_corr,
            "corruption_names": [c[0] for c in corruptions],
            "static_cnn": r_cnn,
            "dirtyman": r_dm,
            "gap": gap,
        }

    # Write results
    out_path = os.path.join(RESULTS, "scaling_curves.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Summary
    print("\n=== SCALING SUMMARY ===", flush=True)
    print(f"{'n_corr':>6}  {'CNN':>6}  {'DM':>6}  {'Gap':>7}  {'Entropy':>7}")
    for n_corr in all_results["n_corruption_range"]:
        r = all_results["results"][str(n_corr)]
        print(f"{n_corr:>6}  {r['static_cnn']['mean_acc']:>6.3f}  "
              f"{r['dirtyman']['mean_acc']:>6.3f}  {r['gap']:>+7.4f}  "
              f"{r['dirtyman'].get('routing_entropy', 0):>7.3f}")

    return all_results


if __name__ == "__main__":
    main()
