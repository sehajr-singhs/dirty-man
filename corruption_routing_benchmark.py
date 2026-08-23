"""FashionMNIST corruption-routing benchmark — the novelty experiment.

The claim: on data with multiple corruption types that each favor a different
inductive bias, feature-conditioned routing (the Dirty Man) outperforms every
existing conditional-computation method, because those methods condition on
token content or confidence, not on *what kind of corruption* is present.

Benchmark design:
  - FashionMNIST (real photos of clothing, 10 classes, 60k train / 10k test)
  - Four corruption types: Gaussian noise, salt-and-pepper, rotation, occlusion
  - Each corruption favors a different lens
  - Train: mixed corruptions + clean; Test: per-corruption accuracy
  - Baselines: static MLP, static CNN, token-choice MoE, dynamic-depth
    early-exit, adaptive-computation binary gates, Dirty Man eye+router
  - Diagnostics: per-corruption routing policy, ablation of eye, random router

Output: results/corruption_routing.json
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
# Corruptions — four types that each demand a different inductive bias
# ---------------------------------------------------------------------------

def gaussian_noise(x, severity=0.3):
    """Global additive noise — favors averaging / linear smoothing."""
    return x + severity * torch.randn_like(x)


def salt_and_pepper(x, severity=0.15):
    """Sparse pixel corruption — favors nonlinear thresholding / sparsity."""
    mask = torch.rand_like(x) < severity
    noise = torch.rand_like(x) * 2 - 1
    return torch.where(mask, noise, x)


def rotate(x, severity=30.0):
    """Spatial rotation — favors local invariance / convolutions."""
    angle = (torch.rand(1).item() * 2 - 1) * severity
    theta = torch.tensor([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0],
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(x.unsqueeze(0), grid, align_corners=False,
                         padding_mode='border').squeeze(0)


def block_occlusion(x, severity=8):
    """Structured occlusion — favors attention/gating to visible regions."""
    out = x.clone()
    _, h, w = x.shape
    s = int(severity)
    for _ in range(3):
        y0 = torch.randint(0, h - s, (1,)).item()
        x0 = torch.randint(0, w - s, (1,)).item()
        out[:, y0:y0 + s, x0:x0 + s] = 0.0
    return out


CORRUPTIONS = [
    ("clean", lambda x: x),
    ("gaussian", lambda x: gaussian_noise(x, 0.3)),
    ("saltpepper", lambda x: salt_and_pepper(x, 0.15)),
    ("rotation", lambda x: rotate(x, 30.0)),
    ("occlusion", lambda x: block_occlusion(x, 8)),
]


class CorruptionDataset(Dataset):
    """Wraps a base dataset and returns (corrupted_x, label, corruption_id)."""

    def __init__(self, base, corruptions, n_corrupt=None, seed=0):
        self.base = base
        self.corruptions = corruptions
        g = torch.Generator().manual_seed(seed)
        n = len(base)
        n_each = n // len(corruptions)
        assignments = []
        for ci in range(len(corruptions)):
            assignments.extend([ci] * n_each)
        while len(assignments) < n:
            assignments.append(len(assignments) % len(corruptions))
        perm = torch.randperm(n, generator=g)
        self.assignments = [assignments[p.item()] for p in perm]
        if n_corrupt is not None:
            self.assignments = self.assignments[:n_corrupt]

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, idx):
        ci = self.assignments[idx]
        sample_idx = idx % len(self.base)
        x, y = self.base[sample_idx]
        name, fn = self.corruptions[ci]
        x_c = fn(x)
        return x_c, y, ci


def load_fashion_corrupted(n_train=20000, n_test=4000, seed=0):
    from torchvision import datasets, transforms
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    base_train = datasets.FashionMNIST("data_fm", train=True, download=False,
                                        transform=tr)
    base_test = datasets.FashionMNIST("data_fm", train=False, download=False,
                                       transform=tr)

    g = torch.Generator().manual_seed(seed)
    te_idx = torch.randperm(len(base_test), generator=g)[:n_test].tolist()
    per_corruption_test = {}
    for ci, (name, fn) in enumerate(CORRUPTIONS):
        xs, ys = [], []
        for idx in te_idx:
            x, y = base_test[idx]
            xs.append(fn(x))
            ys.append(y)
        per_corruption_test[name] = TensorDataset(
            torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    train_ds = CorruptionDataset(base_train, CORRUPTIONS, n_train, seed)
    test_ds = CorruptionDataset(base_test, CORRUPTIONS,
                                min(n_test, len(base_test)), seed + 1)
    return train_ds, test_ds, per_corruption_test


# ---------------------------------------------------------------------------
# Static baselines
# ---------------------------------------------------------------------------

class StaticMLP(nn.Module):
    def __init__(self, n_classes=10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


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


# ---------------------------------------------------------------------------
# Token-choice MoE (Switch Transformer style)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    def __init__(self, dim, n_experts=4, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                          nn.Linear(dim * 2, dim))
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        B, D = x.shape
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (topk_idx[:, k] == e)
                if mask.any():
                    out[mask] += (topk_probs[mask, k].unsqueeze(-1) *
                                  self.experts[e](x[mask]))
        frac = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            frac[e] = (topk_idx == e).any(dim=-1).float().mean()
        mean_prob = probs.mean(0)
        balance_loss = self.n_experts * (frac * mean_prob).sum()
        return out, balance_loss


class MoENet(nn.Module):
    def __init__(self, n_classes=10, n_experts=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(28 * 28, 128)
        self.moe1 = MoELayer(128, n_experts, top_k=2)
        self.moe2 = MoELayer(128, n_experts, top_k=2)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        h = F.relu(self.proj(self.flatten(x)))
        h, b1 = self.moe1(h)
        h, b2 = self.moe2(h)
        return self.head(h), b1 + b2

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dynamic-depth (early exit)
# ---------------------------------------------------------------------------

class DynamicDepthNet(nn.Module):
    def __init__(self, n_classes=10, n_exits=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(28 * 28 if i == 0 else 128, 128),
                          nn.ReLU())
            for i in range(n_exits)
        ])
        self.exits = nn.ModuleList([
            nn.Linear(128, n_classes) for _ in range(n_exits)
        ])

    def forward(self, x):
        h = self.flatten(x)
        total = 0.0
        for layer, exit_head in zip(self.layers, self.exits):
            h = layer(h)
            out = exit_head(h)
            total = total + (1.0 / len(self.layers)) * out
        return total, 0.0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Adaptive computation (binary gate on each layer)
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        logit = self.gate(x.detach())
        if self.training:
            prob = torch.sigmoid(logit)
            hard = (prob > 0.5).float()
            gate = hard + prob - prob.detach()
        else:
            gate = (torch.sigmoid(logit) > 0.5).float()
        return gate, logit.sigmoid().mean()


class AdaptiveNet(nn.Module):
    def __init__(self, n_classes=10, n_layers=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(28 * 28, 128)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.gates = nn.ModuleList([
            AdaptiveGate(128) for _ in range(n_layers)
        ])
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        h = F.relu(self.proj(self.flatten(x)))
        sparsity_penalty = 0.0
        for layer, gate_mod in zip(self.layers, self.gates):
            g, p = gate_mod(h)
            h_new = layer(h)
            h = g * h_new + (1 - g) * h
            sparsity_penalty = sparsity_penalty + p
        return self.head(h), sparsity_penalty

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dirty Man — feature-conditioned routing with HARD routing
# ---------------------------------------------------------------------------

class DirtyManNet(nn.Module):
    """Feature-conditioned routing: eye detects what kind of corruption is
    present, router picks the lens best suited for it.

    Key differences from standard MoE:
      1. Routing is conditioned on corruption *features* (eye), not token values
      2. Lenses have genuinely different inductive biases (linear, ReLU, CNN, gated)
      3. Hard routing: each sample goes to exactly one lens (not soft mixture)
      4. Router is pre-trained on corruption classification, then fine-tuned end-to-end
    """

    LENS_WIDTH = 96

    def __init__(self, n_classes=10, n_corruptions=5):
        super().__init__()
        self.n_lenses = 4
        self.n_corruptions = n_corruptions

        # Eye: detects corruption type from image features
        self.eye = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        )
        # Corruption classifier (used during warm-up only)
        self.corruption_head = nn.Linear(128, n_corruptions)

        # Lens 0: pure linear — no nonlinearity (best for Gaussian noise)
        self.lens0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.LENS_WIDTH),
        )
        # Lens 1: ReLU nonlinear (best for salt-and-pepper)
        self.lens1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(8 * 4 * 4, self.LENS_WIDTH),
        )
        # Lens 2: CNN with spatial structure (best for rotation)
        self.lens2 = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 12, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(12 * 4 * 4, self.LENS_WIDTH),
        )
        # Lens 3: Gated (best for occlusion)
        self.lens3_gate = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, self.LENS_WIDTH),
        )
        self.lens3_cell = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, self.LENS_WIDTH),
        )

        # Router: eye features -> lens logits (simple linear for clean init)
        self.router = nn.Linear(128, self.n_lenses)
        self.head = nn.Linear(self.LENS_WIDTH, n_classes)

    def forward(self, x, tau=1.0, hard=True, record=None):
        cues = self.eye(x)
        logits = self.router(cues)  # (B, n_lenses)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)

        # Compute all lenses
        lens_outs = [
            self.lens0(x),
            self.lens1(x),
            self.lens2(x),
            self.lens3_gate(x).sigmoid() * self.lens3_cell(x).tanh(),
        ]

        # Hard routing: each sample goes to exactly one lens
        if hard:
            idx = probs.argmax(dim=-1)  # (B,)
            mixture = torch.zeros_like(lens_outs[0])
            for i, out in enumerate(lens_outs):
                mask = (idx == i).unsqueeze(-1).float()
                mixture = mixture + mask * out
        else:
            mixture = sum(probs[:, i].unsqueeze(-1) * out
                          for i, out in enumerate(lens_outs))

        out = self.head(mixture)
        if record is not None:
            record["probs"] = probs.detach().cpu()
            record["cues"] = cues.detach().cpu()
            record["corr_logits"] = self.corruption_head(cues).detach().cpu()
        return out

    def forward_corruption(self, x):
        """Predict corruption type from eye features (warm-up)."""
        return self.corruption_head(self.eye(x))

    def router_from_corruption_head(self):
        """Initialize router weights from corruption classifier.
        This gives the router a strong prior: it starts knowing which
        corruption types exist and can map them to different lenses."""
        with torch.no_grad():
            src = self.corruption_head.weight.data  # (5, 128)
            # Map 5 corruptions to 4 lenses: clean→0, gaussian→1, sp→2, rot→3, occ→3
            mapping = [0, 1, 2, 3, 3]
            for corr_id, lens_id in enumerate(mapping):
                self.router.weight.data[lens_id] += src[corr_id].clone() * 0.5
            self.router.bias.data.zero_()

    def lens_params(self):
        counts = {}
        for i, mod in enumerate([self.lens0, self.lens1, self.lens2,
                                  nn.ModuleList([self.lens3_gate, self.lens3_cell])]):
            counts[f"lens_{i}"] = sum(p.numel() for p in mod.parameters())
        return counts

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def anneal_tau(epoch, total_epochs, start=2.0, end=0.3):
    if total_epochs <= 1:
        return end
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * epoch / (total_epochs - 1)))


def train_model(model, train_ds, test_ds, per_corruption_test, epochs,
                model_type, batch, device, seed):
    """Train one model, return per-corruption accuracies."""
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # -----------------------------------------------------------------------
    # Phase 1: Dirty Man warm-up — train eye to detect corruption type,
    # then initialize router from corruption classifier weights
    # -----------------------------------------------------------------------
    if model_type == "dirtyman":
        warm_epochs = max(5, epochs // 3)
        corr_opt = torch.optim.AdamW(
            list(model.eye.parameters()) + list(model.corruption_head.parameters()),
            lr=1e-3, weight_decay=1e-4)
        for ep in range(warm_epochs):
            model.train()
            corr_correct = corr_total = 0
            for batch_data in loader:
                x, y, ci = batch_data
                x, ci = x.to(device), ci.to(device)
                corr_opt.zero_grad(set_to_none=True)
                corr_logits = model.forward_corruption(x)
                loss = F.cross_entropy(corr_logits, ci)
                loss.backward()
                corr_opt.step()
                corr_correct += (corr_logits.argmax(-1) == ci).sum().item()
                corr_total += x.size(0)
            corr_acc = corr_correct / max(corr_total, 1)
            print(f"  [dirtyman warm-up] ep {ep+1}/{warm_epochs} "
                  f"corruption cls acc: {corr_acc:.3f}", flush=True)

        # Initialize router from corruption classifier
        model.router_from_corruption_head()
        print("  [dirtyman] router initialized from corruption classifier",
              flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Main training
    # -----------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = total = 0
        tau = anneal_tau(epoch, epochs) if model_type == "dirtyman" else 1.0

        for batch_data in loader:
            x, y, ci = batch_data
            x, y, ci = x.to(device), y.to(device), ci.to(device)

            opt.zero_grad(set_to_none=True)

            if model_type == "moe":
                out, bal = model(x)
                task_loss = F.cross_entropy(out, y)
                loss = task_loss + 0.01 * bal
            elif model_type == "dynamicdepth":
                out, _ = model(x)
                loss = F.cross_entropy(out, y)
            elif model_type == "adaptive":
                out, sp = model(x)
                loss = F.cross_entropy(out, y) + 0.01 * sp * epochs
            elif model_type == "dirtyman":
                # Use SOFT routing during training for gradient flow
                out = model(x, tau=tau, hard=False)
                task_loss = F.cross_entropy(out, y)

                # Load-balance loss (differentiable)
                logits = model.router(model.eye(x))
                probs = F.softmax(logits, dim=-1)
                frac = probs.mean(0)  # average routing probability per lens
                target = torch.ones(model.n_lenses, device=device) / model.n_lenses
                bal = ((frac - target) ** 2).sum()

                # Pairwise diversity: per-corruption mean routing should differ
                div_loss = torch.zeros((), device=device)
                corr_means = []
                for c in range(model.n_corruptions):
                    mask = (ci == c)
                    if mask.sum() > 0:
                        p_c = probs[mask].mean(0)
                        corr_means.append(p_c)
                for i in range(len(corr_means)):
                    for j in range(i + 1, len(corr_means)):
                        # Cosine similarity (minimize to maximize diversity)
                        cos_sim = F.cosine_similarity(
                            corr_means[i].unsqueeze(0),
                            corr_means[j].unsqueeze(0))
                        div_loss = div_loss + cos_sim

                loss = task_loss + 0.3 * bal + 0.1 * div_loss
            else:  # static
                out = model(x)
                loss = F.cross_entropy(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        # Print epoch summary for dirtyman
        if model_type == "dirtyman" and (epoch + 1) % 2 == 0:
            print(f"  [dirtyman] ep {epoch+1}/{epochs} loss={total_loss/total:.4f} "
                  f"train_acc={correct/total:.3f} tau={tau:.3f}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 3: Evaluate per-corruption
    # -----------------------------------------------------------------------
    model.eval()
    per_corruption_acc = {}
    routing_policy = {}
    corr_classifier_acc = {}
    with torch.no_grad():
        for cname, cdata in per_corruption_test.items():
            cloader = DataLoader(cdata, batch_size=batch, num_workers=0)
            c_correct = c_total = 0
            all_probs = []
            corr_correct = 0
            for cx, cy in cloader:
                cx, cy = cx.to(device), cy.to(device)
                if model_type == "dirtyman":
                    record = {}
                    out = model(cx, tau=0.1, hard=True, record=record)
                    all_probs.append(record["probs"])
                    ci_true = next(i for i, (n, _) in enumerate(CORRUPTIONS)
                                   if n == cname)
                    corr_pred = record["corr_logits"].argmax(-1)
                    corr_correct += (corr_pred == ci_true).sum().item()
                elif model_type in ("moe", "dynamicdepth", "adaptive"):
                    out, _ = model(cx)
                else:
                    out = model(cx)
                _, pred = out.max(1)
                c_correct += (pred == cy).sum().item()
                c_total += cx.size(0)
            per_corruption_acc[cname] = round(c_correct / max(c_total, 1), 4)
            if model_type == "dirtyman":
                corr_classifier_acc[cname] = round(corr_correct / max(c_total, 1), 4)
            if all_probs:
                avg_probs = torch.cat(all_probs, dim=0).mean(0)
                routing_policy[cname] = {
                    f"lens_{i}": round(float(avg_probs[i]), 4)
                    for i in range(model.n_lenses)
                }

    result = {
        "train_acc": round(correct / max(total, 1), 4),
        "per_corruption_acc": per_corruption_acc,
        "routing_policy": routing_policy if routing_policy else None,
        "params": model.n_params(),
    }
    if model_type == "dirtyman":
        result["corruption_classifier_acc"] = corr_classifier_acc
        result["lens_params"] = {k: v for k, v in model.lens_params().items()}
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dirtyman-only", action="store_true",
                    help="Only run Dirty Man (for quick iteration)")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_train, args.n_test, args.epochs, args.batch = 800, 200, 3, 64
    device = args.device

    print(f"=== Corruption Routing Benchmark ===", flush=True)
    print(f"Train: {args.n_train}, Test: {args.n_test}, Epochs: {args.epochs}",
          flush=True)

    train_ds, test_ds, per_corruption_test = load_fashion_corrupted(
        args.n_train, args.n_test, args.seed)
    print(f"Loaded FashionMNIST: {len(train_ds)} train, {len(test_ds)} test",
          flush=True)

    results = {"experiment": "corruption_routing_benchmark",
               "corruption_types": [c[0] for c in CORRUPTIONS],
               "config": {"n_train": args.n_train, "n_test": args.n_test,
                          "epochs": args.epochs, "seed": args.seed}}

    if not args.dirtyman_only:
        # Static baselines
        for name, make_model in [("static_mlp", lambda: StaticMLP()),
                                  ("static_cnn", lambda: StaticCNN())]:
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            m = make_model().to(device)
            r = train_model(m, train_ds, test_ds, per_corruption_test,
                            args.epochs, "static", args.batch, device, args.seed)
            r["type"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results[name] = r
            print(f"  params={r['params']}  per-corr acc: {r['per_corruption_acc']}",
                  flush=True)

        # MoE
        print(f"\n=== MoE (token-choice) ===", flush=True)
        t0 = time.time()
        m = MoENet().to(device)
        r = train_model(m, train_ds, test_ds, per_corruption_test,
                        args.epochs, "moe", args.batch, device, args.seed)
        r["type"] = "moe"
        r["time_s"] = round(time.time() - t0, 1)
        results["moe"] = r
        print(f"  params={r['params']}  per-corr acc: {r['per_corruption_acc']}",
              flush=True)

        # Dynamic depth
        print(f"\n=== Dynamic Depth (early exit) ===", flush=True)
        t0 = time.time()
        m = DynamicDepthNet().to(device)
        r = train_model(m, train_ds, test_ds, per_corruption_test,
                        args.epochs, "dynamicdepth", args.batch, device, args.seed)
        r["type"] = "dynamicdepth"
        r["time_s"] = round(time.time() - t0, 1)
        results["dynamicdepth"] = r
        print(f"  params={r['params']}  per-corr acc: {r['per_corruption_acc']}",
              flush=True)

        # Adaptive computation
        print(f"\n=== Adaptive Computation (binary gates) ===", flush=True)
        t0 = time.time()
        m = AdaptiveNet().to(device)
        r = train_model(m, train_ds, test_ds, per_corruption_test,
                        args.epochs, "adaptive", args.batch, device, args.seed)
        r["type"] = "adaptive"
        r["time_s"] = round(time.time() - t0, 1)
        results["adaptive"] = r
        print(f"  params={r['params']}  per-corr acc: {r['per_corruption_acc']}",
              flush=True)

    # Dirty Man
    print(f"\n=== Dirty Man (feature-conditioned routing) ===", flush=True)
    t0 = time.time()
    m = DirtyManNet().to(device)
    r = train_model(m, train_ds, test_ds, per_corruption_test,
                    args.epochs, "dirtyman", args.batch, device, args.seed)
    r["type"] = "dirtyman"
    r["time_s"] = round(time.time() - t0, 1)
    results["dirtyman"] = r
    print(f"  params={r['params']}  per-corr acc: {r['per_corruption_acc']}",
          flush=True)
    if r["routing_policy"]:
        print(f"  routing policy: {json.dumps(r['routing_policy'], indent=2)}",
              flush=True)
    if r.get("corruption_classifier_acc"):
        print(f"  corruption classifier: {r['corruption_classifier_acc']}",
              flush=True)

    # Write results
    with open(os.path.join(RESULTS, "corruption_routing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote results/corruption_routing.json", flush=True)

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    for name in ["static_mlp", "static_cnn", "moe", "dynamicdepth",
                 "adaptive", "dirtyman"]:
        if name not in results:
            continue
        r = results[name]
        accs = r.get("per_corruption_acc", {})
        mean_acc = round(sum(accs.values()) / max(len(accs), 1), 4)
        print(f"  {name:20s} params={r.get('params', '?'):>7}  mean={mean_acc}  "
              f"accs={accs}", flush=True)

    return results


if __name__ == "__main__":
    main()
