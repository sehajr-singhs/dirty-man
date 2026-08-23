"""FashionMNIST corruption-routing benchmark — the novelty experiment.

Tests whether feature-conditioned routing (Dirty Man) discovers which
computational regime each corruption type requires.

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
# Corruptions
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


CORRUPTIONS = [
    ("clean", lambda x: x),
    ("gaussian", lambda x: gaussian_noise(x, 0.3)),
    ("saltpepper", lambda x: salt_and_pepper(x, 0.15)),
    ("rotation", lambda x: rotate(x, 30.0)),
    ("occlusion", lambda x: block_occlusion(x, 8)),
]


class CorruptionDataset(Dataset):
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
        return fn(x), y, ci


def load_fashion_corrupted(n_train=20000, n_test=4000, seed=0):
    from torchvision import datasets, transforms
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    base_train = datasets.FashionMNIST("data_fm", train=True, download=False, transform=tr)
    base_test = datasets.FashionMNIST("data_fm", train=False, download=False, transform=tr)
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
            nn.Flatten(), nn.Linear(28*28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))
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
            nn.Flatten(), nn.Linear(64*7*7, n_classes))
    def forward(self, x):
        return self.net(x)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# MoE baseline
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    def __init__(self, dim, n_experts=4, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
            for _ in range(n_experts)])
        self.gate = nn.Linear(dim, n_experts)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (topk_idx[:, k] == e)
                if mask.any():
                    out[mask] += (topk_probs[mask, k].unsqueeze(-1) * self.experts[e](x[mask]))
        frac = torch.zeros(self.n_experts, device=x.device)
        for e in range(self.n_experts):
            frac[e] = (topk_idx == e).any(dim=-1).float().mean()
        mean_prob = probs.mean(0)
        return out, self.n_experts * (frac * mean_prob).sum()


class MoENet(nn.Module):
    def __init__(self, n_classes=10, n_experts=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(28*28, 128)
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
# Dynamic-depth baseline
# ---------------------------------------------------------------------------

class DynamicDepthNet(nn.Module):
    def __init__(self, n_classes=10, n_exits=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(28*28 if i==0 else 128, 128), nn.ReLU())
            for i in range(n_exits)])
        self.exits = nn.ModuleList([nn.Linear(128, n_classes) for _ in range(n_exits)])
    def forward(self, x):
        h = self.flatten(x)
        total = 0.0
        for layer, exit_head in zip(self.layers, self.exits):
            h = layer(h)
            total = total + (1.0 / len(self.layers)) * exit_head(h)
        return total, 0.0
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Adaptive computation baseline
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
            return hard + prob - prob.detach(), prob.mean()
        return (torch.sigmoid(logit) > 0.5).float(), logit.sigmoid().mean()


class AdaptiveNet(nn.Module):
    def __init__(self, n_classes=10, n_layers=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(28*28, 128)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(128, 128), nn.ReLU()) for _ in range(n_layers)])
        self.gates = nn.ModuleList([AdaptiveGate(128) for _ in range(n_layers)])
        self.head = nn.Linear(128, n_classes)
    def forward(self, x):
        h = F.relu(self.proj(self.flatten(x)))
        sp = 0.0
        for layer, gate_mod in zip(self.layers, self.gates):
            g, p = gate_mod(h)
            h = g * layer(h) + (1 - g) * h
            sp = sp + p
        return self.head(h), sp
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dirty Man — feature-conditioned routing with COMPLETE CLASSIFIERS
# ---------------------------------------------------------------------------

class DirtyManNet(nn.Module):
    """Feature-conditioned routing: eye detects corruption type, router
    selects which COMPLETE CLASSIFIER processes each sample.

    Each lens is a standalone classifier with different inductive bias
    and its own classification head. This means the router is switching
    between complete computational programs, not mixing features.
    """

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
        self.corruption_head = nn.Linear(128, n_corruptions)

        # === Each lens is a COMPLETE CLASSIFIER (~50k params each) ===

        # Lens 0: Linear classifier — no convolutions, pure linear map
        # Best for: clean data, Gaussian noise (smooth, no spatial structure)
        self.lens0 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        # Lens 1: Deep ReLU MLP — nonlinear thresholding
        # Best for: salt-and-pepper (sparse corruption needs nonlinearity)
        self.lens1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        # Lens 2: CNN — spatial convolutions
        # Best for: rotation (spatial invariance)
        self.lens2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, n_classes),
        )
        # Lens 3: Gated network — attention to visible regions
        # Best for: occlusion (learned gating over corrupted regions)
        self.lens3_gate = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128))
        self.lens3_cell = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128))
        self.lens3_head = nn.Linear(128, n_classes)

        # Router: eye features -> lens logits
        self.router = nn.Linear(128, self.n_lenses)

    def forward(self, x, tau=1.0, hard=True, record=None):
        cues = self.eye(x)
        logits = self.router(cues)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)

        # Each lens produces CLASS LOGITS (not features)
        lens_logits = [
            self.lens0(x),
            self.lens1(x),
            self.lens2(x),
            self.lens3_head(self.lens3_gate(x).sigmoid() * self.lens3_cell(x).tanh()),
        ]

        # Weighted combination of lens logits
        mixture = sum(probs[:, i].unsqueeze(-1) * logits_i
                      for i, logits_i in enumerate(lens_logits))

        if record is not None:
            record["probs"] = probs.detach().cpu()
            record["cues"] = cues.detach().cpu()
            record["corr_logits"] = self.corruption_head(cues).detach().cpu()
        return mixture

    def forward_corruption(self, x):
        return self.corruption_head(self.eye(x))

    def router_from_corruption_head(self):
        with torch.no_grad():
            src = self.corruption_head.weight.data
            mapping = [0, 1, 2, 3, 3]
            for corr_id, lens_id in enumerate(mapping):
                self.router.weight.data[lens_id] += src[corr_id].clone() * 0.5
            self.router.bias.data.zero_()

    def lens_params(self):
        counts = {}
        for i, mod in enumerate([self.lens0, self.lens1, self.lens2,
                                  nn.ModuleList([self.lens3_gate, self.lens3_cell, self.lens3_head])]):
            counts[f"lens_{i}"] = sum(p.numel() for p in mod.parameters())
        return counts

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def anneal_tau(epoch, total_epochs, start=2.0, end=0.3):
    if total_epochs <= 1:
        return end
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * epoch / (total_epochs - 1)))


def train_model(model, train_ds, test_ds, per_corruption_test, epochs,
                model_type, batch, device, seed):
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dirty Man warm-up
    if model_type == "dirtyman":
        warm_epochs = max(5, epochs // 3)
        corr_opt = torch.optim.AdamW(
            list(model.eye.parameters()) + list(model.corruption_head.parameters()),
            lr=1e-3, weight_decay=1e-4)
        for ep in range(warm_epochs):
            model.train()
            corr_correct = corr_total = 0
            for x, y, ci in loader:
                x, ci = x.to(device), ci.to(device)
                corr_opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.forward_corruption(x), ci)
                loss.backward()
                corr_opt.step()
                corr_correct += (model.forward_corruption(x).argmax(-1) == ci).sum().item()
                corr_total += x.size(0)
            print(f"  [warm-up] ep {ep+1}/{warm_epochs} corr acc: {corr_correct/max(corr_total,1):.3f}", flush=True)
        model.router_from_corruption_head()

    for epoch in range(epochs):
        model.train()
        tau = anneal_tau(epoch, epochs) if model_type == "dirtyman" else 1.0
        total_loss = 0.0
        correct = total = 0

        for x, y, ci in loader:
            x, y, ci = x.to(device), y.to(device), ci.to(device)
            opt.zero_grad(set_to_none=True)

            if model_type == "moe":
                out, bal = model(x)
                loss = F.cross_entropy(out, y) + 0.01 * bal
            elif model_type == "dynamicdepth":
                out, _ = model(x)
                loss = F.cross_entropy(out, y)
            elif model_type == "adaptive":
                out, sp = model(x)
                loss = F.cross_entropy(out, y) + 0.01 * sp * epochs
            elif model_type == "dirtyman":
                out = model(x, tau=tau, hard=False)
                task_loss = F.cross_entropy(out, y)

                logits = model.router(model.eye(x))
                probs = F.softmax(logits, dim=-1)
                frac = probs.mean(0)
                target = torch.ones(model.n_lenses, device=device) / model.n_lenses
                bal = ((frac - target) ** 2).sum()

                # Pairwise diversity
                corr_means = []
                for c in range(len(CORRUPTIONS)):
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
            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        if model_type == "dirtyman" and (epoch + 1) % 2 == 0:
            print(f"  [dirtyman] ep {epoch+1}/{epochs} loss={total_loss/total:.4f} "
                  f"acc={correct/total:.3f} tau={tau:.3f}", flush=True)

    # Evaluate per-corruption
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
                    ci_true = next(i for i, (n, _) in enumerate(CORRUPTIONS) if n == cname)
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
                routing_policy[cname] = {f"lens_{i}": round(float(avg_probs[i]), 4)
                                         for i in range(model.n_lenses)}

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
    ap.add_argument("--dirtyman-only", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.n_train, args.n_test, args.epochs, args.batch = 800, 200, 3, 64
    device = args.device

    print(f"=== Corruption Routing Benchmark ===", flush=True)
    print(f"Train: {args.n_train}, Test: {args.n_test}, Epochs: {args.epochs}", flush=True)

    train_ds, test_ds, per_corruption_test = load_fashion_corrupted(
        args.n_train, args.n_test, args.seed)

    results = {"experiment": "corruption_routing_benchmark",
               "corruption_types": [c[0] for c in CORRUPTIONS],
               "config": {"n_train": args.n_train, "n_test": args.n_test,
                          "epochs": args.epochs, "seed": args.seed}}

    if not args.dirtyman_only:
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
            print(f"  params={r['params']}  per-corr: {r['per_corruption_acc']}", flush=True)

        for name, make_model, mtype in [
            ("moe", lambda: MoENet(), "moe"),
            ("dynamicdepth", lambda: DynamicDepthNet(), "dynamicdepth"),
            ("adaptive", lambda: AdaptiveNet(), "adaptive"),
        ]:
            print(f"\n=== {name} ===", flush=True)
            t0 = time.time()
            m = make_model().to(device)
            r = train_model(m, train_ds, test_ds, per_corruption_test,
                            args.epochs, mtype, args.batch, device, args.seed)
            r["type"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results[name] = r
            print(f"  params={r['params']}  per-corr: {r['per_corruption_acc']}", flush=True)

    print(f"\n=== Dirty Man (feature-conditioned routing) ===", flush=True)
    t0 = time.time()
    m = DirtyManNet().to(device)
    r = train_model(m, train_ds, test_ds, per_corruption_test,
                    args.epochs, "dirtyman", args.batch, device, args.seed)
    r["type"] = "dirtyman"
    r["time_s"] = round(time.time() - t0, 1)
    results["dirtyman"] = r
    print(f"  params={r['params']}  per-corr: {r['per_corruption_acc']}", flush=True)
    if r["routing_policy"]:
        print(f"  routing: {json.dumps(r['routing_policy'], indent=2)}", flush=True)
    if r.get("corruption_classifier_acc"):
        print(f"  corruption cls: {r['corruption_classifier_acc']}", flush=True)

    with open(os.path.join(RESULTS, "corruption_routing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote results/corruption_routing.json", flush=True)

    print(f"\n=== SUMMARY ===", flush=True)
    for name in ["static_mlp", "static_cnn", "moe", "dynamicdepth", "adaptive", "dirtyman"]:
        if name not in results:
            continue
        r = results[name]
        accs = r.get("per_corruption_acc", {})
        mean_acc = round(sum(accs.values()) / max(len(accs), 1), 4)
        print(f"  {name:20s} params={r.get('params','?'):>7}  mean={mean_acc}  accs={accs}", flush=True)
    return results


if __name__ == "__main__":
    main()
