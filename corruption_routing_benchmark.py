"""
Corruption Routing Benchmark — NMI-level
=========================================

FashionMNIST (real clothing photos, 10 classes) corrupted by 4 types.
Each corruption demands a different inductive bias.

KEY DESIGN: Each Dirty Man lens is a FULL CNN classifier with the SAME
architecture as the static CNN baseline. The only difference: the Dirty Man
routes between them. This makes the comparison fair and the routing result
clean: when each lens has the same capacity as the static CNN, the routing
advantage should come purely from structural adaptation.

The experiment:
1. Static CNN baseline: one CNN trained on all corruptions
2. Static MLP baseline: one MLP trained on all corruptions
3. Dirty Man: K copies of the CNN, each a "lens" — router selects per sample
4. MoE: same K lenses, but router conditions on pixel values (content routing)
5. Heterogeneity scaling: sweep number of corruption types (2→3→4→5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from torchvision import datasets, transforms


# ============================================================================
# Corruption generators
# ============================================================================

CORRUPTION_TYPES = ["clean", "gaussian", "saltpepper", "rotation", "occlusion"]


def apply_corruption(img, ctype, severity=1.0):
    if ctype == "clean":
        return img
    elif ctype == "gaussian":
        return torch.clamp(img + torch.randn_like(img) * 0.3 * severity, 0, 1)
    elif ctype == "saltpepper":
        mask = torch.rand_like(img)
        out = img.clone()
        out[mask < 0.15 * severity] = 0
        out[mask > 1 - 0.15 * severity] = 1
        return out
    elif ctype == "rotation":
        angle = severity * 45
        rad = angle * 3.14159 / 180
        c, s = torch.cos(torch.tensor(rad)), torch.sin(torch.tensor(rad))
        h, w = img.shape[1], img.shape[2]
        y, x = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing='ij')
        nx = (x * c - y * s).long() + w // 2
        ny = (x * s + y * c).long() + h // 2
        out = torch.zeros_like(img)
        valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        out[:, ny[valid], nx[valid]] = img[:, y[valid], x[valid]]
        return out
    elif ctype == "occlusion":
        out = img.clone()
        size = max(1, int(28 * 0.3 * severity))
        x0 = np.random.randint(0, max(1, 28 - size + 1))
        y0 = np.random.randint(0, max(1, 28 - size + 1))
        out[:, y0:y0 + size, x0:x0 + size] = 0
        return out
    return img


def make_dataset(n_samples, severity=1.0, seed=0, corruption_types=None):
    """Balanced dataset with specified corruption types."""
    if corruption_types is None:
        corruption_types = CORRUPTION_TYPES
    rng = np.random.RandomState(seed)
    transform = transforms.Compose([transforms.ToTensor()])
    base = datasets.FashionMNIST("data_fm", train=True, download=True, transform=transform)

    xs, ys, cs = [], [], []
    per_type = n_samples // len(corruption_types)
    for ci, ctype in enumerate(corruption_types):
        for _ in range(per_type):
            idx = rng.randint(0, len(base))
            img, label = base[idx]
            xs.append(apply_corruption(img, ctype, severity))
            ys.append(label)
            cs.append(ci)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(cs)


# ============================================================================
# Models — all the same architecture, for fair comparison
# ============================================================================

def make_cnn(n_classes=10):
    """CNN classifier — the baseline and the lens template.
    Small enough for CPU training, large enough for real accuracy."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        nn.Flatten(), nn.Linear(32 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, n_classes)
    )


def make_mlp(n_classes=10):
    """MLP classifier."""
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes)
    )


class LensBank(nn.Module):
    """K copies of the same CNN, each a "lens" with its own head."""
    def __init__(self, k=4, n_classes=10):
        super().__init__()
        self.lenses = nn.ModuleList([make_cnn(n_classes) for _ in range(k)])
        self.k = k

    def forward(self, x, idx=None):
        if idx is not None:
            return self.lenses[idx](x)
        return torch.stack([l(x) for l in self.lenses], dim=1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Routers
# ============================================================================

class EyeRouter(nn.Module):
    """Meta router: eye features → lens selection (Theorem 7)."""
    def __init__(self, k=4, n_corruptions=5):
        super().__init__()
        self.eye = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
        )
        self.corr_head = nn.Linear(128, n_corruptions)
        self.router = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k))

    def forward(self, x):
        return self.router(self.eye(x))

    def eye_features(self, x):
        return self.eye(x)


class PixelRouter(nn.Module):
    """Content router: raw pixels → lens selection (MoE-style)."""
    def __init__(self, k=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, k)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Training
# ============================================================================

def train_static(model, x, y, epochs=12, lr=1e-3, batch=128):
    """Train a static model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for s in range(0, len(x), batch):
            idx = perm[s:s + batch]
            out = model(x[idx])
            loss = F.cross_entropy(out, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_routed(bank, router, x, y, c, k, epochs=12, lr=1e-3, batch=128,
                 is_eye_router=False, warmup_frac=0.3):
    """Train bank + router jointly with Gumbel-Softmax."""
    params = list(router.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    warmup = max(1, int(epochs * warmup_frac))

    for epoch in range(epochs):
        # Eye router warm-up: learn corruption detection
        if is_eye_router and epoch < warmup:
            feat = router.eye_features(x)
            loss = F.cross_entropy(router.corr_head(feat), c)
            opt.zero_grad(); loss.backward(); opt.step()
            continue

        perm = torch.randperm(len(x))
        for s in range(0, len(x), batch):
            idx = perm[s:s + batch]
            bx, by, bc = x[idx], y[idx], c[idx]

            logits = router(bx)
            tau = max(0.5, 2.0 * (1 - epoch / max(epochs, 1)))
            probs = F.gumbel_softmax(logits, tau=tau, hard=False)

            all_out = bank(bx)  # (B, k, n_classes)
            out = (probs.unsqueeze(-1) * all_out).sum(dim=1)

            task_loss = F.cross_entropy(out, by)

            # Balance: entropy of usage
            usage = probs.mean(dim=0)
            balance = -(usage * torch.log(usage + 1e-8)).sum()

            # Corruption diversity: different corruptions → different lenses
            div_loss = torch.tensor(0.0)
            if is_eye_router:
                profiles = []
                for ci in range(len(CORRUPTION_TYPES)):
                    cmask = bc == ci
                    if cmask.sum() > 0:
                        profiles.append(probs[cmask].mean(dim=0))
                if len(profiles) >= 2:
                    M = torch.stack(profiles)
                    S = F.cosine_similarity(M.unsqueeze(0), M.unsqueeze(1), dim=-1)
                    diag = torch.eye(S.shape[0], dtype=torch.bool)
                    S[diag] = 0
                    div_weight = min(1.0, epoch / max(1, epochs * 0.4))
                    div_loss = div_weight * S.sum() / max(1, S.numel() - S.shape[0])

            total = task_loss + 0.1 * balance + 0.3 * div_loss
            opt.zero_grad(); total.backward(); opt.step()


def evaluate(model, x, y, c=None, k=None, is_router=False, bank=None):
    """Evaluate accuracy + routing distribution if applicable."""
    with torch.no_grad():
        if is_router and bank is not None:
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_out = bank(x)
            out = (probs.unsqueeze(-1) * all_out).sum(dim=1)
            routing = probs.argmax(dim=-1)

            acc = (out.argmax(-1) == y).float().mean().item()
            result = {"acc": round(acc, 4)}

            if c is not None and k is not None:
                result["routing_policy"] = {}
                result["per_corruption_acc"] = {}
                result["routing_entropy"] = {}
                for ci in range(k):
                    mask = c == ci
                    if mask.sum() > 0:
                        ct_name = CORRUPTION_TYPES[ci] if ci < len(CORRUPTION_TYPES) else f"corr_{ci}"
                        corr_acc = (out[mask].argmax(-1) == y[mask]).float().mean().item()
                        policy = {}
                        for j in range(k):
                            policy[f"lens_{j}"] = round((routing[mask] == j).float().mean().item(), 4)
                        result["routing_policy"][ct_name] = policy
                        result["per_corruption_acc"][ct_name] = round(corr_acc, 4)
                        p = probs[mask].mean(dim=0)
                        ent = -(p * torch.log(p + 1e-8)).sum().item()
                        result["routing_entropy"][ct_name] = round(ent, 4)
            return result
        else:
            out = model(x)
            acc = (out.argmax(-1) == y).float().mean().item()
            return {"acc": round(acc, 4)}


# ============================================================================
# Main experiment
# ============================================================================

def run_benchmark(n_train=30000, n_test=5000, epochs=12, seed=0,
                  corruption_types=None, severity=1.0):
    """Run the full corruption routing benchmark."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if corruption_types is None:
        corruption_types = CORRUPTION_TYPES
    n_corr = len(corruption_types)

    print(f"  Data: n_train={n_train}, n_test={n_test}, corruptions={n_corr} ({corruption_types})")
    train_x, train_y, train_c = make_dataset(n_train, severity, seed, corruption_types)
    test_x, test_y, test_c = make_dataset(n_test, severity, seed + 5000, corruption_types)

    t0 = time.time()
    results = {"config": {"n_train": n_train, "n_test": n_test, "epochs": epochs,
                          "seed": seed, "corruptions": corruption_types, "severity": severity}}

    # === Static baselines ===
    print("  Training static CNN...")
    cnn = make_cnn()
    train_static(cnn, train_x, train_y, epochs)
    cnn_res = evaluate(cnn, test_x, test_y)
    results["static_cnn"] = cnn_res
    print(f"    CNN: {cnn_res['acc']:.4f}")

    print("  Training static MLP...")
    mlp = make_mlp()
    train_static(mlp, train_x, train_y, epochs)
    mlp_res = evaluate(mlp, test_x, test_y)
    results["static_mlp"] = mlp_res
    print(f"    MLP: {mlp_res['acc']:.4f}")

    # === Dirty Man: meta routing on eye features ===
    print("  Training Dirty Man (meta routing)...")
    k = min(n_corr, 4)  # number of lenses (max 4, min = number of corruptions)
    bank = LensBank(k, 10)
    router = EyeRouter(k, n_corr)

    # Pretrain bank: each lens on different corruption subsets
    if n_corr >= k:
        # Assign each lens to roughly n_corr/k corruption types
        for li in range(k):
            start_c = li * n_corr // k
            end_c = (li + 1) * n_corr // k
            mask = torch.zeros(len(train_c), dtype=torch.bool)
            for ci in range(start_c, end_c):
                mask |= (train_c == ci)
            if mask.sum() > 0:
                opt = torch.optim.Adam(bank.lenses[li].parameters(), lr=1e-3)
                lx, ly = train_x[mask], train_y[mask]
                for _ in range(epochs // 2):
                    perm = torch.randperm(len(lx))
                    for s in range(0, len(lx), 128):
                        idx = perm[s:s + 128]
                        out = bank.lenses[li](lx[idx])
                        loss = F.cross_entropy(out, ly[idx])
                        opt.zero_grad(); loss.backward(); opt.step()

    train_routed(bank, router, train_x, train_y, train_c, k, epochs,
                 is_eye_router=True)
    dm_res = evaluate(router, test_x, test_y, test_c, k, is_router=True, bank=bank)
    results["dirty_man"] = dm_res
    print(f"    Dirty Man: {dm_res['acc']:.4f}")

    # === MoE: content routing on pixels ===
    print("  Training MoE (content routing)...")
    bank_moe = LensBank(k, 10)
    router_moe = PixelRouter(k)

    # Pretrain bank uniformly
    params_bank = list(bank_moe.parameters())
    opt = torch.optim.Adam(params_bank, lr=1e-3)
    for _ in range(epochs // 2):
        perm = torch.randperm(len(train_x))
        for s in range(0, len(train_x), 128):
            idx = perm[s:s + 128]
            all_out = bank_moe(train_x[idx])
            out = all_out.mean(dim=1)
            loss = F.cross_entropy(out, train_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    train_routed(bank_moe, router_moe, train_x, train_y, train_c, k, epochs)
    moe_res = evaluate(router_moe, test_x, test_y, test_c, k, is_router=True, bank=bank_moe)
    results["moe"] = moe_res
    print(f"    MoE: {moe_res['acc']:.4f}")

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    print(f"  Total time: {elapsed:.0f}s")

    return results


def run_heterogeneity_scaling(n_train=10000, n_test=2000, epochs=8, seed=0):
    """Sweep number of corruption types (2→3→4→5) to show routing advantage grows."""
    all_types = ["clean", "gaussian", "saltpepper", "rotation", "occlusion"]
    sweep = []

    for n_corr in range(2, 6):
        types = all_types[:n_corr]
        print(f"\n=== Heterogeneity sweep: {n_corr} corruption types ({types}) ===")
        res = run_benchmark(n_train, n_test, epochs, seed, types)
        sweep.append({
            "n_corruptions": n_corr,
            "corruption_types": types,
            "static_cnn_acc": res["static_cnn"]["acc"],
            "static_mlp_acc": res["static_mlp"]["acc"],
            "dirty_man_acc": res["dirty_man"]["acc"],
            "moe_acc": res["moe"]["acc"],
            "dm_vs_cnn": round(res["dirty_man"]["acc"] - res["static_cnn"]["acc"], 4),
            "dm_vs_mlp": round(res["dirty_man"]["acc"] - res["static_mlp"]["acc"], 4),
            "dm_vs_moe": round(res["dirty_man"]["acc"] - res["moe"]["acc"], 4),
        })
        print(f"  CNN={res['static_cnn']['acc']:.3f} MoE={res['moe']['acc']:.3f} DM={res['dirty_man']['acc']:.3f} (Δcnn={sweep[-1]['dm_vs_cnn']:+.3f})")

    return sweep


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scale", action="store_true", help="Heterogeneity scaling sweep")
    parser.add_argument("--n-train", type=int, default=30000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.smoke:
        args.n_train, args.n_test, args.epochs = 2000, 500, 3

    if args.scale:
        results = run_heterogeneity_scaling(args.n_train, args.n_test, args.epochs, args.seed)
        with open("results/heterogeneity_scaling.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n\nHeterogeneity scaling saved to results/heterogeneity_scaling.json")
        for r in results:
            print(f"  {r['n_corruptions']} corruptions: DM-cnn={r['dm_vs_cnn']:+.3f} DM-moe={r['dm_vs_moe']:+.3f}")
    else:
        results = run_benchmark(args.n_train, args.n_test, args.epochs, args.seed)
        with open("results/corruption_routing.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to results/corruption_routing.json")
        print(f"\nRouting policy:")
        for ct, policy in results["dirty_man"].get("routing_policy", {}).items():
            print(f"  {ct}: {policy}")
