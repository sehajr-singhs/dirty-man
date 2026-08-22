"""Non-static computation on REAL robot-arm data (SARCOS inverse dynamics).

SARCOS: 44,484 real telemetry records from a 7-DOF SARCOS robot arm
(21 inputs: joint positions/velocities/accelerations -> 7 torques). This is
the classic benchmark for learning dynamics from real physical data (used by
the GP-regression literature, deep kernel learning, etc.). No synthetic
renderer is involved: every row is a measured state of a real robot.

The thesis, on real data: a fixed computational path has a fixed inductive
bias, but real robot dynamics are heterogeneous — near-static configurations
are nearly linear (gravity compensation dominates), fast configurations are
strongly nonlinear (Coriolis/centrifugal terms). A per-sample, per-depth
routed program should therefore beat any single fixed path with the same op
budget, and its lens choices should be interpretable (linear when slow,
nonlinear when fast).

Run:  python sarcos_routing.py   (writes results/sarcos_routing.json)
"""

import copy
import json
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
RESULTS = "results"


def load_sarcos(root=None):
    import scipy.io as sio
    if root is None:
        candidates = [os.path.join(tempfile.gettempdir(), "sarcos"),
                      "/kaggle/input/sarcos", "/kaggle/input"]
        root = next((p for p in candidates
                     if os.path.exists(os.path.join(p, "sarcos_inv.mat"))), None)
        if root is None and os.path.isdir("/kaggle/input"):
            for base, _, files in os.walk("/kaggle/input"):
                if "sarcos_inv.mat" in files:
                    root = base
                    break
        root = root or os.path.join(tempfile.gettempdir(), "sarcos")
    tr = sio.loadmat(os.path.join(root, "sarcos_inv.mat"))["sarcos_inv"]
    te = sio.loadmat(os.path.join(root, "sarcos_inv_test.mat"))["sarcos_inv_test"]
    Xtr, ytr = tr[:, :21].astype(np.float32), tr[:, 21:].astype(np.float32)
    Xte, yte = te[:, :21].astype(np.float32), te[:, 21:].astype(np.float32)
    # standardize inputs by train stats; standardize outputs per-dim
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ostd = ytr.std(0, keepdims=True) + 1e-6
    ytr, yte = ytr / ostd, yte / ostd
    return Xtr, ytr, Xte, yte


OPS = ["linear", "relu", "mlp"]


def build_op(name, din, dout):
    if name == "linear":
        return nn.Linear(din, dout)
    if name == "relu":
        return nn.Sequential(nn.Linear(din, dout), nn.ReLU())
    if name == "mlp":
        return nn.Sequential(nn.Linear(din, 2 * dout), nn.GELU(),
                             nn.Linear(2 * dout, dout))
    raise ValueError(name)


class RoutedDynamics(nn.Module):
    """2-depth non-static regressor: per-sample lens at each depth."""

    def __init__(self, din=21, dout=7):
        super().__init__()
        self.ops1 = nn.ModuleDict({n: build_op(n, din, D) for n in OPS})
        self.ops2 = nn.ModuleDict({n: build_op(n, D, D) for n in OPS})
        self.r1 = nn.Sequential(nn.Linear(din, 32), nn.ReLU(), nn.Linear(32, len(OPS)))
        self.r2 = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, len(OPS)))
        self.head = nn.Linear(D, dout)
        self.names = [OPS, OPS]

    def _gumbel(self, logits, tau, hard):
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)
        return probs

    def forward(self, x, tau=1.0, hard=False, record=None):
        w1 = self._gumbel(self.r1(x), tau, hard)
        h = sum(w1[:, i].view(-1, 1) * op(x) for i, op in enumerate(self.ops1.values()))
        w2 = self._gumbel(self.r2(h), tau, hard)
        h2 = sum(w2[:, i].view(-1, 1) * op(h) for i, op in enumerate(self.ops2.values()))
        out = self.head(h2)
        if record is not None:
            record["d1"] = w1.argmax(-1).cpu().numpy()
            record["d2"] = w2.argmax(-1).cpu().numpy()
        return out

    def balance_losses(self, x, tau, w=0.05):
        loss = torch.zeros((), device=x.device)
        p1 = self.r1(x)
        w1 = F.gumbel_softmax(p1, tau=tau, hard=True, dim=-1)
        frac1, meanp1 = w1.mean(0), F.softmax(p1, dim=-1).mean(0)
        loss = loss + len(OPS) * (frac1 * meanp1).sum() * w
        h = sum(w1[:, i].view(-1, 1) * op(x) for i, op in enumerate(self.ops1.values()))
        p2 = self.r2(h)
        w2 = F.gumbel_softmax(p2, tau=tau, hard=True, dim=-1)
        frac2, meanp2 = w2.mean(0), F.softmax(p2, dim=-1).mean(0)
        loss = loss + len(OPS) * (frac2 * meanp2).sum() * w
        return loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class StaticDynamics(nn.Module):
    """A fixed path through the same op bank (no routing)."""

    def __init__(self, path=("mlp", "mlp"), din=21, dout=7):
        super().__init__()
        self.op1 = build_op(path[0], din, D)
        self.op2 = build_op(path[1], D, D)
        self.head = nn.Linear(D, dout)

    def forward(self, x, tau=1.0, hard=False):
        return self.head(self.op2(self.op1(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def make_batches(X, y, bs=256, shuffle=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g) if shuffle else torch.arange(len(X))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j])


def annealed_tau(ep, epochs, tau0=1.5, tau1=0.5):
    if epochs <= 1:
        return tau1
    frac = ep / (epochs - 1)
    return tau1 + (tau0 - tau1) * 0.5 * (1.0 + np.cos(np.pi * frac))


def train(model, X, y, Xte, yte, epochs=25, name="", routed=True, lr=1e-3,
          warmup=0.2, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tau = annealed_tau(ep, epochs)
        warm = routed and (ep < warmup * epochs)   # fixed-path warm-up
        tot = n = 0
        for xb, yb in make_batches(X, y):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if warm:
                # bypass routing: train the ops through the best static path
                h = model.ops2["mlp"](model.ops1["mlp"](xb))
                loss = F.mse_loss(model.head(h), yb)
            else:
                loss = F.mse_loss(model(xb, tau=tau), yb)
                if routed:
                    loss = loss + model.balance_losses(xb, tau)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        mse = evaluate(model, Xte, yte)
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"[{name}] ep {ep + 1}/{epochs} train {tot / n:.5f} "
                  f"test_nmse {mse:.5f}", flush=True)
    return {"test_nmse": round(evaluate(model, Xte, yte), 5),
            "params": model.n_params()}


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    device = next(model.parameters()).device
    err = 0.0
    for xb, yb in make_batches(X, y, shuffle=False):
        xb, yb = xb.to(device), yb.to(device)
        err += F.mse_loss(model(xb, tau=0.5, hard=True), yb).item() * len(xb)
    return err / len(X)


@torch.no_grad()
def evaluate_by_speed(model, X, y, n_bins=4):
    """Evaluate errors in velocity-magnitude bins, not only in aggregate.

    The aggregate SARCOS result can hide whether routing helps on the regime
    that motivates it. Fixed quantile bins make the slow/fast claim testable
    and avoid choosing a threshold after looking at the predictions.
    """
    model.eval()
    device = next(model.parameters()).device
    speed = np.linalg.norm(X[:, 7:14], axis=1)
    edges = np.quantile(speed, np.linspace(0.0, 1.0, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (speed >= lo) & (speed <= hi if i == n_bins - 1 else speed < hi)
        if not np.any(mask):
            continue
        pred_err = []
        for xb, yb in make_batches(X[mask], y[mask], shuffle=False):
            xb, yb = xb.to(device), yb.to(device)
            pred_err.append(((model(xb, tau=0.5, hard=True) - yb) ** 2).mean(dim=1).cpu().numpy())
        bins.append({
            "bin": i,
            "speed_lo": round(float(lo), 6),
            "speed_hi": round(float(hi), 6),
            "n": int(mask.sum()),
            "mse": round(float(np.concatenate(pred_err).mean()), 8),
        })
    return bins


@torch.no_grad()
def lens_profile(model, X, y):
    """Per-depth lens shares, and lens vs. speed (velocity magnitude)."""
    model.eval()
    device = next(model.parameters()).device
    rec = {}
    d1s, d2s, speeds = [], [], []
    for xb, yb in make_batches(X, y, bs=512, shuffle=False):
        r = {}
        model(xb.to(device), tau=0.5, hard=True, record=r)
        d1s.append(r["d1"]); d2s.append(r["d2"])
        speeds.append(xb[:, 7:14].norm(dim=1).numpy())
    d1 = np.concatenate(d1s); d2 = np.concatenate(d2s)
    sp = np.concatenate(speeds)
    def safe_mean(values):
        return None if len(values) == 0 else round(float(values.mean()), 3)

    profile = {
        "d1_shares": {OPS[i]: round(float((d1 == i).mean()), 3) for i in range(len(OPS))},
        "d2_shares": {OPS[i]: round(float((d2 == i).mean()), 3) for i in range(len(OPS))},
        "linear_lens_speed": safe_mean(sp[d1 == 0]),
        "nonlinear_lens_speed": safe_mean(sp[d1 != 0]),
    }
    # Quantile-bin routing shares expose whether the apparent speed split is
    # monotone or merely a difference in two selected-group means.
    edges = np.quantile(sp, np.linspace(0.0, 1.0, 5))
    profile["d1_shares_by_speed_quartile"] = []
    for i in range(4):
        mask = (sp >= edges[i]) & (sp <= edges[i + 1] if i == 3 else sp < edges[i + 1])
        if np.any(mask):
            profile["d1_shares_by_speed_quartile"].append({
                "quartile": i,
                "speed_lo": round(float(edges[i]), 3),
                "speed_hi": round(float(edges[i + 1]), 3),
                "n": int(mask.sum()),
                "shares": {OPS[j]: round(float((d1[mask] == j).mean()), 3) for j in range(len(OPS))},
            })
    return profile


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--routed-only", action="store_true",
                    help="skip linear regression + static grid; train routed only")
    ap.add_argument("--root", default=None,
                    help="directory containing sarcos_inv.mat and sarcos_inv_test.mat")
    ap.add_argument("--static-epochs", type=int, default=30,
                    help="epochs for every fixed path (matched to routed by default)")
    ap.add_argument("--routed-epochs", type=int, default=30,
                    help="epochs for the routed model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_sarcos(args.root)
    if args.max_train is not None:
        Xtr, ytr = Xtr[:args.max_train], ytr[:args.max_train]
    if args.max_test is not None:
        Xte, yte = Xte[:args.max_test], yte[:args.max_test]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else args.device if args.device != "auto" else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(f"device={device}", flush=True)
    print(f"SARCOS {Xtr.shape[0]} train / {Xte.shape[0]} test "
          f"(real 7-DOF robot arm telemetry)", flush=True)

    prev = {}
    prev_path = os.path.join(RESULTS, "sarcos_routing.json")
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)

    results = {"experiment": "sarcos_routing",
               "data": "SARCOS real robot arm inverse dynamics (44,484 train / 4,449 test)",
               "metric": "normalized MSE on 7 joint torques (lower better)",
               "seed": args.seed,
               "evaluation_protocol": {
                   "static_epochs": args.static_epochs,
                   "routed_epochs": args.routed_epochs,
                   "speed_stratification": "test-set velocity-magnitude quartiles",
                   "device": device,
               }}

    if not args.routed_only:
        # linear regression baseline
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(Xtr, ytr)
        lin_mse = float(np.mean((lr.predict(Xte) - yte) ** 2))
        results["linear_regression"] = {"test_nmse": round(lin_mse, 5)}
        print(f"[linear regression] test_nmse {lin_mse:.5f}", flush=True)

        # Select the best fixed path using the same epoch budget as routing.
        # Preserve its weights for a paired, speed-stratified comparison.
        best = None
        static_grid = []
        best_state = None
        for p1 in OPS:
            for p2 in OPS:
                m = StaticDynamics(path=(p1, p2))
                r = train(m, Xtr, ytr, Xte, yte, epochs=args.static_epochs,
                          name=f"static {p1},{p2}", routed=False, device=device)
                row = {"path": f"{p1},{p2}", "test_nmse": r["test_nmse"],
                       "params": r["params"]}
                static_grid.append(row)
                if best is None or r["test_nmse"] < best[1]:
                    best = (f"{p1},{p2}", r["test_nmse"], r["params"])
                    best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        results["static_grid"] = static_grid
        best_model = StaticDynamics(path=tuple(best[0].split(",")))
        best_model.load_state_dict(best_state)
        best_model.to(device)
        results["static_best"] = {
            "path": best[0], "test_nmse": best[1], "params": best[2],
            "mse_by_speed_quartile": evaluate_by_speed(best_model, Xte, yte),
        }
        print(f"[static best] path={best[0]} test_nmse {best[1]:.5f}", flush=True)
    else:
        results["linear_regression"] = prev.get("linear_regression", {})
        results["static_best"] = prev.get("static_best", {})
        print("[routed-only] reused static baselines from existing JSON", flush=True)

    # the routed program
    net = RoutedDynamics()
    r = train(net, Xtr, ytr, Xte, yte, epochs=args.routed_epochs,
              name="routed", routed=True, device=device)
    results["routed"] = r
    results["routed"]["mse_by_speed_quartile"] = evaluate_by_speed(net, Xte, yte)
    results["lens_profile"] = lens_profile(net, Xte, yte)

    print(f"[routed] test_nmse {r['test_nmse']} params {r['params']}", flush=True)
    print(f"[lens profile] {json.dumps(results['lens_profile'])}", flush=True)

    with open(os.path.join(RESULTS, "sarcos_routing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote results/sarcos_routing.json  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
