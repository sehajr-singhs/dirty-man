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
          warmup=0.2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        tau = annealed_tau(ep, epochs)
        warm = routed and (ep < warmup * epochs)   # fixed-path warm-up
        tot = n = 0
        for xb, yb in make_batches(X, y):
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
    err = 0.0
    for xb, yb in make_batches(X, y, shuffle=False):
        err += F.mse_loss(model(xb, tau=0.5, hard=True), yb).item() * len(xb)
    return err / len(X)


@torch.no_grad()
def lens_profile(model, X, y):
    """Per-depth lens shares, and lens vs. speed (velocity magnitude)."""
    model.eval()
    rec = {}
    d1s, d2s, speeds = [], [], []
    for xb, yb in make_batches(X, y, bs=512, shuffle=False):
        r = {}
        model(xb, tau=0.5, hard=True, record=r)
        d1s.append(r["d1"]); d2s.append(r["d2"])
        speeds.append(xb[:, 7:14].norm(dim=1).numpy())
    d1 = np.concatenate(d1s); d2 = np.concatenate(d2s)
    sp = np.concatenate(speeds)
    return {
        "d1_shares": {OPS[i]: round(float((d1 == i).mean()), 3) for i in range(len(OPS))},
        "d2_shares": {OPS[i]: round(float((d2 == i).mean()), 3) for i in range(len(OPS))},
        "linear_lens_speed": round(float(sp[d1 == 0].mean()), 3),   # mean |v| when linear chosen
        "nonlinear_lens_speed": round(float(sp[d1 != 0].mean()), 3),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--routed-only", action="store_true",
                    help="skip linear regression + static grid; train routed only")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_sarcos()
    print(f"SARCOS {Xtr.shape[0]} train / {Xte.shape[0]} test "
          f"(real 7-DOF robot arm telemetry)", flush=True)

    prev = {}
    prev_path = os.path.join(RESULTS, "sarcos_routing.json")
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)

    results = {"experiment": "sarcos_routing",
               "data": "SARCOS real robot arm inverse dynamics (44,484 train / 4,449 test)",
               "metric": "normalized MSE on 7 joint torques (lower better)"}

    if not args.routed_only:
        # linear regression baseline
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression().fit(Xtr, ytr)
        lin_mse = float(np.mean((lr.predict(Xte) - yte) ** 2))
        results["linear_regression"] = {"test_nmse": round(lin_mse, 5)}
        print(f"[linear regression] test_nmse {lin_mse:.5f}", flush=True)

        # best fixed path (grid over the 9 static paths)
        best = None
        for p1 in OPS:
            for p2 in OPS:
                m = StaticDynamics(path=(p1, p2))
                r = train(m, Xtr, ytr, Xte, yte, epochs=15, name=f"static {p1},{p2}",
                          routed=False)
                if best is None or r["test_nmse"] < best[1]:
                    best = (f"{p1},{p2}", r["test_nmse"], r["params"])
        results["static_best"] = {"path": best[0], "test_nmse": best[1],
                                  "params": best[2]}
        print(f"[static best] path={best[0]} test_nmse {best[1]:.5f}", flush=True)
    else:
        results["linear_regression"] = prev.get("linear_regression", {})
        results["static_best"] = prev.get("static_best", {})
        print("[routed-only] reused static baselines from existing JSON", flush=True)

    # the routed program
    net = RoutedDynamics()
    r = train(net, Xtr, ytr, Xte, yte, epochs=30, name="routed", routed=True)
    results["routed"] = r
    results["lens_profile"] = lens_profile(net, Xte, yte)
    print(f"[routed] test_nmse {r['test_nmse']} params {r['params']}", flush=True)
    print(f"[lens profile] {json.dumps(results['lens_profile'])}", flush=True)

    with open(os.path.join(RESULTS, "sarcos_routing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote results/sarcos_routing.json  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
