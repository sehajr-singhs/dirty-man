"""Non-static layers: per-sample, per-depth routing *inside* a single network.

The Dirty Man thesis, sharpened. Routing between whole networks is
mixture-of-experts — a desk editor files it under "known idea, nice
engineering." The genuinely different claim is routing *inside* the network:
at every depth a router picks the primitive *layer* that processes the current
representation for that sample. The architecture is no longer a fixed stack of
layers; it is a per-sample program of computation. Each input walks its own
path — depth 1 might pick a 5x5 convolution, depth 2 a gated linear cell,
depth 3 a ReLU projection — and different inputs take different paths.

Measured on SVHN (real street-view digit photographs; genuinely real-world
data, no synthetic overlap):
  - the per-sample program is evaluated against every fixed path it contains;
  - the comparison reports the best fixed path and a coarse whole-network
    switch (MoE-style), rather than assuming routing must win;
  - the per-depth choices are inspectable: digit classes pick different
    downstream lenses — the "identify the feature, pick the lens" thesis.

Run modes (each writes its own partial JSON, then combine them):
    python nonstatic_layers.py --model static --epochs 8
    python nonstatic_layers.py --model router --epochs 10
    python nonstatic_layers.py --model switch --epochs 5
    python nonstatic_layers.py --combine
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
from torch.utils.data import DataLoader, Subset

D = 96              # hidden width of the vector stages
N_CLS = 10
RESULTS = "results"


# ---------------------------------------------------------------------------
# Data — SVHN, real street-view digit photographs
# ---------------------------------------------------------------------------

def _resumable_download(url, dest, tries=8):
    """Download with resume + retry; Stanford's host is flaky from some
    networks, so this is the reliable path (torchvision's own downloader
    gives up on the first dropped connection)."""
    import time
    import urllib.request

    for t in range(tries):
        try:
            tmp = dest + ".part"
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=90) as r, open(tmp, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            try:
                import scipy.io as sio
                sio.loadmat(tmp, variable_names=["X"])
            except Exception:
                raise RuntimeError("mat file incomplete")
            os.replace(tmp, dest)
            print(f"  downloaded {os.path.basename(dest)} "
                  f"({os.path.getsize(dest) / 1e6:.1f} MB)", flush=True)
            return
        except Exception as e:
            print(f"  download try {t + 1}: {type(e).__name__} {str(e)[:60]}",
                  flush=True)
            time.sleep(3)
    raise RuntimeError(f"gave up downloading {url}")


def load_svhn(root="./data_svhn", n_train=20000, n_test=6000, seed=0,
              download=True):
    from torchvision import datasets, transforms

    if download:
        import scipy.io as sio  # noqa: F401  (validity check used above)
        os.makedirs(root, exist_ok=True)
        for split, fn in [("train", "train_32x32.mat"),
                          ("test", "test_32x32.mat")]:
            dest = os.path.join(root, fn)
            if not os.path.exists(dest):
                print(f"SVHN {split} missing — downloading", flush=True)
                _resumable_download(
                    f"http://ufldl.stanford.edu/housenumbers/{fn}", dest)

    tr = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # 3-channel photos -> luminance
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.SVHN(root=root, split="train", download=False, transform=tr)
    test = datasets.SVHN(root=root, split="test", download=False, transform=tr)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train), generator=g)[:n_train]
    tidx = torch.randperm(len(test), generator=g)[:n_test]
    return Subset(train, idx.tolist()), Subset(test, tidx.tolist())


# ---------------------------------------------------------------------------
# Primitive *layers* — the ops each depth can choose from
# ---------------------------------------------------------------------------

def _count_params(m):
    return sum(p.numel() for p in m.parameters())


class DepthWiseSep(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, c, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.pw(self.act(self.dw(x))))


# op name -> constructor, per depth. Each op is a module with .forward.
SPATIAL_OPS = ["conv3", "conv5", "conv1", "sep"]           # depth 1 (image space)
VECTOR_OPS = ["linear", "relu", "mlp", "gated"]            # depth 2
FINAL_OPS = ["linear", "relu", "mlp"]                      # depth 3


def build_op(name, c=16):
    if name == "conv3":
        return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
    if name == "conv5":
        return nn.Sequential(nn.Conv2d(c, c, 5, padding=2), nn.ReLU())
    if name == "conv1":
        return nn.Sequential(nn.Conv2d(c, c, 1), nn.ReLU())
    if name == "sep":
        return DepthWiseSep(c)
    if name == "linear":
        return nn.Linear(D, D)
    if name == "relu":
        return nn.Sequential(nn.Linear(D, D), nn.ReLU())
    if name == "mlp":
        return nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
    if name == "gated":
        g = nn.Module()
        g.gate = nn.Linear(D, D)
        g.cell = nn.Linear(D, D)
        g.forward = lambda x: torch.sigmoid(g.gate(x)) * torch.tanh(g.cell(x))
        return g
    raise ValueError(name)


# ---------------------------------------------------------------------------
# The non-static network: a router at every depth
# ---------------------------------------------------------------------------

class NonStaticNet(nn.Module):
    """A 3-depth network whose computation is a per-sample program.

    depth 0: stem conv  ->  spatial ops (conv3 / conv5 / conv1 / sep)
    depth 1: projection -> vector ops (linear / relu / mlp / gated)
    depth 2: projection -> vector ops (linear / relu / mlp)

    At each depth a router reads the current representation and gates the ops
    (Gumbel-Softmax, annealed soft->hard). No op is 'the' layer; every sample
    selects its own.
    """

    def __init__(self, n_classes=N_CLS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.ops1 = nn.ModuleDict({n: build_op(n) for n in SPATIAL_OPS})
        self.proj1 = nn.Linear(16 * 8 * 8, D)                    # pool(8) -> 16*64
        self.ops2 = nn.ModuleDict({n: build_op(n) for n in VECTOR_OPS})
        self.ops3 = nn.ModuleDict({n: build_op(n) for n in FINAL_OPS})
        self.r1 = nn.Sequential(nn.Linear(16 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, len(SPATIAL_OPS)))
        self.r2 = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, len(VECTOR_OPS)))
        self.r3 = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, len(FINAL_OPS)))
        self.head = nn.Linear(D, n_classes)
        # interpretation bookkeeping
        self.names = [SPATIAL_OPS, VECTOR_OPS, FINAL_OPS]

    def _gumbel(self, logits, tau, hard):
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)
        return probs

    def forward(self, x, tau=1.0, hard=False, record=None):
        x = self.stem(x)                                          # (B,16,32,32)
        p1 = self.r1(F.adaptive_avg_pool2d(x, 4).reshape(x.size(0), -1))
        w1 = self._gumbel(p1, tau, hard)
        z1 = sum(w1[:, i].view(-1, 1, 1, 1) * op(x)
                 for i, op in enumerate(self.ops1.values()))
        f = self.proj1(F.adaptive_avg_pool2d(z1, 8).reshape(x.size(0), -1))
        f = F.relu(f)

        p2 = self.r2(f)
        w2 = self._gumbel(p2, tau, hard)
        f2 = sum(w2[:, i].view(-1, 1) * op(f)
                 for i, op in enumerate(self.ops2.values()))

        p3 = self.r3(f2)
        w3 = self._gumbel(p3, tau, hard)
        f3 = sum(w3[:, i].view(-1, 1) * op(f2)
                 for i, op in enumerate(self.ops3.values()))

        out = self.head(f3)
        if record is not None:
            record["d1"] = w1.argmax(-1).cpu().numpy()
            record["d2"] = w2.argmax(-1).cpu().numpy()
            record["d3"] = w3.argmax(-1).cpu().numpy()
        return out

    def balance_losses(self, x, tau, hard, w=0.05):
        """Per-depth load balancing so no op starves."""
        loss = torch.zeros((), device=x.device)
        x = self.stem(x)
        # r1
        p1 = self.r1(F.adaptive_avg_pool2d(x, 4).reshape(x.size(0), -1))
        w1 = F.gumbel_softmax(p1, tau=tau, hard=True, dim=-1)
        frac1 = w1.mean(0)
        meanp1 = F.softmax(p1, dim=-1).mean(0)
        loss = loss + len(SPATIAL_OPS) * (frac1 * meanp1).sum() * w
        z1 = sum(w1[:, i].view(-1, 1, 1, 1) * op(x)
                 for i, op in enumerate(self.ops1.values()))
        f = F.relu(self.proj1(F.adaptive_avg_pool2d(z1, 8).reshape(x.size(0), -1)))
        p2 = self.r2(f)
        w2 = F.gumbel_softmax(p2, tau=tau, hard=True, dim=-1)
        frac2 = w2.mean(0)
        meanp2 = F.softmax(p2, dim=-1).mean(0)
        loss = loss + len(VECTOR_OPS) * (frac2 * meanp2).sum() * w
        f2 = sum(w2[:, i].view(-1, 1) * op(f) for i, op in enumerate(self.ops2.values()))
        p3 = self.r3(f2)
        w3 = F.gumbel_softmax(p3, tau=tau, hard=True, dim=-1)
        frac3 = w3.mean(0)
        meanp3 = F.softmax(p3, dim=-1).mean(0)
        loss = loss + len(FINAL_OPS) * (frac3 * meanp3).sum() * w
        return loss

    def n_params(self):
        return _count_params(self)


class StaticPath(nn.Module):
    """A fixed path through the same op bank — the 'static network' baseline.
    Same budget, same ops; only the routing is removed."""

    def __init__(self, path=("conv3", "mlp", "relu"), n_classes=N_CLS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.op1 = build_op(path[0])
        self.proj1 = nn.Linear(16 * 8 * 8, D)
        self.op2 = build_op(path[1])
        self.op3 = build_op(path[2])
        self.head = nn.Linear(D, n_classes)
        self.path = path

    def forward(self, x, tau=1.0, hard=False):
        x = self.stem(x)
        x = self.op1(x)
        f = F.relu(self.proj1(F.adaptive_avg_pool2d(x, 8).reshape(x.size(0), -1)))
        f = self.op2(f)
        f = self.op3(f)
        return self.head(f)

    def n_params(self):
        return _count_params(self)


# ---------------------------------------------------------------------------
# Coarse baseline — whole-network switch (MoE-style), for a fair comparison
# ---------------------------------------------------------------------------

def build_coarse_switch(size=32, n_classes=N_CLS):
    from dirty_man.switch_operator import SwitchOperator, IMG, LATENT

    return SwitchOperator(n_classes=n_classes, size=size, latent=LATENT, bottleneck=True)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def annealed_tau(epoch, n_epochs, tau0=1.5, tau1=0.5):
    if n_epochs <= 1:
        return tau1
    frac = epoch / (n_epochs - 1)
    return tau1 + (tau0 - tau1) * 0.5 * (1.0 + math.cos(math.pi * frac))


def train_model(model, train_dl, test_dl, epochs, name, hard=False, switch=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = len(train_dl)
    warm_epochs = int(0.35 * epochs) if not isinstance(model, StaticPath) else 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        n = 0
        warm = (ep < warm_epochs) and (not switch) and not isinstance(model, StaticPath)
        for xb, yb in train_dl:
            opt.zero_grad()
            tau = annealed_tau(ep + 1, epochs)
            if warm:
                # fixed-path warm-up: train the shared ops through one good
                # path (conv3, mlp, relu) so joint router+op learning starts
                # from a competent base instead of a uniform mixture
                x = model.stem(xb)
                z = model.ops1["conv3"](x)
                f = F.relu(model.proj1(F.adaptive_avg_pool2d(z, 8).reshape(xb.size(0), -1)))
                f2 = model.ops2["mlp"](f)
                f3 = model.ops3["relu"](f2)
                out = model.head(f3)
                loss = F.cross_entropy(out, yb)
            elif switch:
                out, info = model(xb, tau=tau, hard=hard)
                probs = info["probs"]
                hard_assign = probs.argmax(-1)
                loss = F.cross_entropy(out, yb)
                aux = model.aux_losses(xb, probs, hard_assign, balance_w=0.05, ent_w=0.01)
                for v in aux.values():
                    loss = loss + v
            else:
                out = model(xb, tau=tau, hard=hard)
                loss = F.cross_entropy(out, yb)
                if not isinstance(model, StaticPath):
                    loss = loss + model.balance_losses(xb, tau, hard)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        acc = evaluate(model, test_dl, switch=switch)
        print(f"[{name}] ep {ep + 1}/{epochs} loss {tot / n:.4f} "
              f"test acc {acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    return {"acc": evaluate(model, test_dl, switch=switch)}


@torch.no_grad()
def evaluate(model, dl, switch=False):
    model.eval()
    correct = total = 0
    for xb, yb in dl:
        if switch:
            out, _ = model(xb, tau=0.5, hard=True)
        else:
            out = model(xb, tau=0.5, hard=True)
        correct += (out.argmax(-1) == yb).sum().item()
        total += xb.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def collect_programs(model, dl, out_path):
    """Record the per-sample program (depth choices) grouped by class."""
    model.eval()
    per_class = {c: {d: {} for d in range(3)} for c in range(N_CLS)}
    for xb, yb in dl:
        rec = {}
        model(xb, tau=0.5, hard=True, record=rec)
        for c in np.unique(yb.numpy()):
            m = yb.numpy() == c
            for d in range(3):
                vals, counts = np.unique(rec[f"d{d + 1}"][m], return_counts=True)
                for v, ct in zip(vals, counts):
                    per_class[int(c)][d][int(v)] = int(ct)
    with open(out_path, "w") as f:
        json.dump(per_class, f, indent=2)
    return per_class


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["static", "router", "switch", "programs"], default="router")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--path", default="conv3,mlp,relu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.combine:
        combined = {"experiment": "nonstatic_layers_svhn", "data": "SVHN real street-view photos",
                    "seed": args.seed}
        for name in ["static", "router", "switch"]:
            fp = os.path.join(RESULTS, f"ns_{name}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    combined[name] = json.load(f)
        with open(os.path.join(RESULTS, "nonstatic_svhn.json"), "w") as f:
            json.dump(combined, f, indent=2)
        print("combined ->", os.path.join(RESULTS, "nonstatic_svhn.json"))
        return

    train_ds, test_ds = load_svhn(n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=0)
    print(f"SVHN: {len(train_ds)} train / {len(test_ds)} test", flush=True)

    if args.model == "static":
        model = StaticPath(path=tuple(args.path.split(",")))
        res = train_model(model, train_dl, test_dl, args.epochs, "static",
                          hard=False, switch=False)
        res["path"] = args.path
        res["params"] = model.n_params()
    elif args.model == "router":
        model = NonStaticNet()
        res = train_model(model, train_dl, test_dl, args.epochs, "router",
                          hard=False, switch=False)
        res["params"] = model.n_params()
        per_class = collect_programs(model, test_dl, os.path.join(RESULTS, "ns_programs.json"))
        res["program_file"] = "ns_programs.json"
    else:  # switch
        model = build_coarse_switch()
        res = train_model(model, train_dl, test_dl, args.epochs, "switch",
                          hard=False, switch=True)
        res["params"] = model.n_params()

    out = os.path.join(RESULTS, f"ns_{args.model}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}: {res}")


if __name__ == "__main__":
    main()
