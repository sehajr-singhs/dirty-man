"""Run the DirtyMan experiment suite.

Four protocols, every paper number measured from these runs:

  A  Mixed-domain specialization — the operator discovers per-domain structure
     (sim vs real) with zero domain labels, beating every single primitive and
     the uniform ensemble.
  B  Sim-to-real transfer — trained on sim only, tested on real; and the
     structure-adaptation vs weight-adaptation shootout on a tiny labeled real
     set.
  C  Goal pathway — the operator rewires its computation for the task it is
     trying to achieve (classify vs reconstruct).
  D  Ablations — anneal, router, bottleneck, eye, domain-invariance, uniform.

Training uses a three-stage protocol (see train_model): warm-start the
primitive bank under uniform routing, train the router with the primitives
frozen (annealed soft->hard Gumbel), then fine-tune everything jointly.
This is what keeps the operator from collapsing onto its most expressive
primitive before it has learned anything.

Every protocol checkpoints per seed, so a killed run resumes where it left
off instead of restarting. JSON writes are atomic (tmp + rename).

Usage:
    python run_experiments.py                          # everything, 3 seeds
    python run_experiments.py --only A --seeds 3 --epochs 12
    python run_experiments.py --only B --seed-start 1 --seeds 1   # just seed 1
    python run_experiments.py --smoke                  # tiny, fast check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirty_man.data_glyphs import make_datasets, save_playground_data
from dirty_man.switch_operator import (IMG, PRIMITIVES, Standalone, SwitchOperator,
                                       annealed_tau, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# On CPU boxes, torch thrashes on tiny ops with many threads; single-thread
# gives 6-35x speedups for the small networks here (harmless when on GPU).
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FLAT = IMG * IMG            # 576 for 24x24 inputs
VERSION = "v6"              # bump when data or training changes; invalidates checkpoints

# Specialist niches for the primitive bank (regimes: 0=sim, 1=spatial, 2=stat).
# Flat lenses never see spatial corruption; spatial lenses never see
# statistical corruption — so each expert is *visibly* the wrong tool outside
# its niche, which is what makes the operator's routing learnable.
SPECIALISTS = {
    "linear": [0, 2], "dense": [0, 2], "relu": [0, 2],
    "cnn": [0, 1], "autoencoder": [0, 1],
    "rnn": [0], "lstm": [0], "transformer": [0], "gan": [0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_device(*ts):
    """Move collated tensors to the compute device (no-op on CPU)."""
    return tuple(t.to(DEVICE) for t in ts)


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    d = torch.stack([b[2] for b in batch])
    s = torch.stack([b[3] for b in batch])
    r = torch.stack([b[4] for b in batch])
    return to_device(x, y, d, s, r)


def make_loader(ds, batch=128, shuffle=True):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate)


def freeze(module: nn.Module, freeze_: bool):
    for p in module.parameters():
        p.requires_grad = not freeze_


def atomic_write(path: str, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# per-seed checkpoints: config + version are in the filename so stale
# checkpoints from a different run configuration are never reused.
def chk_path(name, seed, epochs, n_train):
    return os.path.join(OUT, f"chk_{name}_{VERSION}_e{epochs}_t{n_train}_s{seed}.json")


def load_chk(name, seed, epochs, n_train):
    p = chk_path(name, seed, epochs, n_train)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_chk(name, seed, epochs, n_train, data):
    atomic_write(chk_path(name, seed, epochs, n_train), data)
    print(f"  [seed {seed}] checkpointed", flush=True)


def with_chk(name, seed, epochs, n_train, key, fn, tag):
    """Run fn() and checkpoint its result under (name, key, seed). On a rerun,
    return the cached result instead. `tag` is a short label for logging."""
    chk = load_chk(f"{name}_{key}", seed, epochs, n_train)
    if chk is not None:
        print(f"  [seed {seed}] {tag}: cached, skipping", flush=True)
        return chk, False
    t0 = time.time()
    data = fn()
    save_chk(f"{name}_{key}", seed, epochs, n_train, data)
    print(f"  [seed {seed}] {tag}: done ({time.time() - t0:.0f}s)", flush=True)
    return data, True


def eval_cls(model, ds, tau=0.5, goal=None, uniform=False):
    model.eval()
    model.force_uniform = uniform
    loader = make_loader(ds, batch=256, shuffle=False)
    correct, total = 0, 0
    sim_c, sim_t, real_c, real_t = 0, 0, 0, 0
    probs_all, sev_all, dom_all, argmax_all = [], [], [], []
    is_switch = hasattr(model, "route_logits")
    with torch.no_grad():
        for x, y, d, s, r in loader:
            if is_switch:
                out, info = model(x, goal=goal, tau=tau, hard=False)
                p = info["probs"]
            else:
                out = model(x)
                p = torch.full((x.size(0), 9), 1.0 / 9.0)
            acc = (out.argmax(-1) == y)
            correct += acc.sum().item()
            total += y.size(0)
            sim_c += acc[d == 0].sum().item(); sim_t += (d == 0).sum().item()
            real_c += acc[d == 1].sum().item(); real_t += (d == 1).sum().item()
            probs_all.append(p); sev_all.append(s); dom_all.append(d)
            argmax_all.append(p.argmax(-1))
    probs = torch.cat(probs_all); sev = torch.cat(sev_all)
    dom = torch.cat(dom_all); argmax = torch.cat(argmax_all)
    return {
        "acc": correct / max(total, 1),
        "acc_sim": sim_c / max(sim_t, 1),
        "acc_real": real_c / max(real_t, 1),
        "probs": probs, "severity": sev, "domain": dom, "argmax": argmax,
    }


def routing_stats(res: dict, n_experts: int = 9):
    probs, sev, dom, argmax = res["probs"], res["severity"], res["domain"], res["argmax"]
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1).mean().item()
    util = {PRIMITIVES[i]: round(float((argmax == i).float().mean()), 4)
            for i in range(n_experts)}
    # switching rate on the severity-sorted sequence: how often the chosen
    # structure changes as the world gets dirtier.
    order = sev.argsort()
    seq = argmax[order]
    switch_rate = float((seq[1:] != seq[:-1]).float().mean())
    # per-domain routing distribution
    dom_routing = {}
    for d, name in [(0, "sim"), (1, "real")]:
        m = dom == d
        if m.sum() > 0:
            frac = torch.bincount(argmax[m], minlength=n_experts).float()
            frac = frac / frac.sum()
            dom_routing[name] = {PRIMITIVES[i]: round(float(v), 4)
                                 for i, v in enumerate(frac)}
    # gating vs severity trajectory (mean probs per severity bin)
    bins = torch.linspace(0, 1, 7)
    traj = []
    for i in range(6):
        m = (sev >= bins[i]) & (sev <= bins[i + 1])
        if m.sum() > 0:
            traj.append({"bin": round(float(bins[i]), 2),
                         "probs": probs[m].mean(0).tolist()})
    return {"entropy": entropy, "utilization": util, "switch_rate": switch_rate,
            "domain_routing": dom_routing, "trajectory": traj}


def save(name: str, data):
    atomic_write(os.path.join(OUT, name), data)
    print(f"  wrote {os.path.join(OUT, name)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_phase(model, train_ds, epochs, seed, lr, task, domain_w, goals,
                verbose, tag, t0, history, use_aux, anneal, uniform,
                switch_w=0.4, balance_w=0.2, batch=128, oracle=None,
                oracle_w=0.5, oracle_only=False, fixed_tau=None):
    """Train one stage for `epochs` epochs. Appends to and returns `history`.

    `oracle`: dict {primitive_name: standalone_head} used to supervise the
    router with the true per-sample best expert (argmin over each expert's own
    quality-judge head's CE) — the signal that turns routing into a learnable
    per-sample choice instead of a batch-average collapse.
    """
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=1e-5)
    loader = make_loader(train_ds, batch=batch)
    model.force_uniform = uniform
    for ep in range(epochs):
        model.train()
        if fixed_tau is not None:
            tau = fixed_tau
        else:
            tau = 1.0 if getattr(model, "fixed_tau", False) else (
                annealed_tau(ep, epochs) if anneal else 0.5)
        tot, n = 0.0, 0
        for x, y, d, s, r in loader:
            hard = anneal and ep >= epochs - 2
            if goals is not None:                      # multi-goal protocol C
                # two passes, one per goal, so each goal's head and its
                # routing see clean targets
                g0 = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                out0, info = model(x, goal=g0, tau=tau, hard=hard)
                g1 = torch.ones(x.size(0), dtype=torch.long, device=x.device)
                out1, _ = model(x, goal=g1, tau=tau, hard=hard)
                loss = (0.5 * F.cross_entropy(out0[:, :10], y)
                        + 0.5 * F.mse_loss(out1[:, :FLAT], x.reshape(x.size(0), -1)))
            elif oracle_only and oracle is not None:
                # routing-stack warm-up: learn the per-regime targets alone
                loss = torch.zeros((), device=x.device)
                out, info = model(x, goal=None, tau=tau, hard=hard)
            else:
                out, info = model(x, goal=None, tau=tau, hard=hard)
                if task == "rec":
                    loss = F.mse_loss(out, x.reshape(x.size(0), -1))
                else:
                    loss = F.cross_entropy(out, y)
            hard_assign = None if not use_aux else info["probs"].argmax(-1)
            aux = model.aux_losses(x, info["probs"], hard_assign,
                                   balance_w=balance_w,
                                   switch_w=switch_w if use_aux else 0.0)
            if use_aux and not oracle_only:
                loss = loss + aux["balance"] + aux["entropy"]
            loss = loss + aux.get("ae_recon", 0.0) + aux.get("gan_gen", 0.0) \
                + aux.get("gan_disc", 0.0)
            # regime-level oracle supervision for the router: each sample is
            # routed to the expert that is best on average for its regime
            # (sim / spatial / statistical). The per-sample argmin is noise;
            # the regime-level argmin is a stable, learnable target.
            if oracle is not None and goals is None and task == "cls":
                target = oracle[0][r]                # oracle[0]: (3,) best expert per regime
                loss = loss + oracle_w * F.cross_entropy(info["logits"], target)
            if domain_w > 0 and getattr(model, "with_domain_head", False):
                loss = loss + domain_w * model.domain_loss(x, d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        history.append(round(tot / max(n, 1), 4))
        if verbose:
            print(f"    [{tag}] ep {ep + 1}/{epochs} tau={tau:.2f} "
                  f"loss={history[-1]}  ({time.time() - t0:.0f}s)", flush=True)
    return history


def train_model(model, train_ds, epochs, seed, lr=1e-3, task="cls",
                domain_w=0.0, goals=None, verbose=True, tag="", staged=True,
                switch_w=0.4, balance_w=0.2, batch=128, bank=None, oracle_w=0.5):
    """Staged trainer for the Switch Operator.

    With a pre-trained bank (each primitive trained standalone on its niche):
      Stage 1 (head): primitives frozen; only the shared head warms up under
      uniform routing.
      Stage 2 (routewarm): eye + router learn the per-regime best-expert
      targets in isolation — this is what teaches the operator to tell a
      warped image from a quantized one before the task loss muddies it.
      Stage 3 (router): primitives frozen; the eye + router train on the task
      plus regime oracle + load-balancing and switching-pressure aux losses.
      Stage 4 (joint): everything fine-tunes together at a lower learning
      rate.

    Without a bank, the stages instead warm-start the primitives themselves
    under uniform routing so no primitive starves before it has learned
    anything.
    """
    set_seed(seed)
    model.to(DEVICE)
    is_switch = hasattr(model, "route_logits")
    force_uniform = getattr(model, "force_uniform", False)
    router_frozen = is_switch and all(not p.requires_grad
                                      for p in model.router.parameters())
    staged = staged and is_switch and not force_uniform and not router_frozen
    history = []
    t0 = time.time()

    oracle = None
    if bank is not None:
        for pname, sd in bank["prims"].items():
            model.primitives[pname].load_state_dict(sd)
        # the standalone heads double as per-expert quality judges: which
        # expert is best on average for each regime? (regimes: 0 sim,
        # 1 spatial, 2 statistical)
        oracle_heads = {}
        for pname, sd in bank["heads"].items():
            h = nn.Linear(64, 10)
            h.load_state_dict(sd)
            h.eval()
            oracle_heads[pname] = h
        model.eval()
        with torch.no_grad():
            x0, y0, _, _, r0 = next(iter(make_loader(train_ds, batch=1024)))
            n_reg = int(r0.max().item()) + 1
            ces_r = torch.zeros(n_reg, len(PRIMITIVES), device=x0.device)
            for i, nm in enumerate(PRIMITIVES):
                ce = F.cross_entropy(oracle_heads[nm](model.primitives[nm](x0)),
                                     y0, reduction="none")
                for reg in range(n_reg):
                    m = r0 == reg
                    if m.sum() > 0:
                        ces_r[reg, i] = ce[m].mean()
        best_per_regime = ces_r.argmin(-1)
        oracle = (best_per_regime, None)
        model.train()

    if not staged:
        train_phase(model, train_ds, epochs, seed, lr, task, domain_w, goals,
                    verbose, tag, t0, history,
                    use_aux=is_switch and not force_uniform,
                    anneal=is_switch and not force_uniform,
                    uniform=force_uniform, switch_w=switch_w,
                    balance_w=balance_w, batch=batch)
        return history

    if bank is not None:
        # staged on a pre-trained bank: head -> routing warm-up -> router ->
        # joint. The routing warm-up teaches the eye + router the per-regime
        # targets in isolation first; otherwise the eye has to learn the
        # sim/spatial/statistical split through the noisy task gradient and
        # never resolves it.
        n1 = 2                                    # shared-head warm-up
        n2 = max(2, int(epochs * 0.2))            # routing-stack warm-up
        n3 = max(3, int(epochs * 0.35))           # train the router
        n4 = max(3, epochs - n1 - n2 - n3)        # joint fine-tune
    else:
        n1 = max(2, int(epochs * 0.3))      # warm-start primitives
        n2 = max(2, int(epochs * 0.4))      # train the router
        n3 = max(1, epochs - n1 - n2)       # joint fine-tune

    freeze(model.primitives, True)
    freeze(model.eye, True); freeze(model.router, True)
    train_phase(model, train_ds, n1, seed, lr, task, domain_w, goals,
                verbose, tag + "/head", t0, history, use_aux=False,
                anneal=False, uniform=True, switch_w=0.0,
                balance_w=balance_w, batch=batch, fixed_tau=0.7)

    if bank is not None and oracle is not None:
        freeze(model.eye, False); freeze(model.router, False)
        train_phase(model, train_ds, n2, seed, lr, task, domain_w, goals,
                    verbose, tag + "/routewarm", t0, history, use_aux=False,
                    anneal=False, uniform=False, switch_w=0.0,
                    balance_w=balance_w, batch=batch, oracle=oracle,
                    oracle_w=1.0, oracle_only=True, fixed_tau=0.7)

        train_phase(model, train_ds, n3, seed, lr, task, domain_w, goals,
                    verbose, tag + "/router", t0, history, use_aux=True,
                    anneal=False, uniform=False, switch_w=switch_w,
                    balance_w=balance_w, batch=batch, oracle=oracle,
                    oracle_w=oracle_w, fixed_tau=0.7)

        freeze(model.primitives, False)
        train_phase(model, train_ds, n4, seed, lr * 0.3, task, domain_w, goals,
                    verbose, tag + "/joint", t0, history, use_aux=True,
                    anneal=False, uniform=False, switch_w=switch_w,
                    balance_w=balance_w, batch=batch, fixed_tau=0.7)
        return history

    if bank is not None:
        # no oracle available (multi-goal): head warm-up, then router, joint
        n3 = max(3, int((epochs - n1) * 0.7))
        n4 = max(1, epochs - n1 - n3)
        freeze(model.eye, False); freeze(model.router, False)
        train_phase(model, train_ds, n3, seed, lr, task, domain_w, goals,
                    verbose, tag + "/router", t0, history, use_aux=True,
                    anneal=False, uniform=False, switch_w=switch_w,
                    balance_w=balance_w, batch=batch, fixed_tau=0.7)
        freeze(model.primitives, False)
        train_phase(model, train_ds, n4, seed, lr * 0.3, task, domain_w, goals,
                    verbose, tag + "/joint", t0, history, use_aux=True,
                    anneal=False, uniform=False, switch_w=switch_w,
                    balance_w=balance_w, batch=batch, fixed_tau=0.7)
        return history

    freeze(model.eye, False); freeze(model.router, False)
    train_phase(model, train_ds, n2, seed, lr, task, domain_w, goals,
                verbose, tag + "/router", t0, history, use_aux=True,
                anneal=True, uniform=False, switch_w=switch_w,
                balance_w=balance_w, batch=batch)

    freeze(model.primitives, False)
    train_phase(model, train_ds, n3, seed, lr * 0.3, task, domain_w, goals,
                verbose, tag + "/joint", t0, history, use_aux=True,
                anneal=True, uniform=False, switch_w=switch_w,
                balance_w=balance_w, batch=batch)
    return history


def train_standalone(name, train_ds, epochs, seed, batch=128):
    set_seed(seed)
    model = Standalone(name, n_classes=10)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = make_loader(train_ds, batch=batch)
    for _ in range(epochs):
        model.train()
        for x, y, d, s, r in loader:
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def get_bank(name, train_ds, epochs, seed, batch=128, subsets=None):
    """Pre-train the primitive bank — each primitive standalone with its own
    head, full training signal on its niche — and cache it to disk. The
    Switch Operator is assembled on this bank; the standalone heads double as
    the per-expert 'quality judges' used to supervise the router.

    `subsets`: {primitive_name: [regime, ...]} restricts each expert's
    training data to those regimes (0=sim, 1=spatial, 2=statistical), turning
    the bank into true specialists — a flat lens that never saw spatial
    corruption is *visibly* the wrong tool for a warped image, which is what
    makes routing learnable. Default: each expert trains on everything.

    Returns {"prims": {name: state_dict}, "heads": {name: state_dict}}."""
    if subsets is not None:
        tag = "_sp" + "".join(f"{k[:2]}{''.join(map(str, v))}" for k, v in sorted(subsets.items()))
    else:
        tag = ""
    bdir = os.path.join(OUT, f"bank_{name}_{VERSION}_e{epochs}_t{len(train_ds)}_s{seed}{tag}")
    os.makedirs(bdir, exist_ok=True)
    prims, heads = {}, {}
    for pname in PRIMITIVES:
        pp, hp = os.path.join(bdir, f"{pname}_prim.pt"), os.path.join(bdir, f"{pname}_head.pt")
        if os.path.exists(pp) and os.path.exists(hp):
            prims[pname] = torch.load(pp, weights_only=True)
            heads[pname] = torch.load(hp, weights_only=True)
        else:
            ds = train_ds
            if subsets is not None and pname in subsets:
                mask = torch.isin(train_ds.tensors[4].to("cpu"),
                                  torch.tensor(subsets[pname]))
                ds = TensorDataset(*[t[mask] for t in train_ds.tensors])
            m = train_standalone(pname, ds, epochs, seed, batch)
            torch.save(m.prim.state_dict(), pp)
            torch.save(m.head.state_dict(), hp)
            prims[pname] = m.prim.state_dict()
            heads[pname] = m.head.state_dict()
            print(f"  [bank {name}] trained {pname}", flush=True)
    return {"prims": prims, "heads": heads}


def train_static(make, train_ds, epochs, seed, batch=128):
    set_seed(seed)
    m = make()
    m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loader = make_loader(train_ds, batch=batch)
    for _ in range(epochs):
        m.train()
        for x, y, d, s, r in loader:
            loss = nn.functional.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


# ---------------------------------------------------------------------------
# Protocol A — mixed-domain specialization
# ---------------------------------------------------------------------------

def protocol_a(seeds, epochs, n_train, n_test, smoke=False, batch=128):
    print("=== Protocol A: mixed-domain specialization ===")
    train_ds, test_ds = make_datasets(n_train, n_test)
    cfg = {"protocol": "A", "n_train": n_train, "n_test": n_test,
           "seeds": seeds, "epochs": epochs, "batch": batch}
    per_seed = {}
    statics = {
        "mlp": lambda: nn.Sequential(
            nn.Flatten(), nn.Linear(FLAT, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)),
        "cnn": lambda: nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 6 * 6, 10)),
    }
    variants = {
        "switch_full": dict(force_uniform=False),
        "switch_uniform": dict(force_uniform=True),
        "switch_random_router": dict(freeze_router=True),
    }
    for seed in seeds:
        t_seed = time.time()
        d = {}

        # 1. every single primitive as a generalist (trained on all data) —
        #    the 'best static lens' bound a routed operator must match.
        gen_bank = get_bank("Agen", train_ds, epochs, seed, batch)
        standalone = {}
        for name in PRIMITIVES:
            def _standalone(name=name, sd=gen_bank["prims"][name], hd=gen_bank["heads"][name]):
                m = Standalone(name, n_classes=10)
                m.to(DEVICE)
                m.prim.load_state_dict(sd); m.head.load_state_dict(hd)
                return round(eval_cls(m, test_ds)["acc"], 4)
            standalone[name], _ = with_chk("A", seed, epochs, n_train,
                                           f"standalone_{name}", _standalone,
                                           f"standalone {name}")
        d["standalone"] = standalone

        # 2. the specialist bank the operator is assembled on (niche-trained)
        bank = get_bank("A", train_ds, epochs, seed, batch, subsets=SPECIALISTS)

        # 2. static shared networks (classic single-topology baselines)
        for sname, make in statics.items():
            def _static(make=make, sname=sname):
                m = train_static(make, train_ds, epochs, seed, batch)
                return round(eval_cls(m, test_ds)["acc"], 4)
            d[f"static_{sname}"], _ = with_chk("A", seed, epochs, n_train,
                                                f"static_{sname}", _static,
                                                f"static {sname}")

        # 3. Switch Operator — full, uniform ensemble, random router
        for vname, opts in variants.items():
            def _variant(vname=vname, opts=opts, bank=bank):
                set_seed(seed)
                model = SwitchOperator(n_classes=10)
                if opts.get("freeze_router"):
                    freeze(model.eye, True); freeze(model.router, True)
                if opts.get("force_uniform"):
                    # the no-routing baseline: same pre-trained bank, same
                    # shared head, but the router is fixed to uniform and only
                    # the head may learn — primitives stay frozen so it cannot
                    # secretly de-specialize the bank.
                    model.force_uniform = True
                    freeze(model.eye, True); freeze(model.router, True)
                    freeze(model.primitives, True)
                train_model(model, train_ds, epochs, seed, tag=vname,
                            verbose=not smoke, batch=batch, bank=bank)
                res = eval_cls(model, test_ds,
                               uniform=opts.get("force_uniform", False))
                return {"acc": round(res["acc"], 4),
                        "stats": routing_stats(res)}
            out, _ = with_chk("A", seed, epochs, n_train, vname, _variant, vname)
            d[vname] = out["acc"]
            d[vname + "_stats"] = out["stats"]

        # 4. detailed diagnostics + playground data (seed 0 only)
        if seed == seeds[0]:
            def _diag(bank=bank):
                set_seed(seed)
                model = SwitchOperator(n_classes=10)
                train_model(model, train_ds, epochs, seed, tag="diag",
                            verbose=not smoke, batch=batch, bank=bank)
                res = eval_cls(model, test_ds)
                xs, ys, ds_, ss = (test_ds.tensors[i] for i in range(4))
                real_idx = torch.nonzero(ds_ == 1).squeeze(-1)
                pick = real_idx[torch.linspace(0, real_idx.numel() - 1, 24).long()]
                model.eval()
                with torch.no_grad():
                    _, info = model(xs[pick].to(DEVICE), tau=0.5, hard=False)
                save_playground_data(info["probs"].tolist(), ss[pick].tolist(),
                                     ys[pick].tolist(), xs[pick].numpy(),
                                     os.path.join(OUT, "playground.json"))
                return {"seed": seed, "routing": routing_stats(res),
                        "acc_sim": res["acc_sim"], "acc_real": res["acc_real"],
                        "acc": res["acc"]}
            d["diagnostics"], _ = with_chk("A", seed, epochs, n_train,
                                            "diag", _diag, "diagnostics")

        per_seed[seed] = d
        print(f"  [seed {seed}] full={d['switch_full']:.3f} "
              f"cnn={d['static_cnn']:.3f} ({time.time() - t_seed:.0f}s)", flush=True)

    # ---- merge across seeds ----------------------------------------------
    def mean(key):
        return round(sum(per_seed[s][key] for s in seeds) / len(seeds), 4)

    def mean_dict(key):
        return {name: round(sum(per_seed[s][key][name] for s in seeds) / len(seeds), 4)
                for name in PRIMITIVES}

    results = dict(cfg)
    results["models"] = {
        "standalone": mean_dict("standalone"),
        "static_mlp": mean("static_mlp"),
        "static_cnn": mean("static_cnn"),
        "switch_full": mean("switch_full"),
        "switch_uniform": mean("switch_uniform"),
        "switch_random_router": mean("switch_random_router"),
    }
    standalone = results["models"]["standalone"]
    results["models"]["best_static"] = {
        "name": max(standalone, key=standalone.get),
        "acc": max(standalone.values()),
    }
    if any("diagnostics" in per_seed[s] for s in seeds):
        results["diagnostics"] = next(per_seed[s]["diagnostics"] for s in seeds
                                      if "diagnostics" in per_seed[s])
    print(f"  standalone best: {results['models']['best_static']['name']} "
          f"= {results['models']['best_static']['acc']:.3f}")
    print(f"  switch_full={results['models']['switch_full']:.3f} "
          f"switch_uniform={results['models']['switch_uniform']:.3f} "
          f"static_cnn={results['models']['static_cnn']:.3f}")
    print(f"  diagnostics: acc_sim={results['diagnostics']['acc_sim']:.3f} "
          f"acc_real={results['diagnostics']['acc_real']:.3f} "
          f"switch_rate={results['diagnostics']['routing']['switch_rate']:.3f}")
    save("protocol_a.json", results)
    return results


# ---------------------------------------------------------------------------
# Protocol B — sim -> real transfer
# ---------------------------------------------------------------------------

def protocol_b(seeds, epochs, n_train, n_test, n_real_finetune=200, batch=128):
    print("=== Protocol B: sim->real transfer ===")
    # The toolbox: the primitive bank pre-trained on the mixed world (sim +
    # synthetic corruptions), exactly as in protocol A — one cache, three
    # protocols. The operator's routing stack then trains on *clean sim*
    # only: deployment-time experience. On real input the eye must route to
    # the right tool using cues it only learned on sim.
    mixed_train, _ = make_datasets(n_train, n_test)
    sim_train, _ = make_datasets(n_train, n_test, domain="sim")
    _, test_ds = make_datasets(n_train, n_test, domain="mixed")
    cfg = {"protocol": "B", "n_train": n_train, "n_test": n_test,
           "n_real_finetune": n_real_finetune, "seeds": seeds,
           "epochs": epochs, "batch": batch, "toolbox": "bank_A"}
    per_seed = {}

    def static_cnn():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 6 * 6, 10))

    def oracle_heads_from(bank):
        oh = {}
        for pname, sd in bank["heads"].items():
            h = nn.Linear(64, 10)
            h.load_state_dict(sd)
            h.eval()
            oh[pname] = h.to(DEVICE)
        return oh

    for seed in seeds:
        chk = load_chk("B", seed, epochs, n_train)
        if chk is not None:
            print(f"  [seed {seed}] cached, skipping")
            per_seed[seed] = chk
            continue
        t_seed = time.time()

        # toolbox bank (reuses the protocol-A cache: mixed world, specialists)
        bank = get_bank("A", mixed_train, epochs, seed, batch, subsets=SPECIALISTS)
        oracle_h = oracle_heads_from(bank)

        # static CNN baseline: trained on sim only, one fixed topology
        cnn = train_static(static_cnn, sim_train, epochs, seed, batch)
        r_cnn = eval_cls(cnn, test_ds)

        # the operator: toolbox bank, routing stack trained on sim only
        set_seed(seed + 777)
        switch = SwitchOperator(n_classes=10)
        train_model(switch, sim_train, epochs, seed + 777, tag="switch(sim)",
                    verbose=False, batch=batch, bank=bank)
        r_sw = eval_cls(switch, test_ds)
        base = {
            "static_cnn": {"acc_sim": r_cnn["acc_sim"], "acc_real": r_cnn["acc_real"]},
            "switch": {"acc_sim": r_sw["acc_sim"], "acc_real": r_sw["acc_real"]},
        }
        for k, v in base.items():
            base[k]["gap"] = round(v["acc_sim"] - v["acc_real"], 4)

        # --- tiny labeled real set: weight vs structure adaptation ---------
        real_ds, _ = make_datasets(n_real_finetune * 2, 100, domain="real")
        finetune_ds = TensorDataset(*[t[:n_real_finetune] for t in real_ds.tensors])

        # weight adaptation: unfreeze every weight of the static CNN
        cnn2 = train_static(static_cnn, sim_train, epochs, seed + 1, batch)
        opt2 = torch.optim.Adam(cnn2.parameters(), lr=5e-4)
        for _ in range(3):
            cnn2.train()
            for x, y, d, s, r in make_loader(finetune_ds, batch=64):
                loss = nn.functional.cross_entropy(cnn2(x), y)
                opt2.zero_grad(); loss.backward(); opt2.step()
        r_w = eval_cls(cnn2, test_ds)

        # structure adaptation: freeze tools AND the eye; adapt only the
        # router + task heads (4.7k params) with oracle supervision — the
        # routing learns which tool the real world needs; no weight is
        # rewritten.
        sw2 = SwitchOperator(n_classes=10).to(DEVICE)
        sw2.load_state_dict(switch.state_dict())
        for p in sw2.primitives.parameters():
            p.requires_grad = False
        for p in sw2.eye.parameters():
            p.requires_grad = False
        opt3 = torch.optim.Adam([p for p in sw2.parameters() if p.requires_grad],
                                lr=5e-4)
        for _ in range(6):
            sw2.train()
            for x, y, d, s, r in make_loader(finetune_ds, batch=min(64, n_real_finetune)):
                out, info = sw2(x, tau=0.5, hard=False)
                loss = nn.functional.cross_entropy(out, y)
                # oracle: which toolbox tool is least-bad on this real sample?
                with torch.no_grad():
                    ce = torch.zeros(len(PRIMITIVES), x.size(0), device=x.device)
                    for i, nm in enumerate(PRIMITIVES):
                        prim = Standalone(nm, n_classes=10).to(DEVICE)
                        prim.prim.load_state_dict(bank["prims"][nm])
                        prim.eval()
                        ce[i] = F.cross_entropy(oracle_h[nm](prim.prim(x)), y,
                                                reduction="none")
                    tgt = ce.argmin(0)
                loss = loss + 0.7 * F.cross_entropy(info["logits"], tgt)
                opt3.zero_grad(); loss.backward(); opt3.step()
        r_s = eval_cls(sw2, test_ds)

        n_w = sum(p.numel() for p in cnn2.parameters())
        n_s = sum(p.numel() for p in sw2.parameters() if p.requires_grad)
        per_seed[seed] = {
            "static_cnn": base["static_cnn"],
            "switch": base["switch"],
            "weight_adapted": {"acc_real": r_w["acc_real"],
                               "acc_sim": r_w["acc_sim"],
                               "adapted_params": n_w},
            "structure_adapted": {"acc_real": r_s["acc_real"],
                                  "acc_sim": r_s["acc_sim"],
                                  "adapted_params": n_s},
        }
        save_chk("B", seed, epochs, n_train, per_seed[seed])
        print(f"  [seed {seed}] cnn real={base['static_cnn']['acc_real']:.3f} "
              f"switch real={base['switch']['acc_real']:.3f} | "
              f"weight-adapt={r_w['acc_real']:.3f} struct-adapt={r_s['acc_real']:.3f} "
              f"({time.time() - t_seed:.0f}s)", flush=True)

    # ---- merge across seeds ----------------------------------------------
    def mean(key, sub):
        return round(sum(per_seed[s][key][sub] for s in seeds) / len(seeds), 4)
    results = dict(cfg)
    results["mean"] = {
        "static_cnn": {"acc_sim": mean("static_cnn", "acc_sim"),
                       "acc_real": mean("static_cnn", "acc_real"),
                       "gap": mean("static_cnn", "gap")},
        "switch": {"acc_sim": mean("switch", "acc_sim"),
                   "acc_real": mean("switch", "acc_real"),
                   "gap": mean("switch", "gap")},
        "weight_adapted": {"acc_real": mean("weight_adapted", "acc_real"),
                           "acc_sim": mean("weight_adapted", "acc_sim"),
                           "adapted_params": mean("weight_adapted", "adapted_params")},
        "structure_adapted": {"acc_real": mean("structure_adapted", "acc_real"),
                              "acc_sim": mean("structure_adapted", "acc_sim"),
                              "adapted_params": mean("structure_adapted", "adapted_params")},
    }
    print(f"  mean: switch real={results['mean']['switch']['acc_real']:.3f} "
          f"vs cnn real={results['mean']['static_cnn']['acc_real']:.3f} | "
          f"struct-adapt={results['mean']['structure_adapted']['acc_real']:.3f} "
          f"vs weight-adapt={results['mean']['weight_adapted']['acc_real']:.3f}")
    save("protocol_b.json", results)
    return results


# ---------------------------------------------------------------------------
# Protocol C — goal pathway
# ---------------------------------------------------------------------------

def protocol_c(seeds, epochs, n_train, n_test, batch=128):
    print("=== Protocol C: goal pathway ===")
    train_ds, test_ds = make_datasets(n_train, n_test)
    cfg = {"protocol": "C", "seeds": seeds, "epochs": epochs,
           "n_train": n_train, "n_test": n_test, "batch": batch}
    per_seed = {}

    for label, use_goal, n_goals in [("goal_conditioned", True, 2),
                                     ("goal_agnostic", False, 1)]:
        for seed in seeds:
            chk = load_chk(f"C_{label}", seed, epochs, n_train)
            if chk is not None:
                print(f"  [seed {seed}] {label}: cached, skipping")
                per_seed[(label, seed)] = chk
                continue
            t_seed = time.time()
            bank = get_bank("C", train_ds, 8, seed, batch)
            set_seed(seed)
            heads_out = [10, FLAT] if n_goals == 2 else [FLAT]
            model = SwitchOperator(n_classes=10, n_goals=n_goals,
                                   heads_out=heads_out)
            model.use_goal = use_goal
            train_model(model, train_ds, epochs, seed, task="multi", goals=[0, 1],
                        tag=label, verbose=False, batch=batch, bank=bank)
            # evaluate per-goal
            model.eval()
            loader = make_loader(test_ds, shuffle=False)
            acc_tot, rec_tot, n_acc, n_rec = 0, 0, 0, 0
            route_g0, route_g1 = [], []
            with torch.no_grad():
                for x, y, d, s, r in loader:
                    g0 = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                    out0, info0 = model(x, goal=g0, tau=0.5)
                    acc_tot += (out0[:, :10].argmax(-1) == y).sum().item()
                    n_acc += y.size(0)
                    route_g0.append(info0["probs"])
                    g1 = torch.ones(x.size(0), dtype=torch.long, device=x.device)
                    out1, info1 = model(x, goal=g1, tau=0.5)
                    rec_tot += nn.functional.mse_loss(
                        out1[:, :FLAT], x.reshape(x.size(0), -1)).item() * x.size(0)
                    n_rec += x.size(0)
                    route_g1.append(info1["probs"])
            route_g0 = torch.cat(route_g0).mean(0)
            route_g1 = torch.cat(route_g1).mean(0)
            d = {
                "acc_cls": round(acc_tot / n_acc, 4),
                "mse_rec": round(rec_tot / n_rec, 4),
                "route_classify": {PRIMITIVES[i]: round(float(v), 3)
                                   for i, v in enumerate(route_g0)},
                "route_reconstruct": {PRIMITIVES[i]: round(float(v), 3)
                                      for i, v in enumerate(route_g1)},
            }
            per_seed[(label, seed)] = d
            save_chk(f"C_{label}", seed, epochs, n_train, d)
            print(f"  [seed {seed}] {label}: cls={d['acc_cls']:.3f} "
                  f"rec={d['mse_rec']:.4f} ({time.time() - t_seed:.0f}s)", flush=True)

    results = dict(cfg)
    results["models"] = {}
    for label in ["goal_conditioned", "goal_agnostic"]:
        ds = [per_seed[(label, s)] for s in seeds]
        results["models"][label] = {
            "acc_cls": round(sum(d["acc_cls"] for d in ds) / len(ds), 4),
            "mse_rec": round(sum(d["mse_rec"] for d in ds) / len(ds), 4),
            "route_by_goal": {
                "classify": {PRIMITIVES[i]: round(
                    sum(d["route_classify"][PRIMITIVES[i]] for d in ds) / len(ds), 3)
                    for i in range(9)},
                "reconstruct": {PRIMITIVES[i]: round(
                    sum(d["route_reconstruct"][PRIMITIVES[i]] for d in ds) / len(ds), 3)
                    for i in range(9)},
            },
        }
        print(f"  {label}: cls acc={results['models'][label]['acc_cls']:.3f} "
              f"recon mse={results['models'][label]['mse_rec']:.4f}")
    save("protocol_c.json", results)
    return results


# ---------------------------------------------------------------------------
# Protocol D — ablations
# ---------------------------------------------------------------------------

def protocol_d(seeds, epochs, n_train, n_test, batch=128):
    print("=== Protocol D: ablations ===")
    train_ds, test_ds = make_datasets(n_train, n_test)
    cfg = {"protocol": "D", "seeds": seeds, "epochs": epochs,
           "n_train": n_train, "n_test": n_test, "batch": batch}
    per_seed = {}

    variants = {
        "full": dict(),
        "no_anneal": dict(fixed_tau=True),
        "random_router": dict(freeze_router=True),
        "no_bottleneck": dict(bottleneck=False),
        "no_eye": dict(no_eye=True),
        "domain_inv": dict(domain_inv=True),
    }
    for seed in seeds:
        t_seed = time.time()
        d = {}
        bank = get_bank("D", train_ds, 8, seed, batch)
        for vname, opts in variants.items():
            def _variant(vname=vname, opts=opts, bank=bank):
                set_seed(seed)
                model = SwitchOperator(n_classes=10,
                                       bottleneck=opts.get("bottleneck", True),
                                       with_domain_head=opts.get("domain_inv", False))
                if opts.get("freeze_router"):
                    freeze(model.eye, True); freeze(model.router, True)
                model.no_eye = opts.get("no_eye", False)
                model.fixed_tau = opts.get("fixed_tau", False)
                # no-eye: route on raw pixels instead of learned cues
                if model.no_eye:
                    model.router = nn.Sequential(nn.Flatten(),
                                                 nn.Linear(FLAT + 16, 64), nn.GELU(),
                                                 nn.Linear(64, 9))
                train_model(model, train_ds, epochs, seed, tag=vname,
                            verbose=False,
                            domain_w=1.0 if opts.get("domain_inv") else 0.0,
                            batch=batch, bank=bank)
                res = eval_cls(model, test_ds)
                return {"acc": round(res["acc"], 4),
                        "acc_real": round(res["acc_real"], 4)}
            d[vname], _ = with_chk("D", seed, epochs, n_train, vname,
                                   _variant, vname)
        per_seed[seed] = d
        print(f"  [seed {seed}] full={d['full']['acc']:.3f} "
              f"no_eye={d['no_eye']['acc']:.3f} "
              f"domain_inv={d['domain_inv']['acc']:.3f} "
              f"({time.time() - t_seed:.0f}s)", flush=True)

    results = dict(cfg)
    results["models"] = {}
    for vname in variants:
        ds = [per_seed[s][vname] for s in seeds]
        results["models"][vname] = {
            "acc": round(sum(d["acc"] for d in ds) / len(ds), 4),
            "acc_real": round(sum(d["acc_real"] for d in ds) / len(ds), 4),
        }
        print(f"  {vname:16s} acc={results['models'][vname]['acc']:.3f} "
              f"real={results['models'][vname]['acc_real']:.3f}")
    save("protocol_d.json", results)
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="ABCD", help="protocols to run")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None,
                    help="compute device: cuda/cpu (default: auto-detect)")
    args = ap.parse_args()
    if args.device is not None and args.device != "auto":
        global DEVICE
        DEVICE = args.device
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            print(f"[warn] --device {DEVICE} requested but CUDA unavailable; using cpu", flush=True)
            DEVICE = "cpu"

    if args.smoke:
        args.seeds, args.epochs, args.n_train, args.n_test = 1, 2, 400, 200
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    t0 = time.time()
    kw = dict(epochs=args.epochs, n_train=args.n_train, n_test=args.n_test,
              batch=args.batch)
    if "A" in args.only:
        protocol_a(seeds, smoke=args.smoke, **kw)
    if "B" in args.only:
        protocol_b(seeds, **kw)
    if "C" in args.only:
        protocol_c(seeds, **kw)
    if "D" in args.only:
        protocol_d(seeds, **kw)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
