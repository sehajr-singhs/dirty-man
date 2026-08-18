"""The flagship result: routing beats every fixed network *because* no fixed
network can fit all regimes. This is the inverted-design thesis made concrete:
embed the physics you know as exact first-class experts, learn only the law
you don't know, and route between them.

A pendulum obeys different physical laws in different regimes, and each law
demands a different, mutually-exclusive inductive bias:

    R0 conservative   theta'' = -(g/L) sin(theta)            energy is CONSERVED
    R1 damped         theta'' = -(g/L) sin(theta) - b theta' energy DECAYS
    R2 driven         theta'' = -(g/L) sin(theta) - b theta' + A sin(Omega t)
                                                             energy is PUMPED

The property that flips the correct bias is *energy conservation*. A single
fixed network has one bias: it cannot conserve energy on R0 and dissipate it
on R1 at the same time. The switch operator's eye watches a short trajectory,
detects which law is governing (energy flat vs decaying vs pumped), and routes
each trajectory to the expert that obeys that law.The experts embody the inverted-design principle: each is the exact,
closed-form physics law, hardcoded as a harness with no free parameters
(conservative, damped, driven). The two learned pieces are the *router* —
which must detect which law is governing from a short trajectory — and the
single static network, which is the brute-force baseline.

A single static network (one MLP trained on everything) cannot fit any regime
well: the governing law is *not identifiable from a single state*, so a fixed
state-to-state map cannot both conserve energy (R0) and dissipate it (R1). It
must compromise, and it fails on every regime. The headline is therefore not
"routing is a bit better than the best static net" — it is that **a single
fixed map cannot obey two mutually-exclusive laws, and only routing can**.
This is the same separation the companion theorem makes precise.

    short trajectory ──► [eye] ──► [router] ──► {conservative | damped | driven} ──► rollout

Run:
    python flagship_regime_routing.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

G = 9.81
L = 1.0
DT = 0.05
ESCALE = G * L              # energy unit: gL
OMEGA_DRIVE = 2.0 / 3.0

# ---------------------------------------------------------------------------
# The three regimes (three different laws) and their ground-truth dynamics
# ---------------------------------------------------------------------------

REGIMES = [
    {"name": "conservative", "b": 0.0, "A": 0.0, "Omega": 0.0},
    {"name": "damped",       "b": 0.8, "A": 0.0, "Omega": 0.0},
    {"name": "driven",       "b": 0.1, "A": 0.6, "Omega": OMEGA_DRIVE},
]

STATE_DIM = 3               # (theta, omega, phase)  phase = drive clock
WINDOW = 16                 # eye watches this many consecutive states
FEAT_DIM = 4                # (theta, omega, phase, energy) per state


def energy(theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return 0.5 * L * L * omega ** 2 + G * L * (1.0 - torch.cos(theta))


def advance_phase(phi):
    return (phi + OMEGA_DRIVE * DT) % (2.0 * math.pi)


def step_physics(theta, omega, phi, b, A, Omega):
    """One semi-implicit Euler step for the (b, A, Omega) law."""
    accel = -(G / L) * torch.sin(theta) - b * omega + A * torch.sin(phi)
    omega_next = omega + accel * DT
    theta_next = theta + omega_next * DT
    phi_next = (phi + Omega * DT) % (2.0 * math.pi)
    return theta_next, omega_next, phi_next


def make_trajectories(n_traj, horizon, regime, seed):
    """Roll out `n_traj` trajectories of `horizon` steps under one regime's law.
    Returns:
      states    (n_traj, horizon, 3)   full state trajectories
      feats     (n_traj, horizon, 4)   (theta, omega, phase, energy) per step
      windows   (N, WINDOW*4)          flattened feature windows ending at t
    """
    rng = np.random.default_rng(seed)
    th = torch.tensor(rng.uniform(-math.pi, math.pi, n_traj), dtype=torch.float32)
    w = torch.tensor(rng.uniform(-2.5, 2.5, n_traj), dtype=torch.float32)
    ph = torch.tensor(rng.uniform(0, 2 * math.pi, n_traj), dtype=torch.float32)

    states, feats = [], []
    for _ in range(horizon):
        s = torch.stack([th, w, ph], dim=-1)
        e = energy(th, w).unsqueeze(-1)
        states.append(s)
        feats.append(torch.cat([s, e], dim=-1))
        th, w, ph = step_physics(th, w, ph, regime["b"], regime["A"], regime["Omega"])

    states = torch.stack(states, dim=1)             # (n_traj, horizon, 3)
    feats = torch.stack(feats, dim=1)               # (n_traj, horizon, 4)

    windows = []
    for t in range(WINDOW - 1, horizon):
        windows.append(feats[:, t - WINDOW + 1:t + 1].reshape(n_traj, -1))
    windows = torch.cat(windows, dim=0)
    return states, feats, windows


# ---------------------------------------------------------------------------
# The experts — inverted design: exact physics where known, learned where not
# ---------------------------------------------------------------------------

class HardcodedExpert(nn.Module):
    """An exact, closed-form physics expert (the 'flawless harness'): integrates
    the known (b, A, Omega) law with no free parameters. Zero error on its own
    regime by construction; wrong wherever a different law governs."""

    def __init__(self, b, A, Omega):
        super().__init__()
        self.b, self.A, self.Omega = b, A, Omega

    def forward(self, s):
        th, w, ph = s[:, 0], s[:, 1], s[:, 2]
        th_n, w_n, ph_n = step_physics(th, w, ph, self.b, self.A, self.Omega)
        return torch.stack([th_n, w_n, ph_n], dim=-1)


class StaticBest(nn.Module):
    """The single best static network (brute force): one MLP over
    (theta, omega, phase) trained on ALL regimes jointly. One inductive bias,
    so it must compromise."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 2))

    def forward(self, s):
        out = self.mlp(s)
        return torch.cat([out, advance_phase(s[:, 2]).unsqueeze(-1)], dim=-1)


class Router(nn.Module):
    """Eye over a short feature trajectory (WINDOW x 4) + router over 3 experts."""

    def __init__(self, n_experts=3, window=WINDOW):
        super().__init__()
        self.eye = nn.Sequential(nn.Linear(window * FEAT_DIM, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))
        self.router = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, n_experts))

    def forward(self, window):
        return self.router(self.eye(window))


# ---------------------------------------------------------------------------
# Training + evaluation helpers
# ---------------------------------------------------------------------------

def train_one(model, states, nxts, epochs, lr, seed):
    """Train a learned model on one-step (state -> next-state) transitions."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(states)
        loss = F.mse_loss(pred[:, :2], nxts[:, :2])
        loss.backward()
        opt.step()
    model.eval()


def rollout(model, s0, steps):
    """Autoregressive rollout from s0 (N, 3). Returns (steps+1, N, 3)."""
    s = s0
    traj = [s]
    for _ in range(steps):
        s = model(s)
        traj.append(s)
    return torch.stack(traj, dim=0)


def rollout_state_error(pred_traj, true_traj):
    """Mean squared (theta, omega) error over the rollout, averaged over steps."""
    d = (pred_traj[:, :, :2] - true_traj[:, :, :2]) ** 2
    return d.mean().item()


def rollout_energy_error(pred_traj, true_traj):
    """Mean |E_pred(t) - E_true(t)| / gL over the rollout."""
    ep = energy(pred_traj[:, :, 0], pred_traj[:, :, 1])
    et = energy(true_traj[:, :, 0], true_traj[:, :, 1])
    return ((ep - et).abs() / ESCALE).mean().item()


def main(seed=0, n_traj=400, horizon=128, epochs=800, n_test=200,
         test_steps=100, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- data: per-regime trajectories ------------------------------------
    data = {}
    for r, reg in enumerate(REGIMES):
        states, feats, windows = make_trajectories(n_traj, horizon, reg, seed=seed + r)
        s_flat = states[:, :-1].reshape(-1, STATE_DIM).to(DEVICE)
        n_flat = states[:, 1:].reshape(-1, STATE_DIM).to(DEVICE)
        data[r] = dict(reg=reg, s_flat=s_flat, n_flat=n_flat, windows=windows.to(DEVICE))

    # ---- the three experts: the exact, closed-form physics harness -------
    conservative = HardcodedExpert(REGIMES[0]["b"], REGIMES[0]["A"], REGIMES[0]["Omega"]).to(DEVICE)
    damped = HardcodedExpert(REGIMES[1]["b"], REGIMES[1]["A"], REGIMES[1]["Omega"]).to(DEVICE)
    driven = HardcodedExpert(REGIMES[2]["b"], REGIMES[2]["A"], REGIMES[2]["Omega"]).to(DEVICE)
    experts = [conservative, damped, driven]

    # ---- the single best static network, trained on ALL regimes ------------
    all_s = torch.cat([d["s_flat"] for d in data.values()], dim=0)
    all_n = torch.cat([d["n_flat"] for d in data.values()], dim=0)
    static = StaticBest().to(DEVICE)
    train_one(static, all_s, all_n, epochs, 1e-3, seed + 3)

    # ---- the router: oracle-supervised to identify the governing law -------
    router = Router().to(DEVICE)
    opt = torch.optim.Adam(router.parameters(), lr=1e-3)
    win = torch.cat([d["windows"] for d in data.values()], dim=0)
    lab = torch.cat([torch.full((d["windows"].size(0),), r, dtype=torch.long)
                     for r, d in data.items()], dim=0).to(DEVICE)
    for _ in range(600):
        opt.zero_grad()
        loss = F.cross_entropy(router(win), lab)
        loss.backward()
        opt.step()

    # ---- evaluation: multi-step rollout per regime -------------------------
    rows = {}
    n_steps = test_steps + 1
    for r, reg in enumerate(REGIMES):
        states, feats, windows = make_trajectories(n_test, horizon, reg, seed=seed + 100 + r)
        s0 = states[:, 0].to(DEVICE)
        true_traj = states[:, :n_steps].to(DEVICE).transpose(0, 1)

        with torch.no_grad():
            tr_cons = rollout(conservative, s0, test_steps)
            tr_damp = rollout(damped, s0, test_steps)
            tr_driv = rollout(driven, s0, test_steps)
            tr_static = rollout(static, s0, test_steps)

            # switch: route each trajectory ONCE from its opening window, then
            # commit to that expert for the whole rollout
            first_win = feats[:, :WINDOW].reshape(n_test, -1).to(DEVICE)
            choice = router(first_win).argmax(-1)
            tr_switch = torch.zeros_like(true_traj)
            for e in range(3):
                mask = choice == e
                if mask.any():
                    tr_switch[:, mask] = rollout(experts[e], s0[mask], test_steps)

        rows[reg["name"]] = {
            "conservative_expert": round(rollout_state_error(tr_cons, true_traj), 4),
            "damped_expert": round(rollout_state_error(tr_damp, true_traj), 4),
            "driven_expert": round(rollout_state_error(tr_driv, true_traj), 4),
            "static_best": round(rollout_state_error(tr_static, true_traj), 4),
            "switch": round(rollout_state_error(tr_switch, true_traj), 4),
            "static_energy_err": round(rollout_energy_error(tr_static, true_traj), 4),
            "switch_energy_err": round(rollout_energy_error(tr_switch, true_traj), 4),
        }
        with torch.no_grad():
            acc = (router(first_win).argmax(-1) == r).float().mean().item()
        rows[reg["name"]]["router_accuracy"] = round(acc, 3)

    result = {
        "experiment": "flagship_regime_routing",
        "claim": "no single network fits all regimes; routing does",
        "seed": seed,
        "window": WINDOW,
        "rollout_steps": test_steps,
        "per_regime": rows,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "flagship_regime_routing.json"), "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print("=== Flagship: regime-switching pendulum (rollout state error) ===")
        print(f"{'regime':<14} {'cons':>8} {'damped':>8} {'driven':>8} {'static':>8} {'SWITCH':>8}")
        for r, reg in enumerate(REGIMES):
            d = rows[reg["name"]]
            print(f"{reg['name']:<14} {d['conservative_expert']:>8.3f} {d['damped_expert']:>8.3f} "
                  f"{d['driven_expert']:>8.3f} {d['static_best']:>8.3f} {d['switch']:>8.3f}")
        print()
        print("energy-profile error (gL units; lower = obeys the true law):")
        for reg in REGIMES:
            d = rows[reg["name"]]
            print(f"  {reg['name']:<14} static {d['static_energy_err']:.4f}  "
                  f"switch {d['switch_energy_err']:.4f}  (router acc {d['router_accuracy']:.2f})")
        print(f"  wrote {os.path.join(OUT, 'flagship_regime_routing.json')}")

    return result


if __name__ == "__main__":
    main()
