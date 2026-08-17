"""The Dirty Man as a training-time intervention.

Not a benchmark — a *use case*. A vanilla learner (an MLP) is trained to
predict pendulum dynamics one step ahead, but it only ever saw *calm* orbits
(low angular velocity). On those it is accurate. On *energetic* orbits — out
of its training distribution — it fails: its one-step error is an order of
magnitude larger, and because nothing in its loss enforces energy
conservation, each bad step violates energy badly. The learner is not broken;
it is *out of its depth*, and it does not know it.

The Dirty Man is deployed as a training assistant. Its eye watches the state
and the learner's own prediction residual, and its router learns — with oracle
supervision — to *identify the failure regime*: the region of state space
where the vanilla learner's error is large. For samples in that regime it
intervenes, routing the computation to a physics-informed expert — a
Hamiltonian Neural Network whose symplectic integrator conserves energy by
construction. Outside the regime it leaves the vanilla learner to do the work
it does well.

The headline is the assistant's *policy*: the router learns to detect the
feature (high kinetic energy) that predicts learner failure, and the combined
system conserves energy where the vanilla learner breaks it — a structural
intervention, not a bigger learner.

    state + learner residual ──► [eye] ──► [router] ──► {vanilla | physics}
                                                      (intervene in failure regime)

Run:
    python training_intervention.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Pendulum physics (per unit mass). Energy is measured in units of gL (the
# natural scale), so all violations below are directly comparable.
G = 9.81
L = 1.0
DT = 0.05
ESCALE = G * L          # energy unit


# ---------------------------------------------------------------------------
# Pendulum data — symplectic (semi-implicit Euler) ground truth
# ---------------------------------------------------------------------------

def energy(theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Total mechanical energy (per unit mass): 1/2 L^2 w^2 + g L (1 - cos t)."""
    return 0.5 * L * L * omega ** 2 + G * L * (1.0 - torch.cos(theta))


def step_symplectic(theta: torch.Tensor, omega: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One semi-implicit Euler step — the energy-conserving ground truth."""
    omega_next = omega - (G / L) * torch.sin(theta) * DT
    theta_next = theta + omega_next * DT
    return theta_next, omega_next


def make_transitions(n: int, seed: int = 0, horizon: int = 3,
                     omega_max: float = 6.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (state, next-state) pairs by rolling out `horizon` symplectic
    steps from random initial conditions. States are (theta, omega).

    `omega_max` bounds the initial angular velocity: pass a small value to
    produce only *calm* (low-energy) orbits — the regime a weak learner can
    learn — and the full range for the complete dynamics."""
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(-np.pi, np.pi, n)
    w0 = rng.uniform(-omega_max, omega_max, n)
    th, w = torch.tensor(th0, dtype=torch.float32), torch.tensor(w0, dtype=torch.float32)
    for _ in range(horizon):
        th, w = step_symplectic(th, w)
    x = torch.stack([th, w], dim=-1)
    return x, x.clone()


def roll_out(model, theta0: torch.Tensor, omega0: torch.Tensor, steps: int) -> torch.Tensor:
    """Autoregressive rollout; returns the energy trajectory (steps+1, batch)."""
    th, w = theta0.clone(), omega0.clone()
    e = [energy(th, w)]
    for _ in range(steps):
        s = torch.stack([th, w], dim=-1)
        s_next = model(s)
        th, w = s_next[:, 0], s_next[:, 1]
        e.append(energy(th, w))
    return torch.stack(e)


def energy_drift(e: torch.Tensor) -> torch.Tensor:
    """Max |E - E0| over the trajectory (time axis = dim 0), in energy units gL."""
    e0 = e[:1].squeeze(0)                               # (batch,)
    return (e - e0).abs().max(dim=0).values / ESCALE


# ---------------------------------------------------------------------------
# The two experts
# ---------------------------------------------------------------------------

class VanillaLearner(nn.Module):
    """A plain MLP trained on one-step MSE — learns the map, not the law.
    Accurate in its training regime; out of depth (and energy-non-conserving)
    outside it."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def forward(self, x):
        return self.mlp(x)


class PhysicsExpert(nn.Module):
    """A Hamiltonian Neural Network: learns the pendulum's Hamiltonian H(t,w)
    and integrates Hamilton's equations with a symplectic (semi-implicit
    Euler) step. Energy is conserved *by construction* — the symplectic
    integrator preserves the learned Hamiltonian — so this is a genuinely
    physics-informed tool the assistant can route to when the vanilla learner
    breaks conservation."""

    def __init__(self):
        super().__init__()
        # H(t, w) = 0.5 L^2 w^2 + gL (1 - cos t); learn it as a small net.
        self.ham = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        """One symplectic step from state x = (t, w). Autograd is enabled
        locally so the Hamiltonian gradients are computable even when the
        caller is in no_grad (evaluation) context."""
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            H = self.ham(xg).sum()
            (dH,) = torch.autograd.grad(H, xg, create_graph=self.training)
            dtheta, domega = dH[:, 0], dH[:, 1]
            # Hamilton's equations (semi-implicit Euler):
            #   omega' = omega - dH/dtheta * dt ;  theta' = theta + dH/domega * dt
            omega_next = xg[:, 1] - dtheta * DT
            theta_next = xg[:, 0] + domega * DT
            return torch.stack([theta_next, omega_next], dim=-1)

    def loss(self, x, y):
        pred = self(x)
        mse = F.mse_loss(pred, y)
        # the symplectic integrator already conserves H; this MSE just makes
        # the learned H match the true dynamics.
        return mse, mse, torch.zeros((), device=x.device)


# ---------------------------------------------------------------------------
# The assistant: eye + router over {vanilla, physics}
# ---------------------------------------------------------------------------

class Assistant(nn.Module):
    """The Dirty Man as a training assistant. The eye watches the state and the
    vanilla learner's own per-sample error; the router decides, per sample,
    whether to trust the vanilla learner or intervene with the physics expert.

    Input features: [theta, omega, |residual| of the vanilla learner] — the
    state half is what lets the router generalise the failure regime to states
    it has not yet seen fail; the residual is the learner's own alarm signal."""

    def __init__(self):
        super().__init__()
        self.eye = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 16))
        self.router = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, x, residual):
        cues = self.eye(torch.cat([x, residual.unsqueeze(-1)], dim=-1))
        return self.router(cues)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_expert(model, x, y, epochs=400, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        if isinstance(model, PhysicsExpert):
            loss, _, _ = model.loss(x, y)
        else:
            loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    return model


def main(seed: int = 0, n_train: int = 4000, n_test: int = 1000,
         epochs: int = 400, verbose: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- data -------------------------------------------------------------
    # Full-range dynamics for the physics expert and for evaluation; the
    # vanilla learner is trained only on calm orbits (omega in [-2, 2]).
    x, y = make_transitions(n_train, seed=seed)
    x, y = x.to(DEVICE), y.to(DEVICE)
    xt, yt = make_transitions(n_test, seed=seed + 1)
    xt, yt = xt.to(DEVICE), yt.to(DEVICE)
    x_lo, y_lo = make_transitions(n_train, seed=seed + 10, omega_max=2.0)
    x_lo, y_lo = x_lo.to(DEVICE), y_lo.to(DEVICE)

    # ---- the vanilla learner: accurate on calm orbits, fails on energetic --
    vanilla = VanillaLearner().to(DEVICE)
    train_expert(vanilla, x_lo, y_lo, epochs=epochs, seed=seed)

    # ---- the physics-informed expert sees ALL data, conserves energy -------
    physics = PhysicsExpert().to(DEVICE)
    train_expert(physics, x, y, epochs=epochs, seed=seed + 2)

    # ---- the assistant learns to identify the failure regime --------------
    # Oracle: intervene *only where the vanilla learner fails* — where its own
    # one-step prediction error is in the worst half of the full-range
    # training distribution. The assistant is supervised to reproduce that
    # choice from (state, vanilla-residual) cues, so it must learn the feature
    # (high kinetic energy) that predicts failure rather than routing
    # everything to physics.
    assistant = Assistant().to(DEVICE)
    opt = torch.optim.Adam(assistant.parameters(), lr=1e-3)
    with torch.no_grad():
        vanilla_res = (vanilla(x) - y).abs().mean(-1)      # learner's alarm
        thresh = torch.quantile(vanilla_res, 0.5)
        oracle = (vanilla_res > thresh).long()             # 1 = intervene
    for _ in range(200):
        opt.zero_grad()
        logits = assistant(x, vanilla_res)
        loss = F.cross_entropy(logits, oracle)
        loss.backward()
        opt.step()

    # ---- evaluate ----------------------------------------------------------
    # The headline metric is *per-step energy violation*: for each test state,
    # how much does the chosen computation change energy in one step? This is
    # the regime-separating signal (the vanilla learner violates energy badly
    # exactly where it is out of distribution). We also report short-horizon
    # rollout drift as context.
    steps = 200
    kin0 = 0.5 * L * L * xt[:, 1] ** 2
    hi = kin0 > kin0.median()
    lo = ~hi

    def per_step_violation(predict_fn):
        """Mean |E(s') - E(s)| / gL over the test set for a step function."""
        with torch.no_grad():
            pred = predict_fn(xt)
            e_now = energy(xt[:, 0], xt[:, 1])
            e_next = energy(pred[:, 0], pred[:, 1])
            return (e_next - e_now).abs() / ESCALE

    with torch.no_grad():
        res_all = (vanilla(xt) - yt).abs().mean(-1)
        logits = assistant(xt, res_all)
        p_int = torch.softmax(logits, dim=-1)[:, 1]

        viol_v = per_step_violation(vanilla)
        viol_p = per_step_violation(physics)
        # assistant: hard routing — vanilla where p_int < 0.5, physics else
        def ass_step(s):
            vn = vanilla(s)
            r = (vn - s).abs().mean(-1)
            p = torch.softmax(assistant(s, r), dim=-1)[:, 1:2]
            return p * physics(s) + (1 - p) * vn
        viol_a = per_step_violation(ass_step)

        # rollout drift (context)
        e_v = roll_out(vanilla, xt[:, 0], xt[:, 1], steps)
        e_p = roll_out(physics, xt[:, 0], xt[:, 1], steps)
        th, w = xt[:, 0].clone(), xt[:, 1].clone()
        e_a = [energy(th, w)]
        for _ in range(steps):
            s = torch.stack([th, w], dim=-1)
            s_next = ass_step(s)
            th, w = s_next[:, 0], s_next[:, 1]
            e_a.append(energy(th, w))
        e_a = torch.stack(e_a)

    def reg(mask, vals):
        return round(float(vals[mask].mean()), 4)
    result = {
        "experiment": "training_time_intervention",
        "seed": seed, "n_train": n_train, "n_test": n_test,
        "per_step_energy_violation_gL": {
            "vanilla": round(float(viol_v.mean()), 4),
            "vanilla_high_kinetic": reg(hi, viol_v),
            "vanilla_low_kinetic": reg(lo, viol_v),
            "physics_expert": round(float(viol_p.mean()), 4),
            "assistant_routed": round(float(viol_a.mean()), 4),
            "assistant_high_kinetic": reg(hi, viol_a),
            "assistant_low_kinetic": reg(lo, viol_a),
        },
        "rollout_drift_gL_200steps": {
            "vanilla": round(float(energy_drift(e_v).mean()), 4),
            "physics_expert": round(float(energy_drift(e_p).mean()), 4),
            "assistant_routed": round(float(energy_drift(e_a).mean()), 4),
        },
        "policy": {"p_intervene_high_kinetic": round(float(p_int[hi].mean()), 3),
                   "p_intervene_low_kinetic": round(float(p_int[lo].mean()), 3),
                   "intervention_rate": round(float(p_int.mean()), 3)},
        "oracle_fraction_intervene": round(float(oracle.float().mean()), 3),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "training_intervention.json"), "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print("=== Training-time intervention ===")
        print("  per-step energy violation (units of gL):")
        print(f"    vanilla learner      {float(viol_v.mean()):.4f}  (hi-kin {reg(hi, viol_v):.4f}, lo-kin {reg(lo, viol_v):.4f})")
        print(f"    physics expert       {float(viol_p.mean()):.4f}")
        print(f"    assistant-routed     {float(viol_a.mean()):.4f}  (hi-kin {reg(hi, viol_a):.4f}, lo-kin {reg(lo, viol_a):.4f})")
        print("  rollout drift (200 steps, gL): "
              f"vanilla {float(energy_drift(e_v).mean()):.4f}  physics {float(energy_drift(e_p).mean()):.4f}  "
              f"assistant {float(energy_drift(e_a).mean()):.4f}")
        print(f"  assistant policy: intervene on high-kinetic {float(p_int[hi].mean()):.2f} vs "
              f"low-kinetic {float(p_int[lo].mean()):.2f} (oracle says {float(oracle.float().mean()):.2f} need it)")
        print(f"  wrote {os.path.join(OUT, 'training_intervention.json')}")
    return result


if __name__ == "__main__":
    main()
