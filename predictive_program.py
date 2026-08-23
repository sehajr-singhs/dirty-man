"""Self-supervised feature-conditioned program induction.

This is the next Dirty Man direction beyond ordinary MoE. Instead of receiving
regime labels or an oracle expert id, the learner creates its own competence
signal: each primitive predicts the target encoder's representation of a
second augmented view, and the router is trained to select the primitive with
the lowest *counterfactual* prediction error for that sample. The target
encoder is an EMA copy, so the objective is latent prediction rather than
pixel reconstruction (JEPA-like), while the primitive bank still contains
structurally different programs.

The resulting policy is:

    view_1 -> online encoder -> router -> {linear, MLP, gated} predictor
    view_2 -> EMA target encoder ------------------------------^ target latent

No class, domain, corruption, or expert labels are used by the objective.
At deployment the router can commit to one primitive and can optionally take
an early exit when its confidence is high, making the computation itself an
input-conditioned object.

This file deliberately reports counterfactual regret and route diversity.
Those diagnostics distinguish genuine competence-conditioned routing from a
router that merely collapses onto the largest expert.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = ("linear", "mlp", "gated")
RESULTS = "results"


class Encoder(nn.Module):
    def __init__(self, width=32, dim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 2, dim),
        )

    def forward(self, x):
        return F.normalize(self.body(x), dim=-1)


class PredictorBank(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.ops = nn.ModuleDict({
            "linear": nn.Linear(dim, dim),
            "mlp": nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                  nn.Linear(dim * 2, dim)),
            "gated": nn.Sequential(nn.Linear(dim, dim * 2), nn.GLU(dim=-1),
                                    nn.Linear(dim, dim)),
        })

    def forward(self, z):
        return torch.stack([op(z) for op in self.ops.values()], dim=1)


class PredictiveProgram(nn.Module):
    """A router trained from per-sample counterfactual competence."""

    def __init__(self, dim=64, momentum=0.996):
        super().__init__()
        self.online = Encoder(dim=dim)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictors = PredictorBank(dim=dim)
        self.router = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                    nn.Linear(dim, len(OPS)))
        self.momentum = momentum

    @torch.no_grad()
    def update_target(self):
        for online, target in zip(self.online.parameters(), self.target.parameters()):
            target.mul_(self.momentum).add_(online, alpha=1.0 - self.momentum)

    def forward(self, view_a, view_b, tau=1.0, hard=False):
        context = self.online(view_a)
        with torch.no_grad():
            target = self.target(view_b)
        candidates = self.predictors(context)             # (B, E, D)
        counterfactual = ((candidates - target[:, None, :]) ** 2).mean(-1)
        # Competitive assignment: each sample goes to its lowest-error
        # predictor, balanced to prevent collapse. Only the assigned
        # predictor receives gradient for that sample.
        pseudo = balanced_assignments(counterfactual.detach())
        # Select the assigned predictor's output ONLY (hard, not soft)
        selected = candidates[torch.arange(candidates.size(0)), pseudo]
        # Prediction loss ONLY on the selected predictor per sample.
        # This creates specialization: each predictor improves on its
        # own samples, creating a positive feedback loop.
        selected_loss = F.mse_loss(selected, target)
        # Router learns to predict assignments from context
        logits = self.router(context)
        route_loss = F.cross_entropy(logits, pseudo)
        # Output diversity: push predictor outputs apart (L2 distance)
        # so they genuinely specialize rather than converge to the same thing.
        diffs = (candidates[:, :, None, :] - candidates[:, None, :, :]) ** 2
        diversity = -diffs.mean()  # negative, so minimizing loss = maximizing diversity
        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            probs = F.softmax(logits / max(tau, 1e-3), dim=-1)
        hard_assign = probs.detach().argmax(-1)
        frac = F.one_hot(hard_assign, len(OPS)).float().mean(0)
        mean_p = probs.mean(0)
        balance_loss = (mean_p.clamp_min(1e-8) *
                        (mean_p.clamp_min(1e-8) * len(OPS)).log()).sum()
        confidence = probs.max(-1).values.mean()
        route_acc = (hard_assign == pseudo).float().mean()
        loss = (selected_loss + 0.3 * route_loss + 0.05 * balance_loss
                + 0.02 * diversity)
        return {
            "loss": loss,
            "selected_loss": selected_loss.detach(),
            "route_loss": route_loss.detach(),
            "balance_loss": balance_loss.detach(),
            "diversity": diversity.detach(),
            "counterfactual": counterfactual.detach(),
            "pseudo": pseudo.detach(),
            "probs": probs,
            "confidence": confidence.detach(),
            "route_acc": route_acc.detach(),
            "pseudo_utilization": F.one_hot(pseudo, len(OPS)).float().mean(0).detach(),
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def augment(x, generator=None):
    """Two-view augmentation that preserves digit identity and needs no labels."""
    if generator is None:
        generator = torch.default_generator
    noise = torch.randn(x.shape, device=x.device, generator=generator) * 0.08
    out = (x + noise).clamp(-1, 1)
    # Per-example integer translation: the router must attend to stable
    # structure, not memorize a fixed pixel coordinate.
    shifts = torch.randint(-2, 3, (x.size(0), 2), device=x.device, generator=generator)
    shifted = torch.empty_like(out)
    for i, (dy, dx) in enumerate(shifts.tolist()):
        shifted[i] = torch.roll(out[i], (dy, dx), dims=(-2, -1))
    return shifted


def balanced_assignments(cost, capacity=None):
    """Assign each sample to a low-cost expert while preventing collapse.

    This is a small, dependency-free balanced matching heuristic. Edges are
    visited from lowest to highest counterfactual cost; an edge is accepted
    when it assigns an unassigned sample to an expert below capacity. A final
    pass fills any unassigned samples with their cheapest available expert.
    The assignment is used only as a detached pseudo-label, so the heuristic
    never appears in the gradient path.
    """
    if cost.ndim != 2:
        raise ValueError("cost must have shape (batch, experts)")
    batch, experts = cost.shape
    if batch == 0:
        return torch.empty(0, dtype=torch.long, device=cost.device)
    if capacity is None:
        base, remainder = divmod(batch, experts)
        capacity = [base + int(i < remainder) for i in range(experts)]
    if len(capacity) != experts or sum(capacity) < batch or any(c < 0 for c in capacity):
        raise ValueError("capacity must cover the batch")

    labels = torch.full((batch,), -1, dtype=torch.long, device=cost.device)
    used = [0] * experts
    flat = torch.argsort(cost.detach().reshape(-1))
    for flat_idx in flat.tolist():
        sample = flat_idx // experts
        expert = flat_idx % experts
        if labels[sample] < 0 and used[expert] < capacity[expert]:
            labels[sample] = expert
            used[expert] += 1
    # Greedy capacity can leave a sample unmatched even when total capacity
    # is sufficient. Assign those samples to the cheapest remaining expert.
    for sample in (labels < 0).nonzero(as_tuple=False).flatten().tolist():
        available = [e for e in range(experts) if used[e] < capacity[e]]
        expert = min(available, key=lambda e: float(cost[sample, e]))
        labels[sample] = expert
        used[expert] += 1
    return labels


def anneal(epoch, epochs, start=1.5, end=0.35):
    if epochs <= 1:
        return end
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * epoch / (epochs - 1)))


def _image_loader(dataset, batch, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0)


def train(model, dataset, epochs=10, batch=128, device="cpu", seed=0):
    torch.manual_seed(seed)
    loader = _image_loader(dataset, batch, True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-5)
    history = []
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "selected_loss": 0.0,
                "route_loss": 0.0, "balance_loss": 0.0,
                "diversity": 0.0,
                "confidence": 0.0, "route_acc": 0.0,
                "raw_best_acc": 0.0}
        pseudo_counts = torch.zeros(len(OPS), dtype=torch.float64)
        count = 0
        tau = anneal(epoch, epochs)
        for batch_data in loader:
            x = batch_data[0].to(device)
            va, vb = augment(x), augment(x)
            opt.zero_grad(set_to_none=True)
            result = model(va, vb, tau=tau, hard=epoch >= epochs - 2)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            n = x.size(0)
            count += n
            for key in sums:
                if key == "raw_best_acc":
                    value = (result["probs"].detach().argmax(-1) ==
                             result["counterfactual"].detach().argmin(-1)).float().mean()
                elif key == "diversity":
                    value = result.get("diversity", torch.tensor(0.0))
                else:
                    value = result[key].detach() if torch.is_tensor(result[key]) else result[key]
                sums[key] += float(value) * n
            pseudo_counts += result["pseudo_utilization"].detach().cpu().double() * n
        row = {"epoch": epoch + 1, "tau": round(tau, 4),
               **{k: round(v / max(count, 1), 6) for k, v in sums.items()}}
        pseudo_counts /= max(count, 1)
        row["pseudo_utilization"] = {
            name: round(float(value), 6)
            for name, value in zip(OPS, pseudo_counts)
        }
        history.append(row)
        print(f"[predictive] {row}", flush=True)
    return history


@torch.no_grad()
def evaluate(model, dataset, batch=256, device="cpu"):
    model.eval()
    loader = _image_loader(dataset, batch, False)
    total = 0
    loss_sum = 0.0
    route_sum = 0.0
    raw_route_sum = 0.0
    regret_sum = 0.0
    counts = torch.zeros(len(OPS), dtype=torch.long)
    for batch_data in loader:
        x = batch_data[0].to(device)
        result = model(augment(x), augment(x), tau=0.2, hard=True)
        cf = result["counterfactual"]
        selected = cf.gather(1, result["probs"].argmax(-1, keepdim=True)).squeeze(1)
        oracle = cf.min(-1).values
        n = x.size(0)
        total += n
        loss_sum += float(cf.min(-1).values.mean()) * n
        route_sum += float(result["route_acc"]) * n
        raw_route_sum += float((result["probs"].argmax(-1) == cf.argmin(-1)).float().mean()) * n
        regret_sum += float((selected - oracle).mean()) * n
        counts += torch.bincount(result["probs"].argmax(-1).cpu(), minlength=len(OPS))
    frac = (counts.float() / max(int(counts.sum()), 1)).tolist()
    return {
        "counterfactual_best_mse": round(loss_sum / max(total, 1), 6),
        "balanced_assignment_accuracy": round(route_sum / max(total, 1), 6),
        "route_to_counterfactual_best": round(raw_route_sum / max(total, 1), 6),
        "routing_regret": round(regret_sum / max(total, 1), 6),
        "utilization": {name: round(frac[i], 6) for i, name in enumerate(OPS)},
        "utilization_entropy": round(float(-(torch.tensor(frac).clamp_min(1e-8) *
                                             torch.tensor(frac).clamp_min(1e-8).log()).sum()), 6),
    }


def load_dataset(name, n_train, n_test, seed):
    if name == "svhn":
        from nonstatic_layers import load_svhn
        return load_svhn(n_train=n_train, n_test=n_test, seed=seed, download=False)
    if name == "mnist":
        from dirty_man.data_mnist import make_mnist_datasets
        return make_mnist_datasets(n_train=n_train, n_test=n_test)
    raise ValueError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["svhn", "mnist"], default="svhn")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.dataset, args.n_train, args.n_test, args.epochs, args.batch = "synthetic-smoke", 64, 32, 1, 16
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else args.device if args.device != "auto" else "cpu")
    if args.smoke:
        from torch.utils.data import TensorDataset
        g = torch.Generator().manual_seed(args.seed)
        smoke_x = torch.rand(args.n_train + args.n_test, 1, 32, 32, generator=g) * 2 - 1
        smoke_ds = TensorDataset(smoke_x, torch.zeros(len(smoke_x), dtype=torch.long))
        train_ds = torch.utils.data.Subset(smoke_ds, range(args.n_train))
        test_ds = torch.utils.data.Subset(smoke_ds, range(args.n_train, args.n_train + args.n_test))
    else:
        train_ds, test_ds = load_dataset(args.dataset, args.n_train, args.n_test, args.seed)
    # Seed before model construction as well as inside the training loop so
    # the published probe is reproducible across fresh processes.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = PredictiveProgram().to(device)
    history = train(model, train_ds, args.epochs, args.batch, device, args.seed)
    metrics = evaluate(model, test_ds, args.batch, device)
    result = {"experiment": "self_supervised_predictive_program",
              "dataset": args.dataset, "objective": "latent prediction with counterfactual routing",
              "seed": args.seed, "device": device, "params": model.n_params(),
              "history": history, "metrics": metrics}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "predictive_program.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
