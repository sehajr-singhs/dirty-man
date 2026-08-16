"""Switch Operator — structure-switching neural computation.

The Switch Operator is a gating architecture that *rewires its own computation*
based on what it sees. A visual-cue extractor (the Operator's "eye") watches the
input and, together with the goal pathway (the task the model is trying to
achieve), decides which neural primitive — or which soft combination — should
process it. The primitives are the nine "primitive options":

    linear, dense, relu, cnn, rnn, lstm, gan, autoencoder, transformer

each with its own internal layers and inductive lens over the same input. Every
primitive emits into a shared bottleneck latent space, so switched paths stay
geometry-continuous for the downstream head.

Routing is differentiable: the eye + router produce logits over primitives and a
Gumbel-Softmax operator (annealed soft -> hard during training) turns them into
routing weights. The default training protocol is *staged* (see
run_experiments.py): warm-start the primitives, train the router with the
primitives frozen, then fine-tune everything jointly — otherwise an untrained
bank collapses onto its most expressive member and never switches.

    input ──► [eye: visual cues] ─┐
              [goal pathway] ─────┼──► [router] ──► gating weights p
                                  │        │
    input ──► primitive_1 ──► ────┘        │
    input ──► primitive_2 ──► bottleneck ──┴──► p-weighted mixture ──► head ──► out
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64          # shared bottleneck latent dimension
CUE_DIM = 32         # visual-cue descriptor dimension
GOAL_DIM = 16        # goal-pathway embedding dimension
IMG = 24             # input image resolution (square)

PRIMITIVES = ["linear", "dense", "relu", "cnn", "rnn", "lstm", "gan", "autoencoder", "transformer"]


# ---------------------------------------------------------------------------
# Differentiable routing machinery
# ---------------------------------------------------------------------------

def annealed_tau(epoch: int, n_epochs: int, tau0: float = 4.0, tau1: float = 0.5) -> float:
    """Anneal the Gumbel temperature from tau0 (soft, everything trains) to
    tau1 (near-hard, the operator commits to a structure)."""
    if n_epochs <= 1:
        return tau1
    frac = epoch / (n_epochs - 1)
    return tau1 + (tau0 - tau1) * 0.5 * (1.0 + math.cos(math.pi * frac))


class GradientReversal(torch.autograd.Function):
    """Multiplies the gradient by -lambda in the backward pass. Makes the eye's
    visual cues invariant to the domain (sim vs real) label."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def _balance_loss(probs: torch.Tensor, hard_assign: torch.Tensor, n_experts: int) -> torch.Tensor:
    """MoE-style load-balancing loss, gently weighted so it steers utilization
    without fighting the task loss into uniformity."""
    mean_p = probs.mean(dim=0)                       # (E,)
    frac = hard_assign.float().mean(dim=0)           # (E,) fraction of tokens
    return n_experts * (frac * mean_p).sum()


def switching_pressure(probs: torch.Tensor) -> torch.Tensor:
    """Self-supervised 'switching pressure': reward the router for using
    *different* primitives on *different* samples in the batch. This is what
    makes the operator specialize per input instead of collapsing onto one
    primitive. No labels required.

    loss = -mean_i || p_i - mean_batch(p) ||_1   (maximize per-sample deviation
    from the batch-average routing).
    """
    mean_p = probs.mean(dim=0, keepdim=True)
    return -(probs - mean_p).abs().sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# The nine primitive options — each with its own internal layers
# ---------------------------------------------------------------------------

class _Flat(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class _Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dims = (dim0, dim1)

    def forward(self, x):
        return x.transpose(*self.dims)


class Primitive(nn.Module):
    """A primitive option: its own internal layers plus a bottleneck adapter that
    projects its representation into the shared latent space."""

    name: str
    input_format: str = "image"   # "image" | "rows" | "flat"

    def __init__(self, name: str, in_dim: int, internal_dim: int, latent: int = LATENT,
                 size: int = IMG, bottleneck: bool = True):
        super().__init__()
        self.name = name
        self.bottleneck = bottleneck
        self.latent = latent
        self.internal = self._build(in_dim, internal_dim, size)
        if bottleneck:
            self.adapter = nn.Linear(internal_dim, latent)
        else:
            self.adapter = nn.Identity()

    def _build(self, in_dim: int, h: int, size: int) -> nn.Module:
        raise NotImplementedError

    def _raw(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self._raw(x))

    @property
    def out_dim(self) -> int:
        return self.latent if self.bottleneck else self._internal_dim


class LinearPrimitive(Primitive):
    """A pure linear map — the 'flat vector' lens: no nonlinearity at all."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        return nn.Sequential(_Flat(), nn.Linear(in_dim, h))

    def _raw(self, x):
        return self.internal(x)


class DensePrimitive(Primitive):
    """A dense MLP (two hidden layers, GELU) — the 'everything is connected'
    lens over the flattened input."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        return nn.Sequential(_Flat(),
                             nn.Linear(in_dim, h * 2), nn.GELU(),
                             nn.Linear(h * 2, h * 2), nn.GELU(),
                             nn.Linear(h * 2, h))

    def _raw(self, x):
        return self.internal(x)


class ReLUCombinationPrimitive(Primitive):
    """A deep ReLU combination — three stacked ReLU layers. The 'brittle, sharp'
    lens: maximum nonlinearity, no normalization, no residual connections."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        layers = [_Flat(), nn.Linear(in_dim, h * 2)]
        for _ in range(2):
            layers += [nn.ReLU(), nn.Linear(h * 2, h * 2)]
        layers += [nn.ReLU(), nn.Linear(h * 2, h)]
        return nn.Sequential(*layers)

    def _raw(self, x):
        return self.internal(x)


class CNNPrimitive(Primitive):
    """Convolutional — the 'spatial lens': assumes pixels live in a grid and
    nearby pixels matter relative to each other (translation equivariance)."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        s = size // 4
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            _Flat(), nn.Linear(32 * s * s, h),
        )

    def _raw(self, x):
        return self.internal(x)


class _Rows(nn.Module):
    """Reshape (B,1,H,W) into a row sequence (B,H,W) for temporal lenses."""

    def forward(self, x):
        return x.squeeze(1)


class RNNPrimitive(Primitive):
    """Vanilla RNN over the image's scanline sequence — the 'order matters'
    lens: reads the image row by row like a stream."""

    input_format = "rows"

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        return nn.Sequential(
            _Rows(), nn.RNN(input_size=size, hidden_size=h, batch_first=True),
        )

    def _raw(self, x):
        out, _ = self.internal(x)
        return out[:, -1]


class LSTMPrimitive(Primitive):
    """LSTM over the scanline sequence — the 'long memory' lens."""

    input_format = "rows"

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        return nn.Sequential(
            _Rows(), nn.LSTM(input_size=size, hidden_size=h, batch_first=True),
        )

    def _raw(self, x):
        out, _ = self.internal(x)
        return out[:, -1]


class GANPrimitive(Primitive):
    """A generative expert. A small generator hallucinates a cleaned feature
    field from the input, a critic must tell the raw input from the generated
    field, and the two play a gentle feature-matching adversarial game."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        s = size // 4                                   # 6 for 24x24
        self.gen = nn.Sequential(
            nn.AdaptiveAvgPool2d((s, s)), _Flat(),
            nn.Linear(s * s, s * s * 16), nn.GELU(),
            nn.Unflatten(1, (16, s, s)),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.GELU(),   # 2s
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),               # 4s = size
        )
        self.pool = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool2d(2),
            _Flat(), nn.Linear(16 * s * s, h),
        )
        self.critic = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(4, 4, 3, padding=1), nn.LeakyReLU(0.2),
            _Flat(), nn.Linear(4 * size * size, 1),
        )
        return nn.Identity()

    def _raw(self, x):
        return self.pool(self.gen(x))

    def g_loss(self, x: torch.Tensor) -> torch.Tensor:
        fake = self.gen(x)
        return F.softplus(-self.critic(fake)).mean()

    def d_loss(self, x: torch.Tensor) -> torch.Tensor:
        fake = self.gen(x).detach()
        return F.softplus(-self.critic(x)).mean() + F.softplus(self.critic(fake)).mean()


class AutoencoderPrimitive(Primitive):
    """Autoencoder — the 'compress then rebuild' lens. The routing representation
    is the latent code; a decoder is trained with a reconstruction auxiliary loss
    so the code keeps the information the task needs."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        s = size // 4
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            _Flat(), nn.Linear(16 * s * s, h),
        )
        self.dec = nn.Sequential(
            nn.Linear(h, 16 * s * s), nn.ReLU(),
            nn.Unflatten(1, (16, s, s)),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
        )
        return nn.Identity()

    def _raw(self, x):
        return self.enc(x)

    def recon(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class TransformerPrimitive(Primitive):
    """Small vision transformer over patches — the 'attention' lens: long-range,
    content-based pairing of patches regardless of distance."""

    def _build(self, in_dim, h, size):
        self._internal_dim = h
        p = 6
        n_patches = (size // p) ** 2
        patch_dim = p * p
        self.to_patches = nn.Sequential(
            nn.Unfold(kernel_size=p, stride=p),          # (B, patch_dim, L)
            _Transpose(1, 2),                            # (B, L, patch_dim)
            nn.Linear(patch_dim, 32), nn.GELU(),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, 32) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, 32) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                               batch_first=True, activation="gelu",
                                               dropout=0.0)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Linear(32, h)
        return nn.Identity()

    def _raw(self, x):
        B = x.size(0)
        patches = self.to_patches(x)                     # (B, L, d)
        tokens = torch.cat([self.cls.expand(B, -1, -1), patches], dim=1) + self.pos
        out = self.enc(tokens)
        return self.head(out[:, 0])


def build_primitive(name: str, latent: int = LATENT, size: int = IMG,
                    bottleneck: bool = True, in_dim: int | None = None) -> Primitive:
    in_dim = in_dim or size * size
    h = {"linear": 64, "dense": 64, "relu": 64, "cnn": 32, "rnn": 32,
         "lstm": 32, "gan": 32, "autoencoder": 32, "transformer": 32}[name]
    cls = {"linear": LinearPrimitive, "dense": DensePrimitive, "relu": ReLUCombinationPrimitive,
           "cnn": CNNPrimitive, "rnn": RNNPrimitive, "lstm": LSTMPrimitive,
           "gan": GANPrimitive, "autoencoder": AutoencoderPrimitive,
           "transformer": TransformerPrimitive}[name]
    return cls(name, in_dim, h, latent=latent, size=size, bottleneck=bottleneck)


# ---------------------------------------------------------------------------
# The Switch Operator
# ---------------------------------------------------------------------------

class SwitchOperator(nn.Module):
    """The full architecture: eye (visual cues) + goal pathway + router + the
    primitive bank + task head(s)."""

    def __init__(self, n_classes: int = 10, size: int = IMG, latent: int = LATENT,
                 bottleneck: bool = True, n_goals: int = 1, cue_dim: int = CUE_DIM,
                 goal_dim: int = GOAL_DIM, with_domain_head: bool = False,
                 heads_out: list[int] | None = None):
        super().__init__()
        self.size = size
        self.latent = latent
        self.bottleneck = bottleneck
        self.n_goals = n_goals
        self.n_prims = len(PRIMITIVES)
        self.with_domain_head = with_domain_head
        self.force_uniform = False       # baseline: equal weights over primitives
        self.use_goal = True             # ablation: ignore the goal pathway
        self.fixed_tau = False           # ablation: no temperature annealing
        self.no_eye = False              # ablation: router sees raw pixels
        in_dim = size * size

        # The primitive bank — nine options, each with its own internal layers
        # plus a bottleneck adapter into the shared latent. The model-level
        # `bottleneck` flag only controls how the bank's outputs are mixed
        # (weighted sum in the shared latent vs concatenation), so the
        # no-bottleneck ablation still loads a pre-trained bank.
        self.primitives = nn.ModuleDict({
            name: build_primitive(name, latent=latent, size=size, bottleneck=True, in_dim=in_dim)
            for name in PRIMITIVES
        })

        # The eye: a small CNN that watches the input and produces visual cues.
        s = size // 4
        self.eye = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            _Flat(), nn.Linear(16 * s * s, 64), nn.ReLU(),
            nn.Linear(64, cue_dim),
        )

        # The goal pathway: an embedding of the task the model is trying to do.
        self.goal_net = nn.Sequential(
            nn.Embedding(n_goals, goal_dim), nn.Linear(goal_dim, goal_dim), nn.GELU(),
        )

        # The router: visual cues + goal -> logits over primitives.
        self.router = nn.Sequential(
            nn.Linear(cue_dim + goal_dim, 64), nn.GELU(), nn.Linear(64, self.n_prims),
        )

        # Domain head on the cues (optional; gradient-reversal makes cues
        # domain-invariant so routing follows content, not domain style).
        if with_domain_head:
            self.domain_head = nn.Linear(cue_dim, 2)

        # Task heads. For n_goals > 1 each goal gets its own head.
        mix_dim = latent if bottleneck else sum(p.out_dim for p in self.primitives.values())
        self.heads = nn.ModuleDict({
            str(g): nn.Linear(mix_dim, heads_out[g] if heads_out else n_classes)
            for g in range(n_goals)
        })

    # ------------------------------------------------------------------
    def route_logits(self, x: torch.Tensor, goal: torch.Tensor | None) -> torch.Tensor:
        if self.force_uniform:
            logits = torch.zeros(x.size(0), self.n_prims, device=x.device)
            cues = torch.zeros(x.size(0), CUE_DIM, device=x.device)
            return logits, cues
        # the Operator's eye: learned visual cues, or raw pixels when ablated
        cues = x.reshape(x.size(0), -1) if self.no_eye else self.eye(x)
        if goal is None or not self.use_goal:
            g = torch.zeros(x.size(0), GOAL_DIM, device=x.device)
        else:
            g = self.goal_net(goal)
        logits = self.router(torch.cat([cues, g], dim=-1))
        return logits, cues

    def _route(self, x, goal, tau, hard):
        logits, cues = self.route_logits(x, goal)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            # deterministic evaluation: softmax at the annealed temperature.
            probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)
        return probs, logits, cues

    def forward(self, x: torch.Tensor, goal: torch.Tensor | None = None, tau: float = 1.0,
                hard: bool = False, return_all: bool = False):
        probs, logits, cues = self._route(x, goal, tau, hard)
        z = {name: prim(x) for name, prim in self.primitives.items()}

        if self.bottleneck:
            h = sum(p.unsqueeze(-1) * z[name]
                    for name, p in zip(self.primitives, probs.unbind(-1)))
        else:
            h = torch.cat([p.unsqueeze(-1) * z[name]
                           for name, p in zip(self.primitives, probs.unbind(-1))], dim=-1)

        if goal is None or self.n_goals == 1:
            out = self.heads["0"](h)
        else:
            # heads may have different output dims (e.g. classify vs reconstruct)
            max_dim = max(head.out_features for head in self.heads.values())
            out = torch.zeros(h.size(0), max_dim, device=h.device)
            for g, head in enumerate(self.heads.values()):
                m = goal == g
                if m.any():
                    out[m, :head.out_features] = head(h[m])

        info = {"probs": probs, "logits": logits, "cues": cues, "z": z, "h": h}
        return out, info

    def aux_losses(self, x: torch.Tensor, probs: torch.Tensor,
                   hard_assign: torch.Tensor | None = None,
                   balance_w: float = 0.05, ent_w: float = 0.01,
                   ae_w: float = 0.5, gan_w: float = 0.05,
                   switch_w: float = 0.0) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        if hard_assign is not None:
            out["balance"] = _balance_loss(probs, hard_assign, self.n_prims) * balance_w
        p = probs.clamp_min(1e-8)
        out["entropy"] = -ent_w * (p * p.log()).sum(dim=-1).mean()   # maximize confidence
        if switch_w > 0:
            out["switch"] = switch_w * switching_pressure(probs)

        ae = self.primitives["autoencoder"]
        p_ae = probs[:, PRIMITIVES.index("autoencoder")].mean()
        if p_ae.item() > 1e-4:
            out["ae_recon"] = ae_w * p_ae.detach() * F.mse_loss(ae.recon(x), x)

        gan = self.primitives["gan"]
        p_gan = probs[:, PRIMITIVES.index("gan")].mean()
        if p_gan.item() > 1e-4:
            out["gan_gen"] = gan_w * p_gan.detach() * gan.g_loss(x)
        out["gan_disc"] = 0.1 * gan.d_loss(x)
        return out

    def domain_loss(self, x: torch.Tensor, domain: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        if not self.with_domain_head:
            return torch.zeros((), device=x.device)
        cues = self.eye(x)
        cues = GradientReversal.apply(cues, lam)
        return F.cross_entropy(self.domain_head(cues), domain)

    def utilization(self, probs: torch.Tensor) -> dict[str, float]:
        idx = probs.argmax(dim=-1)
        counts = torch.bincount(idx, minlength=self.n_prims).float()
        frac = counts / counts.sum().clamp_min(1.0)
        return {name: round(float(f), 4) for name, f in zip(PRIMITIVES, frac)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Standalone(nn.Module):
    """A single primitive + head, trained with no routing. The 'best static
    network' baseline: one fixed lens for every input."""

    def __init__(self, name: str, n_classes: int, size: int = IMG, latent: int = LATENT):
        super().__init__()
        self.prim = build_primitive(name, latent=latent, size=size, bottleneck=True)
        self.head = nn.Linear(latent, n_classes)

    def forward(self, x):
        return self.head(self.prim(x))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    import random
    random.seed(seed)
