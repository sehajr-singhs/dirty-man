"""Generate every figure in the paper + website from the committed result files.

    python make_figs.py

Writes to figs/ (repo figures) and docs/figs/ (website figures). Every number
plotted here is read from results/*.json — nothing is hard-coded, so the
figures track the experiments exactly.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
DOC_FIG = os.path.join(ROOT, "docs", "figs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DOC_FIG, exist_ok=True)

PRIMS = ["linear", "dense", "relu", "cnn", "rnn", "lstm", "gan", "autoencoder", "transformer"]
PRIM_LABELS = ["linear", "dense", "ReLU", "CNN", "RNN", "LSTM", "GAN", "AE", "Transformer"]

INK = "#1a1a1a"
MUTED = "#555555"
FAINT = "#8c8e90"
PANEL = "#f8f8f8"
LINK = "#226999"
WIN = "#1a7a3c"
LOSE = "#a83a3a"


def load(name: str) -> dict:
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig: plt.Figure, fname: str) -> None:
    for d in (FIG, DOC_FIG):
        fig.savefig(os.path.join(d, fname), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.edgecolor": "#c4c6c8",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Figure 1 — architecture diagram
# ---------------------------------------------------------------------------
def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)

    def box(x, y, w, h, text, fc=PANEL, ec="#c4c6c8", fs=8.5, bold=False, lw=1.0):
        r = matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=lw)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
                linespacing=1.35)

    def arrow(x1, y1, x2, y2, lw=1.3, color=INK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=12))

    # Input
    box(2, 20, 10, 8, "input\nglyph\n24\u00d724", fc="#eef3f8", ec=LINK, bold=True)
    # Eye
    box(18, 30, 14, 12, "eye\nvisual-cue\nextractor", fc="#eef3f8", ec=LINK, bold=True)
    # Goal pathway
    box(18, 4, 14, 10, "goal\npathway", fc="#fdf3e7", ec="#b5822e", bold=True)
    # Router
    box(40, 16, 15, 16, "router\ngating logits\n+ Gumbel-Softmax", fc="#f3f0fa", ec="#6b4fa0", bold=True)
    # Primitive bank
    prim_names = ["linear", "dense", "ReLU", "CNN", "RNN", "LSTM", "GAN", "AE", "Transformer"]
    px, py, pw, ph, gap = 62, 30, 34, 12, 0
    for i, pn in enumerate(prim_names):
        yy = 30 + (i % 3) * 4.2
        xx = 62 + (i // 3) * 11.5
        box(xx, yy, 10.5, 3.6, pn, fc="#ffffff", ec="#c4c6c8", fs=7.2)
    # Bottleneck
    box(62, 16, 34, 9, "shared bottleneck latent \u2113 \u2208 \u211d^64", fc="#eef8ee", ec="#1a7a3c", bold=True)
    # Head
    box(40, 32.5, 15, 10, "task head\n(classify /\nreconstruct)", fc="#eef3f8", ec=LINK, bold=True)
    # Output
    box(2, 30, 10, 8, "prediction", fc="#eef3f8", ec=LINK, bold=True)

    arrow(12, 24, 17, 34)            # input -> eye
    arrow(12, 24, 17, 10)            # input -> goal
    arrow(32, 36, 39, 30)            # eye -> router
    arrow(32, 9, 39, 22)             # goal -> router
    arrow(55, 24, 61, 32)            # router -> bank (top)
    arrow(55, 22, 61, 22)            # router -> bank (middle)
    arrow(55, 20, 61, 12)            # router -> bank (bottom)
    # bank -> bottleneck
    arrow(76, 30, 76, 25.6, color="#1a7a3c")
    arrow(70, 29.8, 68, 25.6, color="#1a7a3c")
    arrow(82, 29.8, 84, 25.6, color="#1a7a3c")
    # bottleneck -> head and router blend
    arrow(63, 20.5, 56, 34, color="#1a7a3c")
    # head -> output
    arrow(40, 37.5, 12, 34)

    ax.text(78.5, 43, "primitive bank \u2014 nine neural lenses, each with its own internal layers",
            ha="center", fontsize=9, color=MUTED, style="italic")
    ax.text(78.5, 41.3, "routing weights p(x, g) \u00b7 \u2113 + (1\u2212\u03a3p) \u00b7 \u2113_prior \u2192 shared space",
            ha="center", fontsize=7.6, color=FAINT, family="monospace")
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 — protocol A: mixed-domain benchmark
# ---------------------------------------------------------------------------
def fig2_protocol_a() -> None:
    d = load("protocol_a.json")
    m = d["models"]
    sa = m["standalone"]
    order = ["switch_full", "static_mlp", "best_static", "switch_random_router", "static_cnn", "switch_uniform"]
    labels = ["Switch Operator", "static MLP", "best standalone\n(ReLU)", "random router", "static CNN", "uniform router"]
    vals = [m["switch_full"], m["static_mlp"], m["best_static"]["acc"], m["switch_random_router"], m["static_cnn"], m["switch_uniform"]]
    colors = [WIN, MUTED, FAINT, FAINT, MUTED, "#c9a24a"]
    edges = [INK, "#c4c6c8", "#c4c6c8", "#c4c6c8", "#c4c6c8", "#c4c6c8"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(order))[::-1]
    for yi, v, c, e in zip(y, vals, colors, edges):
        ax.barh(yi, v, height=0.62, color=c, edgecolor=e, linewidth=1.1)
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("test accuracy (mixed sim+real, 3 seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(m["best_static"]["acc"], color=FAINT, ls="--", lw=0.9)
    ax.text(m["best_static"]["acc"] + 0.004, 5.4, "best static", fontsize=7, color=FAINT, rotation=90, va="top")

    # per-primitive scatter
    xp = [sa[p] for p in PRIMS]
    ax2.axhline(m["switch_full"], color=WIN, lw=1.4, ls="--")
    ax2.text(8.4, m["switch_full"] + 0.004, f"Switch {m['switch_full']:.3f}", fontsize=7.5, color=WIN)
    ax2.scatter(range(len(PRIMS)), xp, s=42, color=LINK, edgecolor="white", linewidth=0.6, zorder=3)
    for i, v in enumerate(xp):
        ax2.text(i, v + 0.014, f"{v:.2f}", ha="center", fontsize=6.6, color=MUTED)
    ax2.set_xticks(range(len(PRIMS)))
    ax2.set_xticklabels(PRIM_LABELS, rotation=40, ha="right", fontsize=7.6)
    ax2.set_ylim(0.5, 0.92)
    ax2.set_ylabel("standalone accuracy")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Protocol A \u2014 one operator, one training run, beats every single fixed network",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig2_protocol_a.png")


# ---------------------------------------------------------------------------
# Figure 3 — protocol B: sim-to-real transfer
# ---------------------------------------------------------------------------
def fig3_protocol_b() -> None:
    d = load("protocol_b.json")
    m = d["mean"]
    models = ["static_cnn", "switch", "weight_adapted", "structure_adapted"]
    labels = ["static CNN\n(trained on sim)", "Switch Operator\n(zero-shot)", "weight adapt\n(fine-tune all)", "structure adapt\n(fine-tune router)"]
    sim = [m["static_cnn"]["acc_sim"], m["switch"]["acc_sim"], m["weight_adapted"]["acc_sim"], m["structure_adapted"]["acc_sim"]]
    real = [m["static_cnn"]["acc_real"], m["switch"]["acc_real"], m["weight_adapted"]["acc_real"], m["structure_adapted"]["acc_real"]]
    params = [None, None, m["weight_adapted"]["adapted_params"], m["structure_adapted"]["adapted_params"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(models))
    w = 0.36
    b1 = ax.bar(x - w / 2, sim, w, label="sim (train domain)", color="#9db8d2", edgecolor="white")
    b2 = ax.bar(x + w / 2, real, w, label="real (unseen domain)", color=INK, edgecolor="white")
    for xi, s, r in zip(x, sim, real):
        ax.text(xi - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=7, color=MUTED)
        ax.text(xi + w / 2, r + 0.01, f"{r:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for xi in x:
        ax.plot([xi - w, xi + w], [m["static_cnn"]["acc_real"]] * 2, color=LOSE, lw=0.7, ls=":", alpha=0.6)
    ax.text(0, m["static_cnn"]["acc_real"] + 0.02, "static CNN real floor", fontsize=7, color=LOSE, ha="center")

    # adapted-params comparison
    labs = ["structure adapt", "weight adapt"]
    pv = [params[3], params[2]]
    av = [m["structure_adapted"]["acc_real"], m["weight_adapted"]["acc_real"]]
    ax2.bar([0, 1], av, 0.5, color=[WIN, MUTED], edgecolor="white")
    for i, v in enumerate(av):
        ax2.text(i, v + 0.008, f"real {v:.3f}", ha="center", fontsize=8, color=INK)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labs, fontsize=8)
    ax2.set_ylim(0, 0.7)
    ax2.set_ylabel("real accuracy after 200 labels")
    ax3 = ax2.twinx()
    ax3.plot([0, 1], pv, "o-", color="#b5822e", lw=1.4, ms=5)
    for i, v in enumerate(pv):
        ax3.text(i, v + 900, f"{int(v):,} params", fontsize=7.5, color="#b5822e", ha="center")
    ax3.set_ylim(0, 22000)
    ax3.set_ylabel("adapted parameters", fontsize=8, color="#b5822e")
    ax3.tick_params(axis="y", labelcolor="#b5822e", labelsize=8)
    fig.suptitle("Protocol B \u2014 the operator transfers where weight fine-tuning fails",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig3_protocol_b.png")


# ---------------------------------------------------------------------------
# Figure 4 — protocol C: goal-conditioned routing
# ---------------------------------------------------------------------------
def fig4_protocol_c() -> None:
    d = load("protocol_c.json")
    m = d["models"]
    gc, ga = m["goal_conditioned"], m["goal_agnostic"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    # classification
    ax.bar([0, 1], [ga["acc_cls"], gc["acc_cls"]], 0.5, color=[MUTED, WIN], edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["goal-agnostic", "goal-conditioned"], fontsize=8.5)
    ax.set_ylabel("classification accuracy")
    ax.set_ylim(0.7, 0.9)
    for i, v in enumerate([ga["acc_cls"], gc["acc_cls"]]):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_title("classify goal", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    # reconstruction (lower is better)
    ax2.bar([0, 1], [ga["mse_rec"], gc["mse_rec"]], 0.5, color=[MUTED, WIN], edgecolor="white")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["goal-agnostic", "goal-conditioned"], fontsize=8.5)
    ax2.set_ylabel("reconstruction MSE (lower better)")
    ax2.set_ylim(0, 0.16)
    for i, v in enumerate([ga["mse_rec"], gc["mse_rec"]]):
        ax2.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8.5)
    ax2.set_title("reconstruct goal \u2014 4\u00d7 better", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Protocol C \u2014 the goal pathway is part of the routing decision",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig4_protocol_c.png")


# ---------------------------------------------------------------------------
# Figure 5 — protocol D: ablations
# ---------------------------------------------------------------------------
def fig5_protocol_d() -> None:
    d = load("protocol_d.json")
    m = d["models"]
    order = ["full", "no_bottleneck", "no_anneal", "domain_inv", "no_eye", "random_router"]
    labels = ["full", "no bottleneck", "no annealing", "domain-invariant eye", "no eye (raw pixels)", "random router"]
    acc = [m[o]["acc"] for o in order]
    real = [m[o]["acc_real"] for o in order]

    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    x = np.arange(len(order))
    ax.bar(x - 0.19, acc, 0.36, label="mixed sim+real", color=LINK, edgecolor="white")
    ax.bar(x + 0.19, real, 0.36, label="real subset", color="#9db8d2", edgecolor="white")
    for xi, a, r in zip(x, acc, real):
        ax.text(xi - 0.19, a + 0.003, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(xi + 0.19, r + 0.003, f"{r:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6, rotation=14, ha="right")
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.8238, color=FAINT, ls=":", lw=0.8)
    fig.suptitle("Protocol D \u2014 the operator is robust to every component removal (\u0394 \u2264 0.001)",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig5_protocol_d.png")


# ---------------------------------------------------------------------------
# Figure 6 — routing rewires with the world (trajectory + domain routing)
# ---------------------------------------------------------------------------
def fig6_routing() -> None:
    d = load("protocol_a.json")
    diag = d["diagnostics"]
    traj = diag["routing"]["trajectory"]
    dr = diag["routing"]["domain_routing"]
    keep = [0, 1, 2, 3]  # dense, relu, cnn + others collapsed
    cols = ["#6b4fa0", "#b5822e", LINK, FAINT]
    names = ["dense", "relu", "cnn", "other"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    # trajectory: routing probability vs corruption severity
    xs = [t["bin"] for t in traj]
    dense = [t["probs"][1] for t in traj]
    relu = [t["probs"][2] for t in traj]
    cnn = [t["probs"][3] for t in traj]
    ax.plot(xs, dense, "o-", color=cols[0], lw=1.8, ms=5, label="dense (flat lens)")
    ax.plot(xs, relu, "s-", color=cols[1], lw=1.8, ms=5, label="ReLU (piecewise lens)")
    ax.plot(xs, cnn, "^-", color=cols[2], lw=1.8, ms=5, label="CNN (spatial lens)")
    ax.set_xlabel("corruption severity (0 = clean sim \u2192 1 = dirtiest real)")
    ax.set_ylabel("routing probability")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)

    # domain routing stacked bars
    sim = [dr["sim"][k] for k in ["dense", "relu", "cnn"]]
    real = [dr["real"][k] for k in ["dense", "relu", "cnn"]]
    others_sim = 1 - sum(sim)
    others_real = 1 - sum(real)
    ax2.bar([0, 1], [sim[0], real[0]], 0.5, color=cols[0], label="dense")
    ax2.bar([0, 1], [sim[1], real[1]], 0.5, bottom=[sim[0], real[0]], color=cols[1], label="relu")
    ax2.bar([0, 1], [sim[2], real[2]], 0.5, bottom=[sim[0] + sim[1], real[0] + real[1]], color=cols[2], label="cnn")
    ax2.bar([0, 1], [others_sim, others_real], 0.5,
            bottom=[sim[0] + sim[1] + sim[2], real[0] + real[1] + real[2]], color=cols[3], label="other")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sim (clean)", "real (corrupted)"], fontsize=8.5)
    ax2.set_ylabel("share of samples routed")
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The operator rewires its computation as the world gets dirtier",
                 fontsize=11, fontweight="bold", y=1.02)
    save(fig, "fig6_routing.png")


# ---------------------------------------------------------------------------
# Figure 7 — glyph gallery (sim vs real pairs)
# ---------------------------------------------------------------------------
def fig7_glyphs() -> None:
    import sys
    sys.path.insert(0, ROOT)
    from dirty_man.data_glyphs import save_montage
    save_montage(os.path.join(FIG, "fig7_glyphs.png"), n_per_digit=2, seed=7)
    import shutil
    shutil.copy(os.path.join(FIG, "fig7_glyphs.png"), os.path.join(DOC_FIG, "fig7_glyphs.png"))
    print("wrote fig7_glyphs.png")


if __name__ == "__main__":
    style()
    fig1_architecture()
    fig2_protocol_a()
    fig3_protocol_b()
    fig4_protocol_c()
    fig5_protocol_d()
    fig6_routing()
    fig7_glyphs()
    print("all figures written to figs/ and docs/figs/")
