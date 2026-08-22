# The Dirty Man

The Dirty Man is a self-reconfiguring neural architecture that rewires which network computes each sample — driven by what it sees and what it wants. Its core component is the **Switch Operator**: a gating mechanism over nine neural *primitives* — **linear, dense, ReLU, CNN, RNN, LSTM, GAN, autoencoder, transformer** — each with its own internal layers and inductive lens. A visual-cue **eye** watches the input, a **goal pathway** encodes the task, and a differentiable **router** decides, per sample, which primitive (or soft combination) processes it. Every primitive emits into a shared bottleneck latent space, so switched paths stay geometry-continuous for the downstream head. Routing uses annealed Gumbel-Softmax (soft during training, hard at deployment) and a staged protocol — warm-start primitives, oracle-supervised router training, joint fine-tune — that prevents mixture collapse and yields a *readable routing policy*.

```text
input ──► [eye: visual cues] ─┐
          [goal pathway] ─────┼──► [router] ──► gating weights p
                              │        │
input ──► primitive_1 ────────┤        │
input ──► primitive_2 ────────┴──► bottleneck ──► p-weighted mixture ──► head ──► out
```

**Headline results** (3 seeds — every number reproduced from `results/*.json`):

| Protocol | Result |
|---|---|
| **A — mixed-domain** | Switch Operator **0.834** > static MLP 0.829 > best standalone (ReLU) 0.822 > static CNN 0.789 > uniform router 0.753 |
| **B — sim→real transfer** | zero-shot real 0.474 vs static CNN 0.388 (gap 0.526 vs 0.611); structure adaptation (4,659 params) 0.475/0.9997 vs weight adaptation (16,330 params) 0.395/0.984 |
| **C — goal pathway** (per-goal oracle) | goal-conditioned cls 0.821 / recon MSE **0.020** vs goal-agnostic 0.817 / 0.143 (**7× better recon**); the router now *specializes per goal* — classify→relu, reconstruct→linear |
| **D — ablations** | removing bottleneck / annealing / eye / domain-invariance changes accuracy by ≤0.001 |
| **E — real handwriting (MNIST)** | everything trained on synthetic glyphs transfers zero-shot to real MNIST digits: Switch Operator **0.334** > static CNN 0.313 > static MLP 0.312; structure adaptation (4.7k params) 0.433 ≈ weight adaptation (16.3k params) 0.432; the router identifies real handwriting needs spatial lenses (CNN 0.67 / dense 0.33) |

**Beyond the benchmark — the Dirty Man as a training assistant.** A vanilla learner trained only on calm pendulum orbits fails on energetic ones (one-step energy violation 0.80 vs 0.048). The Dirty Man's eye+router detects the failure regime and routes those samples to a Hamiltonian physics expert that conserves energy by construction: per-step energy violation drops **16×** (0.42→0.026), and the intervention is *selective* — it fires on high-kinetic samples (0.88) and leaves calm ones alone (0.07). See `training_intervention.py`.

**The flagship — routing does what no single network can.** A pendulum obeys three different laws in three regimes: conservative (energy conserved), damped (energy decays), driven (energy pumped). Each law demands a mutually-exclusive inductive bias. The system embeds each law as an *exact, closed-form physics harness* (inverted design), and a router learns — from a 16-step trajectory — to detect which law governs (energy flat vs decaying vs pumped) at **92–99% accuracy** (full scale, T4 GPU). Every fixed expert is exact on its own regime (0.000) and wrong elsewhere (3.5–8.4); a single brute-force MLP fails on *every* regime (6.6–12.4); the routed system is 5× better even on its worst regime, with **up to 336× lower energy error** (0.001–0.092 vs 0.297–0.549). See `flagship_regime_routing.py`.

**The discovered-law flagship — no physics is given.** A fair objection to the flagship above is that the laws were hardcoded. The discovered-law variant (`flagship_discovered_law.py`) closes that gap: every expert is a *learned* network, and no law is provided to any learned component. The inverted-design principle moves inside each expert — the known pendulum skeleton is the unchangeable harness, and each specialist learns only the *residual force law* of its regime (~0, ~−bω, ~A sin Ωt). Learning the residual on an exact skeleton (instead of raw next-state dynamics) is what keeps rollouts stable. Learned specialists are near-exact on their own regime; the static MLP still fails on every regime; the routed system is **5–720× better** (0.004–0.808 vs 0.96–2.89), tracking the oracle specialist, with the router discovering the law at 93–99%. And as the number of governing laws grows (2→3→4), a single map is stuck at its error floor while routing stays an order of magnitude lower (4.5–9× better at every law count) — each law gets its own specialist instead of one compromise.

**Non-static layers on real robot dynamics — routing *inside* the network.** The sharpest form of the thesis isn't switching between whole networks (that's MoE) but routing *inside* one network: at every depth a router selects the primitive *layer* for *that* sample, so each input walks its own program of computation. On **SARCOS** — real 7-DOF robot-arm telemetry (44,484 train → 7 torques) — the routed program beats the best of nine fixed paths (0.0185 vs 0.0198 NMSE, −6.4%) and its choice is physically interpretable: it routes slow configurations to the linear lens (‖q̇‖ = 1.62) and fast ones to the nonlinear lens (‖q̇‖ = 2.56). On SVHN (real street photos) it also runs per-sample programs. See `sarcos_routing.py` and `nonstatic_layers.py`.

The router's policy is interpretable: on clean sim it uses dense (flat lens) and cnn; as corruption grows it abandons dense and shifts to relu/cnn (spatial lenses). On real handwritten digits it routes to cnn/dense — it has learned to identify which lens the input needs. The NMI paper (`docs/papers/nmi_paper.tex`) includes a theory section with five theorems: oracle-supervised routing learns the regime policy (VC-dimension bound), structural adaptation needs fewer labels than weight adaptation (sample-complexity bound), the shared bottleneck keeps switching geometry-continuous, a **single-map separation bound** — a fixed map must mis-represent at least one law by half the per-step energy gap, while routing drives the gap to zero — and a **scaling bound** — a single map's error is bounded below by a constant independent of the number of laws, while routing's error decays with router accuracy, so more laws hurt one map, not routing.

## Getting started

```bash
pip install -r requirements.txt

# 5 protocols at tiny scale (~2 min, sanity check)
python run_experiments.py --smoke

# Full experiments (each checkpointed per item; killed runs resume)
python run_experiments.py --only A --seeds 3    # mixed-domain benchmark
python run_experiments.py --only B --seeds 3    # sim-to-real transfer
python run_experiments.py --only C --seeds 3    # goal-conditioned routing
python run_experiments.py --only D --seeds 3    # ablations
python run_experiments.py --only E --seeds 3    # real handwriting (MNIST) sim->real

# Training-time intervention (energy conservation)
python training_intervention.py

# THE FLAGSHIP: regime-switching pendulum — no single network, routing does
python flagship_regime_routing.py

# DISCOVERED-LAW FLAGSHIP: learned specialists, no hardcoded physics
python flagship_discovered_law.py

# NON-STATIC LAYERS: per-sample program routing on real SVHN photos
python nonstatic_layers.py --model router --epochs 15

# REAL ROBOT: per-depth routing on SARCOS inverse dynamics
python sarcos_routing.py

# Figures + interactive playground (read committed results/*.json)
python make_figs.py
python make_playground.py

# Tests
python -m pytest tests/ -q
```

CLI options: `--epochs` (default 12), `--n_train` (6000), `--n_test` (2000), `--batch` (128), `--seed-start`, `--smoke`, and `--device cuda|cpu` (default: auto-detect — CUDA is used automatically when available; batches and models are moved to the device throughout).

## Repository layout

```text
dirty_man/            core package
  switch_operator.py  the Switch Operator model: eye, goal pathway, router, 9 primitives, Gumbel routing
  data_glyphs.py      procedural sim/real glyph benchmark (no downloads)
  data_mnist.py       real MNIST handwritten digits (downloaded once) in the standard schema
training_intervention.py  the Dirty Man as a training assistant: detects a learner's failure regime and intervenes
flagship_regime_routing.py  THE FLAGSHIP: a single map cannot obey two laws; routing can (inverted design)
flagship_discovered_law.py  DISCOVERED-LAW FLAGSHIP: learned specialists, no hardcoded physics + scaling curve
nonstatic_layers.py  NON-STATIC LAYERS: per-sample, per-depth program routing on real SVHN photos
sarcos_routing.py    REAL ROBOT: per-depth routing on SARCOS inverse dynamics (interpretable lens selection)
run_experiments.py    protocols A–E, staged training, checkpoint/resume, result JSONs
make_figs.py          figures 1–8 from results/*.json (nothing hard-coded)
make_playground.py    docs/playground.html from results/playground.json
results/              committed result files (protocol_*.json, chk_*.json, banks)
docs/                 website (index.html, playground.html, figs/, papers/)
docs/papers/          LaTeX sources + PDFs (nmi_paper.tex, ieee_paper.tex)
kaggle/               Kaggle GPU kernel (runs protocols A, C, E + intervention on a GPU)
                      → https://www.kaggle.com/code/sehajrsingh/dirty-man-headline-experiments
tests/                sanity tests
```

## How it works

1. **Primitive bank.** Nine primitives process the same input in parallel, each with its own inductive lens and task head.
2. **Eye + goal pathway.** A small conv encoder extracts visual cues `e = Eye(x) ∈ ℝ³²`; the goal pathway embeds the task `g ∈ ℝ¹⁶`.
3. **Router.** `z = Router([e, g])` produces logits over the 9 primitives; Gumbel-Softmax with annealed temperature turns them into routing weights `p`.
4. **Shared bottleneck.** Each primitive emits into ℝ⁶⁴; the mixture `ℓ = Σ pₖ ℓₖ` feeds the task head, so structure switching never changes downstream geometry.
5. **Staged training.** Warm-start primitives (optionally on specialist subsets) → train eye+router on regime-level oracle targets with primitives frozen → joint fine-tune → deploy with hard (argmax) routing. Without this, an untrained router collapses onto the most expressive primitive and switching never happens.

## The benchmark

Digits 0–9 are rendered procedurally as anti-aliased parametric strokes in 24×24 (no downloads, fully reproducible). **sim** = clean render; **real** = sensor-corrupted render (motion blur, Gaussian noise, brightness/contrast drift, occlusion, JPEG-style block quantization, affine warp). Every sample carries a *severity* in [0,1] and a *regime* label (clean/spatial/statistical) so the routing policy can be measured against corruption. Train/test splits use disjoint random streams (no leakage).

## Honest limitations

- The mixed-domain margin over the best static is small — when the bank is strong and data homogeneous, routing is a bonus, not a revolution. The transfer results (Protocols B and E) are where structural adaptation is decisive.
- On real MNIST, structure adaptation (4.7k params) matches — but does not beat — weight adaptation (16.3k params); its advantage is the 3.5× smaller adaptation surface and preserved source performance (Protocol B).
- The real-data test is single-digit grayscale (MNIST); real video and natural-image domains are the essential next test.
- The per-goal oracle needs per-primitive reconstruction heads; a cheaper oracle would broaden applicability.
- The training-time intervention is a proof of concept on pendulum dynamics; scaling it to real training loops is future work.
- The discovered-law flagship replaces hardcoded experts with learned specialists on exact physics skeletons; the remaining simplification is that the *skeleton* itself (the known (g/L) sin θ torque) is still supplied. Learning the skeleton from data — full discovery of both law and structure — is the direct next step.
- The staged training protocol and temperature schedule are delicate; hyperparameters were tuned for this benchmark.

## Papers

- `docs/papers/nmi_paper.tex` → `nmi_paper.pdf` (Nature-Machine-Intelligence-style, `xelatex nmi_paper.tex`)
- `docs/papers/ieee_paper.tex` → `ieee_paper.pdf` (IEEE conference style, `pdflatex ieee_paper.tex`)

## License

MIT — see [LICENSE](LICENSE).
