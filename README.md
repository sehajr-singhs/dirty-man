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

The router's policy is interpretable: on clean sim it uses dense (flat lens) and cnn; as corruption grows it abandons dense and shifts to relu/cnn (spatial lenses). On real handwritten digits it routes to cnn/dense — it has learned to identify which lens the input needs. The NMI paper (`docs/papers/nmi_paper.tex`) now includes a theory section (Sec. Theory) with three theorems: oracle-supervised routing learns the regime policy (VC-dimension bound), structural adaptation needs fewer labels than weight adaptation (sample-complexity bound), and the shared bottleneck keeps switching geometry-continuous.

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
run_experiments.py    protocols A–E, staged training, checkpoint/resume, result JSONs
make_figs.py          figures 1–7 from results/*.json (nothing hard-coded)
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
- The staged training protocol and temperature schedule are delicate; hyperparameters were tuned for this benchmark.

## Papers

- `docs/papers/nmi_paper.tex` → `nmi_paper.pdf` (Nature-Machine-Intelligence-style, `xelatex nmi_paper.tex`)
- `docs/papers/ieee_paper.tex` → `ieee_paper.pdf` (IEEE conference style, `pdflatex ieee_paper.tex`)

## License

MIT — see [LICENSE](LICENSE).
