# The Dirty Man

The Dirty Man is a self-reconfiguring neural architecture that rewires which network computes each sample — driven by what it sees and what it wants. Its core component is the **Switch Operator**: a gating mechanism over nine neural *primitives* — **linear, dense, ReLU, CNN, RNN, LSTM, GAN, autoencoder, transformer** — each with its own internal layers and inductive lens. A visual-cue **eye** watches the input, a **goal pathway** encodes the task, and a differentiable **router** decides, per sample, which primitive (or soft combination) processes it. Every primitive emits into a shared bottleneck latent space, so switched paths stay geometry-continuous for the downstream head. Routing uses annealed Gumbel-Softmax (soft during training, hard at deployment) and a staged protocol — warm-start primitives, oracle-supervised router training, joint fine-tune — that prevents mixture collapse and yields a *readable routing policy*.

```text
input ──► [eye: visual cues] ─┐
          [goal pathway] ─────┼──► [router] ──► gating weights p
                              │        │
input ──► primitive_1 ────────┤        │
input ──► primitive_2 ────────┴──► bottleneck ──► p-weighted mixture ──► head ──► out
```

**Headline results** (procedural sim-to-real glyph benchmark, 3 seeds — every number reproduced from `results/*.json`):

| Protocol | Result |
|---|---|
| **A — mixed-domain** | Switch Operator **0.834** > static MLP 0.829 > best standalone (ReLU) 0.822 > static CNN 0.789 > uniform router 0.753 |
| **B — sim→real transfer** | zero-shot real 0.474 vs static CNN 0.388 (gap 0.526 vs 0.611); structure adaptation (4,659 params) 0.475/0.9997 vs weight adaptation (16,330 params) 0.395/0.984 |
| **C — goal pathway** | goal-conditioned cls 0.821 / recon MSE 0.034 vs goal-agnostic 0.810 / 0.139 (**4× better recon**) |
| **D — ablations** | removing bottleneck / annealing / eye / domain-invariance changes accuracy by ≤0.001 |

The router's policy is interpretable: on clean sim it uses dense (flat lens) and cnn; as corruption grows it abandons dense and shifts to relu/cnn (spatial lenses).

## Getting started

```bash
pip install -r requirements.txt

# 4 protocols at tiny scale (~2 min, sanity check)
python run_experiments.py --smoke

# Full experiments (each checkpointed per item; killed runs resume)
python run_experiments.py --only A --seeds 3    # mixed-domain benchmark
python run_experiments.py --only B --seeds 3    # sim-to-real transfer
python run_experiments.py --only C --seeds 3    # goal-conditioned routing
python run_experiments.py --only D --seeds 3    # ablations

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
run_experiments.py    protocols A–D, staged training, checkpoint/resume, result JSONs
make_figs.py          figures 1–7 from results/*.json (nothing hard-coded)
make_playground.py    docs/playground.html from results/playground.json
results/              committed result files (protocol_*.json, chk_*.json, banks)
docs/                 website (index.html, playground.html, figs/, papers/)
docs/papers/          LaTeX sources + PDFs (nmi_paper.tex, ieee_paper.tex)
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

- The mixed-domain margin over the best static is small — when the bank is strong and data homogeneous, routing is a bonus, not a revolution. The transfer result (Protocol B) is where structural adaptation is decisive.
- Protocol C's router converges to a single lens per goal; the win there comes from goal-dedicated heads. Per-goal structural specialization is the next step.
- All experiments are on the synthetic glyph benchmark; real video/image domains are the essential next test.
- The staged training protocol and temperature schedule are delicate; hyperparameters were tuned for this benchmark.

## Papers

- `docs/papers/nmi_paper.tex` → `nmi_paper.pdf` (Nature-Machine-Intelligence-style, `xelatex nmi_paper.tex`)
- `docs/papers/ieee_paper.tex` → `ieee_paper.pdf` (IEEE conference style, `pdflatex ieee_paper.tex`)

## License

MIT — see [LICENSE](LICENSE).
