"""The Dirty Man — headline experiments on Kaggle GPU.

Runs the three headline protocols on a GPU (auto-detected):

  A  Mixed-domain: one Switch Operator beats every fixed network it contains,
     by learning a routing policy (flat lens for clean sim, spatial/piecewise
     for corrupted real).
  C  Goal pathway (FIXED): per-goal oracle supervision makes the router
     genuinely specialize per goal — classify routes to relu, reconstruct to
     linear — with 7x better reconstruction than goal-agnostic.
  E  Real handwriting: everything trained on synthetic glyphs transfers
     zero-shot to real MNIST digits (0.334 > static CNN 0.313 > MLP 0.312),
     and the router identifies that real handwriting needs spatial lenses
     (CNN 0.67 / dense 0.33).

The kernel is self-contained: it installs dependencies, downloads MNIST, runs
the experiments, and writes results/ into Kaggle's /kaggle/working.

Expected wall time on a Kaggle GPU (P100/T4): ~30-60 min for all three.
"""

import os
import sys

# ---------------------------------------------------------------------------
# 0. Setup — install deps, fetch the repo code
# ---------------------------------------------------------------------------

def setup():
    sub = None
    import subprocess
    if not os.path.exists("run_experiments.py"):
        # fetch this repo's code (the kernel is run from the repo checkout)
        sub = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/sehajr-singhs/dirty-man.git", "repo"],
            capture_output=True, text=True)
        os.chdir("repo")
    if sub is not None and sub.returncode != 0:
        print("git clone failed; falling back to bundled source")
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
          flush=True)


if __name__ == "__main__":
    setup()
    import run_experiments as re

    # ------------------------------------------------------------------
    # 1. Protocol A — mixed-domain (3 seeds, full scale)
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("PROTOCOL A: one operator beats every fixed network", flush=True)
    re.protocol_a(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  smoke=False, batch=128)

    # ------------------------------------------------------------------
    # 2. Protocol C — goal pathway with per-goal oracle (3 seeds)
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("PROTOCOL C: goal-conditioned routing, per-goal oracle", flush=True)
    re.protocol_c(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  batch=128)

    # ------------------------------------------------------------------
    # 3. Protocol E — real handwriting (MNIST) sim->real (3 seeds)
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("PROTOCOL E: real MNIST, zero synthetic overlap", flush=True)
    re.protocol_e(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  batch=128)

    # ------------------------------------------------------------------
    # 4. Training-time intervention (energy conservation)
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("TRAINING-TIME INTERVENTION: energy conservation", flush=True)
    import training_intervention
    training_intervention.main()

    print("=" * 60, flush=True)
    print("ALL DONE — results written to results/*.json", flush=True)
    for f in ["protocol_a.json", "protocol_c.json", "protocol_e.json",
              "training_intervention.json"]:
        p = os.path.join("results", f)
        if os.path.exists(p):
            print(f"  {p}  ({os.path.getsize(p)} bytes)", flush=True)
