"""Build a fully self-contained Kaggle kernel.

The kernel embeds every source file (dirty_man package, run_experiments.py,
training_intervention.py) as string literals, writes them to /kaggle/working,
and runs the headline protocols. No git clone, no repo visibility required.

Run locally:  python kaggle/build_self_contained_kernel.py
Outputs:      kaggle/kernel_dirty_man_self_contained.py
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    "dirty_man/__init__.py",
    "dirty_man/switch_operator.py",
    "dirty_man/data_glyphs.py",
    "dirty_man/data_mnist.py",
    "run_experiments.py",
    "training_intervention.py",
]

BOILERPLATE_HEAD = '''"""The Dirty Man — headline experiments, self-contained (Kaggle GPU).

Runs the three headline protocols plus the training-time intervention:

  A  Mixed-domain: one Switch Operator beats every fixed network it contains.
  C  Goal pathway (FIXED): per-goal oracle supervision — the router genuinely
     specializes per goal (classify->relu, reconstruct->linear), with 7x
     better reconstruction than goal-agnostic.
  E  Real handwriting: everything trained on synthetic glyphs transfers
     zero-shot to real MNIST digits (0.334 > static CNN 0.313 > MLP 0.312).
  Intervention: the operator detects a learner's energy-conservation failure
     regime and routes to a Hamiltonian physics expert (16x less violation).

This kernel is self-contained: all source files are embedded below, written
to disk, then executed. Expected wall time on a Kaggle GPU: ~30-60 min.
"""

import os
import sys

# ---- embed the source files ---------------------------------------------
_SRC = {
'''

BOILERPLATE_MID = '''}

def _write_sources():
    for rel, content in _SRC.items():
        path = os.path.join(os.getcwd(), rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
    sys.path.insert(0, os.getcwd())


if __name__ == "__main__":
    _write_sources()
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
          flush=True)

    import run_experiments as re

    print("=" * 60, flush=True)
    print("PROTOCOL A: one operator beats every fixed network", flush=True)
    re.protocol_a(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  smoke=False, batch=128)

    print("=" * 60, flush=True)
    print("PROTOCOL C: goal-conditioned routing, per-goal oracle", flush=True)
    re.protocol_c(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  batch=128)

    print("=" * 60, flush=True)
    print("PROTOCOL E: real MNIST, zero synthetic overlap", flush=True)
    re.protocol_e(seeds=[0, 1, 2], epochs=12, n_train=6000, n_test=2000,
                  batch=128)

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
'''


def main():
    lines = [BOILERPLATE_HEAD]
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            # dirty_man/__init__.py may not exist; skip with empty
            content = ""
        else:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        lines.append(f"    {rel!r}: {content!r},\n")
    lines.append(BOILERPLATE_MID)
    out = os.path.join(ROOT, "kaggle", "kernel_dirty_man_self_contained.py")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
