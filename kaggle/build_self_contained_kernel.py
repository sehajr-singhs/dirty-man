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
    "flagship_regime_routing.py",
    "flagship_discovered_law.py",
    "nonstatic_layers.py",
    "run_ns_sweep.py",
    "sarcos_routing.py",
]

BOILERPLATE_HEAD = '''"""The Dirty Man — headline experiments, self-contained (Kaggle GPU).

Runs the three headline protocols plus the training-time intervention and the
flagship:

  A  Mixed-domain: one Switch Operator beats every fixed network it contains.
  C  Goal pathway (FIXED): per-goal oracle supervision — the router genuinely
     specializes per goal (classify->relu, reconstruct->linear), with 7x
     better reconstruction than goal-agnostic.
  E  Real handwriting: everything trained on synthetic glyphs transfers
     zero-shot to real MNIST digits (0.334 > static CNN 0.313 > MLP 0.312).
  Intervention: the operator detects a learner's energy-conservation failure
     regime and routes to a Hamiltonian physics expert (16x less violation).
  FLAGSHIP: a regime-switching pendulum where no single network obeys all
     laws but routing between exact physics harnesses does (up to 190x lower
     energy error).

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


def _run(name, fn):
    """Run one protocol; on failure write the full traceback to a file so the
    kernel output is self-diagnosing (Kaggle's own log is not always
    downloadable)."""
    print("=" * 60, flush=True)
    print(name, flush=True)
    try:
        fn()
        print(f"[OK] {name}", flush=True)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        os.makedirs("results", exist_ok=True)
        with open(os.path.join("results", "kernel_error.txt"), "w") as f:
            f.write(f"{name}\\n{tb}")
        print(f"[FAILED] {name} — traceback written to results/kernel_error.txt",
              flush=True)


if __name__ == "__main__":
    _write_sources()
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}",
          flush=True)

    import run_experiments as re

    _run("PROTOCOL A: one operator beats every fixed network",
         lambda: re.protocol_a(seeds=[0, 1, 2], epochs=12, n_train=6000,
                               n_test=2000, smoke=False, batch=128))

    _run("PROTOCOL C: goal-conditioned routing, per-goal oracle",
         lambda: re.protocol_c(seeds=[0, 1, 2], epochs=12, n_train=6000,
                               n_test=2000, batch=128))

    _run("PROTOCOL E: real MNIST, zero synthetic overlap",
         lambda: re.protocol_e(seeds=[0, 1, 2], epochs=12, n_train=6000,
                               n_test=2000, batch=128))

    _run("TRAINING-TIME INTERVENTION: energy conservation",
         lambda: (__import__("training_intervention").main()))

    _run("FLAGSHIP: no single network obeys all laws; routing does",
         lambda: (__import__("flagship_regime_routing").main()))

    _run("FLAGSHIP 2 (discovered law): learned specialists, no hardcoded physics",
         lambda: (__import__("flagship_discovered_law").main(
             n_traj=400, epochs=800, n_test=200, test_steps=100)))

    _run("NON-STATIC LAYERS: per-sample program routing on real SVHN photos",
         lambda: (__import__("run_ns_sweep").main(
             ["--n-train", "20000", "--n-test", "6000",
              "--static-epochs", "8", "--router-epochs", "10",
              "--switch-epochs", "5"])))

    _run("SARCOS: matched-budget non-static routing on real robot telemetry",
         lambda: (__import__("sarcos_routing").main(
             ["--static-epochs", "30", "--routed-epochs", "30",
              "--device", "auto"])))

    print("=" * 60, flush=True)
    print("ALL DONE — results written to results/*.json", flush=True)
    for f in ["protocol_a.json", "protocol_c.json", "protocol_e.json",
              "training_intervention.json", "flagship_regime_routing.json",
              "flagship_discovered_law.json", "nonstatic_svhn.json",
              "sarcos_routing.json"]:
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
