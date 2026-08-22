"""Run the non-static layers comparison on SVHN real street-view photos.

Runs static (fixed path), router (per-sample per-depth program), and a coarse
whole-network MoE switch on the SAME data config, and writes a combined result
JSON. Designed to run detached in the background while the full-scale run
happens on the Kaggle GPU kernel.

    python run_ns_sweep.py --n-train 12000 --n-test 3000 \
        --static-epochs 8 --router-epochs 6 --switch-epochs 5
"""
import argparse
import json
import os
import sys
import time

import nonstatic_layers as nl

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
LOG = os.path.join(RESULTS, "ns_run.log")
COMBINED = os.path.join(RESULTS, "nonstatic_svhn.json")


def _checkpoint(combined):
    with open(COMBINED, "w") as f:
        json.dump(combined, f, indent=2)
    log("checkpointed " + COMBINED)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--static-epochs", type=int, default=8)
    ap.add_argument("--router-epochs", type=int, default=6)
    ap.add_argument("--switch-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main(argv=None):
    args = parse_args(argv)
    open(LOG, "w").close()
    cfg = dict(n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    log(f"SVHN comparison, uniform config {cfg}")

    combined = {"experiment": "nonstatic_layers_svhn", "data": "SVHN real street photos",
                "config": cfg}
    if os.path.exists(COMBINED):
        try:
            with open(COMBINED) as f:
                prev = json.load(f)
            if prev.get("config") == cfg:
                combined = prev
                log("resuming from existing checkpoint")
        except Exception:
            pass

    train_ds, test_ds = nl.load_svhn(**cfg)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=0)
    log(f"SVHN {len(train_ds)} train / {len(test_ds)} test")

    # --- static path (the fixed-path baseline) ---
    if "static" not in combined:
        log(f"=== static (conv3,mlp,relu) {args.static_epochs} epochs ===")
        t0 = time.time()
        sp = nl.StaticPath(path=("conv3", "mlp", "relu"))
        res = nl.train_model(sp, train_dl, test_dl, args.static_epochs, "static",
                             hard=False, switch=False)
        res["params"] = sp.n_params()
        res["acc"] = round(res["acc"], 4)
        combined["static"] = res
        _checkpoint(combined)
        log(f"static done: acc={res['acc']} params={res['params']} ({time.time()-t0:.0f}s)")
    else:
        log(f"static already checkpointed: acc={combined['static']['acc']}")

    # --- non-static router (per-sample program) ---
    if "router" not in combined:
        log(f"=== router (per-depth routing) {args.router_epochs} epochs ===")
        t0 = time.time()
        net = nl.NonStaticNet()
        res = nl.train_model(net, train_dl, test_dl, args.router_epochs, "router",
                             hard=False, switch=False)
        res["params"] = net.n_params()
        res["acc"] = round(res["acc"], 4)
        combined["router"] = res
        _checkpoint(combined)   # save before the (more fragile) program analysis
        log(f"router done: acc={res['acc']} params={res['params']} ({time.time()-t0:.0f}s)")

        # per-class program analysis (best-effort; never lose the router result)
        try:
            per_class = nl.collect_programs(
                net, test_dl, os.path.join(RESULTS, "ns_programs.json"))
            combined["program_file"] = "ns_programs.json"
            dominant = {}
            for c in range(10):
                dom = []
                for d in range(3):
                    cnt = per_class[c][d]   # int keys (as produced by collect_programs)
                    if cnt:
                        op_idx = max(cnt, key=cnt.get)
                        dom.append((net.names[d][int(op_idx)],
                                    round(cnt[op_idx] / sum(cnt.values()), 3)))
                    else:
                        dom.append(None)
                dominant[str(c)] = dom
            combined["dominant_lens_per_class"] = dominant
            log(f"dominant lenses per class: {json.dumps(dominant)}")
            _checkpoint(combined)
            log("checkpointed with dominant-lens program analysis")
        except Exception as e:
            log(f"program analysis failed (keeping router result): "
                f"{type(e).__name__} {e}")
    else:
        log(f"router already checkpointed: acc={combined['router']['acc']}")

    # --- coarse whole-network switch (MoE-style baseline) ---
    if "switch" not in combined:
        log(f"=== switch (whole-network MoE) {args.switch_epochs} epochs ===")
        t0 = time.time()
        sw = nl.build_coarse_switch()
        res = nl.train_model(sw, train_dl, test_dl, args.switch_epochs, "switch",
                             hard=False, switch=True)
        res["params"] = sw.n_params()
        res["acc"] = round(res["acc"], 4)
        combined["switch"] = res
        _checkpoint(combined)
        log(f"switch done: acc={res['acc']} params={res['params']} ({time.time()-t0:.0f}s)")
    else:
        log(f"switch already checkpointed: acc={combined['switch']['acc']}")

    # --- summary ---
    log("=== SUMMARY ===")
    for k in ("static", "router", "switch"):
        if k in combined:
            r = combined[k]
            log(f"{k}: acc={r['acc']} params={r['params']}")
    _checkpoint(combined)
    log(f"wrote {COMBINED}")


if __name__ == "__main__":
    try:
        main()
        log("ALL DONE")
    except Exception as e:
        import traceback
        log(f"FAILED: {traceback.format_exc()}")
        sys.exit(1)
