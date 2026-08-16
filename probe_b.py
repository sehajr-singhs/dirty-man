"""Protocol B redesign probe: mixed-toolbox bank + sim-only router.

The bank is pre-trained on mixed (sim + corruptions) data with specialist
niches — the "toolbox". The operator's router/eye/head train on clean sim
only — the deployment-time experience. Then 200 real labels adapt either
the weights of a static CNN or just the operator's routing stack.

Usage:
    python probe_b.py           # structure vs weight adaptation sweep
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from dirty_man.switch_operator import SwitchOperator, Standalone, PRIMITIVES
from run_experiments import (get_bank, make_datasets, make_loader, train_static,
                             train_model, eval_cls, freeze, set_seed, SPECIALISTS)

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
epochs, n_train, n_test = 12, 6000, 2000


def static_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 6 * 6, 10))


def oracle_targets(bank, xb, yb):
    with torch.no_grad():
        ce = torch.zeros(len(PRIMITIVES), xb.size(0))
        for i, nm in enumerate(PRIMITIVES):
            prim = Standalone(nm, n_classes=10)
            prim.prim.load_state_dict(bank['prims'][nm]); prim.eval()
            h = nn.Linear(64, 10); h.load_state_dict(bank['heads'][nm]); h.eval()
            ce[i] = F.cross_entropy(h(prim.prim(xb)), yb, reduction='none')
    return ce.argmin(0)


def append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    mixed_train, _ = make_datasets(n_train, n_test)          # the toolbox world
    sim_train, _ = make_datasets(n_train, n_test, domain='sim')   # deployment sim
    _, test_ds = make_datasets(n_train, n_test, domain='mixed')
    real_ds, _ = make_datasets(400, 100, domain='real')

    # the toolbox: specialist bank pre-trained on MIXED data (cached from prot A)
    bank = get_bank('A', mixed_train, epochs, 0, subsets=SPECIALISTS)

    # operator router/eye/head trained on clean sim only
    set_seed(777)
    sw = SwitchOperator(n_classes=10)
    train_model(sw, sim_train, epochs, 777, tag='switch(sim)', verbose=False,
                batch=128, bank=bank)
    r0 = eval_cls(sw, test_ds)
    print('unadapted switch: sim %.3f real %.3f' % (r0['acc_sim'], r0['acc_real']), flush=True)
    append(os.path.join(OUT, "probe_b.jsonl"),
           {"kind": "unadapted_switch", "real": round(r0['acc_real'], 4),
            "sim": round(r0['acc_sim'], 4)})

    cnn = train_static(static_cnn, sim_train, epochs, 0)
    rc = eval_cls(cnn, test_ds)
    print('static cnn (sim train): sim %.3f real %.3f' % (rc['acc_sim'], rc['acc_real']), flush=True)
    append(os.path.join(OUT, "probe_b.jsonl"),
           {"kind": "static_cnn", "real": round(rc['acc_real'], 4),
            "sim": round(rc['acc_sim'], 4)})

    for n_ft in [50, 100, 200]:
        ft = TensorDataset(*[t[:n_ft] for t in real_ds.tensors])

        # ---- structure adaptation: router + heads only (4.7k params) ----
        sw2 = SwitchOperator(n_classes=10)
        sw2.load_state_dict(sw.state_dict())
        freeze(sw2.primitives, True); freeze(sw2.eye, True)
        n_adapt = sum(p.numel() for p in sw2.parameters() if p.requires_grad)
        opt = torch.optim.Adam([p for p in sw2.parameters() if p.requires_grad], lr=5e-4)
        for _ in range(8):
            sw2.train()
            for xb, yb, db, sb, rb in make_loader(ft, batch=min(64, n_ft)):
                out, info = sw2(xb, tau=0.5, hard=False)
                loss = F.cross_entropy(out, yb) + 0.7 * F.cross_entropy(
                    info['logits'], oracle_targets(bank, xb, yb))
                opt.zero_grad(); loss.backward(); opt.step()
        r_s = eval_cls(sw2, test_ds)

        # ---- weight adaptation: whole static CNN (16.3k params) ----
        cnn2 = train_static(static_cnn, sim_train, epochs, 1)
        n_w = sum(p.numel() for p in cnn2.parameters())
        opt2 = torch.optim.Adam(cnn2.parameters(), lr=5e-4)
        for _ in range(3):
            cnn2.train()
            for xb, yb, db, sb, rb in make_loader(ft, batch=min(64, n_ft)):
                loss = F.cross_entropy(cnn2(xb), yb)
                opt2.zero_grad(); loss.backward(); opt2.step()
        r_w = eval_cls(cnn2, test_ds)

        row = {"n_real": n_ft,
               "struct_real": round(r_s['acc_real'], 4), "struct_sim": round(r_s['acc_sim'], 4),
               "weight_real": round(r_w['acc_real'], 4), "weight_sim": round(r_w['acc_sim'], 4),
               "struct_params": n_adapt, "weight_params": n_w}
        append(os.path.join(OUT, "probe_b.jsonl"), row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
