"""Few-shot sim->real transfer probe: structure (routing) vs weight adaptation.

Usage:
    python probe_transfer.py struct   # structure adaptation sweep
    python probe_transfer.py weight   # weight adaptation sweep
Results appended to results/probe_transfer.jsonl.
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
                             train_model, eval_cls, freeze, set_seed)

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


def run_struct():
    sim_train, _ = make_datasets(n_train, n_test, domain='sim')
    _, test_ds = make_datasets(n_train, n_test, domain='mixed')
    bank = get_bank('B', sim_train, 6, 0)
    set_seed(777)
    sw = SwitchOperator(n_classes=10)
    train_model(sw, sim_train, epochs, 777, tag='switch(sim)', verbose=False,
                batch=128, bank=bank)
    real_ds, _ = make_datasets(400, 100, domain='real')
    for n_ft in [20, 50, 100, 200]:
        ft = TensorDataset(*[t[:n_ft] for t in real_ds.tensors])
        sw2 = SwitchOperator(n_classes=10)
        sw2.load_state_dict(sw.state_dict())
        freeze(sw2.primitives, True); freeze(sw2.eye, True)
        n_adapt = sum(p.numel() for p in sw2.parameters() if p.requires_grad)
        opt = torch.optim.Adam([p for p in sw2.parameters() if p.requires_grad], lr=5e-4)
        for _ in range(6):
            sw2.train()
            for xb, yb, db, sb, rb in make_loader(ft, batch=min(64, n_ft)):
                out, info = sw2(xb, tau=0.5, hard=False)
                loss = F.cross_entropy(out, yb) + 0.5 * F.cross_entropy(
                    info['logits'], oracle_targets(bank, xb, yb))
                opt.zero_grad(); loss.backward(); opt.step()
        r = eval_cls(sw2, test_ds)
        row = {"kind": "struct", "n_real": n_ft, "real": round(r['acc_real'], 4),
               "sim": round(r['acc_sim'], 4), "params": n_adapt}
        append(os.path.join(OUT, "probe_transfer.jsonl"), row)
        print(row, flush=True)


def run_weight():
    sim_train, _ = make_datasets(n_train, n_test, domain='sim')
    _, test_ds = make_datasets(n_train, n_test, domain='mixed')
    real_ds, _ = make_datasets(400, 100, domain='real')
    for n_ft in [20, 50, 100, 200]:
        ft = TensorDataset(*[t[:n_ft] for t in real_ds.tensors])
        cnn2 = train_static(static_cnn, sim_train, epochs, 1)
        n_adapt = sum(p.numel() for p in cnn2.parameters())
        opt2 = torch.optim.Adam(cnn2.parameters(), lr=5e-4)
        for _ in range(3):
            cnn2.train()
            for xb, yb, db, sb, rb in make_loader(ft, batch=min(64, n_ft)):
                loss = F.cross_entropy(cnn2(xb), yb)
                opt2.zero_grad(); loss.backward(); opt2.step()
        r = eval_cls(cnn2, test_ds)
        row = {"kind": "weight", "n_real": n_ft, "real": round(r['acc_real'], 4),
               "sim": round(r['acc_sim'], 4), "params": n_adapt}
        append(os.path.join(OUT, "probe_transfer.jsonl"), row)
        print(row, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "struct"
    (run_struct if which == "struct" else run_weight)()
