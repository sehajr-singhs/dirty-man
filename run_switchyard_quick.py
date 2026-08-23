"""Quick switchyard comparison at 3 key severity levels."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import switchyard_vs_moe as sm

results = []
for sev in [0.0, 0.75, 1.5]:
    print(f"\n{'='*50}\nSeverity = {sev}\n{'='*50}")
    r = sm.run_comparison(n_train=2000, n_test=500, epochs=12, severity=sev, seed=42)
    r["severity"] = sev
    results.append(r)
    moe = r["moe_content_routing"]["overall_acc"]
    dm = r["dirty_man_meta_routing"]["overall_acc"]
    moe_ra = r["key_finding"]["moe_routing_accuracy"]
    dm_ra = r["key_finding"]["dirtyman_routing_accuracy"]
    print(f"\n  SUMMARY sev={sev}: MoE acc={moe:.3f} ra={moe_ra:.3f} | DM acc={dm:.3f} ra={dm_ra:.3f}")

os.makedirs("results", exist_ok=True)
with open("results/switchyard_vs_moe_sweep.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\n=== SEVERITY SWEEP SUMMARY ===")
for r in results:
    s = r["severity"]
    m = r["moe_content_routing"]["overall_acc"]
    d = r["dirty_man_meta_routing"]["overall_acc"]
    mra = r["key_finding"]["moe_routing_accuracy"]
    dra = r["key_finding"]["dirtyman_routing_accuracy"]
    print(f"  sev={s:.2f}: MoE acc={m:.3f} route_acc={mra:.3f} | DM acc={d:.3f} route_acc={dra:.3f}")
