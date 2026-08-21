#!/usr/bin/env python3
"""Render the probe result JSONs into the markdown tables used by DENSITY_PROBES.md."""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def d1():
    d = load("d1_pruning.json")
    if not d:
        return
    print("### D1 controls\n")
    print("| control | expect | BPB | delta | fired |")
    print("|---|---|---|---|---|")
    for c in d["controls"]:
        print(f"| {c['tag']} | {c.get('expect','')} | {c['bpb']:.5f} | {c['delta']:+.5f} | "
              f"{'YES' if c.get('fired') else 'no'} |")
    sw = d["organ_sweep"]
    organs = []
    for r in sw:
        if r["organ"] not in organs:
            organs.append(r["organ"])
    levels = d["levels"]
    for mode in ("unstructured", "structured"):
        rows = [r for r in sw if r["mode"] == mode]
        if not rows:
            continue
        print(f"\n### D1 organ sweep — {mode} (delta BPB vs baseline {d['baseline_bpb']:.5f})\n")
        print("| organ | axis | " + " | ".join(f"{int(l*100)}%" for l in levels) + " |")
        print("|---" * (len(levels) + 2) + "|")
        for o in organs:
            rr = {r["level"]: r for r in rows if r["organ"] == o}
            if not rr:
                continue
            ax = next(iter(rr.values())).get("axis", "")
            cells = []
            for l in levels:
                cells.append(f"{rr[l]['delta']:+.4f}" if l in rr else "—")
            print(f"| `{o}` | {ax} | " + " | ".join(cells) + " |")
    if d.get("depth_sweep"):
        print("\n### D1 depth sweep (single layer, structured)\n")
        ls = sorted({r["layer"] for r in d["depth_sweep"]})
        print("| layer | attn 50% | attn 90% | mlp 50% | mlp 90% |")
        print("|---|---|---|---|---|")
        for L in ls:
            g = {(r["block"], r["level"]): r for r in d["depth_sweep"] if r["layer"] == L}
            def c(b, l):
                return f"{g[(b,l)]['delta']:+.4f}" if (b, l) in g else "—"
            print(f"| {L} | {c('attn',0.5)} | {c('attn',0.9)} | {c('mlp',0.5)} | {c('mlp',0.9)} |")


def d3():
    d = load("d3_lowrank.json")
    if not d:
        return
    print("### D3 controls\n")
    print("| control | BPB | delta | fired |")
    print("|---|---|---|---|")
    for c in d["controls"]:
        print(f"| {c['tag']} ({c.get('control','')}) | {c['bpb']:.5f} | {c['delta']:+.5f} | "
              f"{'YES' if c.get('fired') else 'no'} |")
    print(f"\n### D3 low-rank sweep (baseline {d['baseline_bpb']:.5f})\n")
    print("| organ | shape | rank | bytes retained | BPB | delta | paired SE |")
    print("|---|---|---|---|---|---|---|")
    for r in d["sweep"]:
        print(f"| `{r['organ']}` | {tuple(r['shape'])} | {r['rank']} | {r['bytes_frac']*100:.1f}% | "
              f"{r['bpb']:.5f} | {r['delta']:+.4f} | {r['paired_se']:.4f} |")


def d0():
    d = load("d0_layout.json")
    if not d:
        return
    print("### D0 controls\n")
    for c in d["controls"]:
        print(f"- `{c['name']}` — expect {c['expect']}; measured mean run "
              f"{c['measured_mean_run']:.4f}, BS64 skippable {c['block64_skippable']:.4f} → "
              f"**{'FIRED' if c['fired'] else 'DID NOT FIRE'}**")
    print("\n### D0 layout, per ordering\n")
    print("| layer | ordering | mean bytes/read | act bytes in runs ≥4KB | ≥16KB | ≥48KB | "
          "skip@BS8 | BS16 | BS32 | BS64 (48KB) | eff. activation @BS64 | speedup@BS64 |")
    print("|---" * 12 + "|")
    for rec in d["layers"]:
        for name, st in rec["orderings"].items():
            b = st["blocks"]
            print(f"| {rec['layer']} | {name} | {st['mean_bytes_per_read']:.0f} | "
                  f"{st['frac_active_bytes_in_runs_ge_4KB']:.4f} | "
                  f"{st['frac_active_bytes_in_runs_ge_16KB']:.4f} | "
                  f"{st['frac_active_bytes_in_runs_ge_48KB']:.4f} | "
                  f"{b['8']['block_skippable_fraction']:.3f} | "
                  f"{b['16']['block_skippable_fraction']:.3f} | "
                  f"{b['32']['block_skippable_fraction']:.3f} | "
                  f"{b['64']['block_skippable_fraction']:.3f} | "
                  f"{b['64']['effective_activation_fraction']:.3f} | "
                  f"{b['64']['speedup_vs_dense']:.2f}x |")
    print("\n### D0 MoE routing view (relative L2 error of the FFN output, oracle router)\n")
    print("| layer | E | k | activation k/E | expert KB | ≥48KB | coactivation | random control | "
          "delta (random − coact) |")
    print("|---" * 9 + "|")
    for rec in d["layers"]:
        for r in rec.get("routing", []):
            print(f"| {rec['layer']} | {r['E']} | {r['k']} | {r['activation_fraction']:.3f} | "
                  f"{r['expert_bytes_per_organ']/1024:.1f} | {'yes' if r['engine_legal_48KB'] else 'no'} | "
                  f"{r['relerr_coactivation']:.4f} | {r['relerr_random_control']:.4f} | "
                  f"{r['relerr_random_control']-r['relerr_coactivation']:+.4f} |")


def d2b():
    d = load("d2b_actbasis.json")
    if not d:
        return
    print("### D2b controls\n")
    for c in d["controls"]:
        print(f"- `{c['name']}` — {c['expect']} → **{'FIRED' if c['fired'] else 'DID NOT FIRE'}** "
              f"({ {k: (round(v,5) if isinstance(v,float) else v) for k,v in c.items() if k not in ('name','expect','fired')} })")
    print("\n### D2b activation basis (rel L2 error of y = down(h) at kept fraction f)\n")
    ks = d["keep_fracs"]
    print("| layer | basis | conc99 | " + " | ".join(f"f={f}" for f in ks) + " |")
    print("|---" * (len(ks) + 3) + "|")
    for rec in d["layers"]:
        for b in rec["bases"]:
            print(f"| {rec['layer']} | {b['basis']} | {b['concentration99']:.4f} | "
                  + " | ".join(f"{b['relerr'][str(f)]:.4f}" for f in ks) + " |")
    print("\n### D2b activation spectrum\n")
    print("| layer | r for 90% | 99% | 99.9% | dim |")
    print("|---|---|---|---|---|")
    for rec in d["layers"]:
        s = rec.get("activation_spectrum")
        if s:
            print(f"| {rec['layer']} | {s['r_for_90pct']} | {s['r_for_99pct']} | "
                  f"{s['r_for_999pct']} | {s['dim']} |")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["d0", "d1", "d2b", "d3"]):
        print(f"\n<!-- ===== {a} ===== -->\n")
        {"d0": d0, "d1": d1, "d2b": d2b, "d3": d3}[a]()
