"""Stage 2, the fourth code, step 1: the verdict, by the registered rule.  It reads the records and chooses nothing else.

Per organ and four-code form, G = 1 - mean over layers of (block loss of the form / block loss of tern).  The organ's best form is the one
with the larger G.  G >= 0.25 passes an organ; the bar is two thirds of what theory gives a Gaussian at equal bytes (0.375).
    THE FOURTH CODE PAYS          every organ passes, and the kernel's parity holds
    THE FOURTH CODE PAYS ON SOME  some organs pass, and the kernel's parity holds
    NOT AT THIS ERROR             no organ passes
    THE KERNEL DOES NOT READ IT   an organ passes, and the parity fails
VOID: a plant fails, the rotation changes the block, or the Gaussian twins' weight gain leaves 0.33-0.42.

Usage: python -I benchmarks/stage2/s2q_verdict.py --run F --parity F --json F
       python -I benchmarks/stage2/s2q_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BAR = 0.25
FOUR = ("asym", "bal")
ORGANS = ("attention", "shared", "experts")
TWIN_GAIN = (0.33, 0.42)
PLANT_RECOVERED, PLANT_MARGIN = 0.999, 1.05
IDENTITY = 1e-6


def gain(layers: list[dict], organ: str, form: str, ref: str = "tern") -> float:
    r = [x["organs"][organ][form]["block"] / x["organs"][organ][ref]["block"] for x in layers]
    return 1 - sum(r) / len(r)


def parity_ok(p: dict) -> dict:
    c = p["cases"]
    return {"cases": len(c), "four_within": all(x["four"] <= 1.5 * x["tern"] for x in c), "neg_within": all(x["neg"] <= 1.5 * x["tern"] for x in c),
            "bal_within": all(x["bal"] <= 1.5 * x["tern"] for x in c), "plant_far": all(x["wrong"] >= 10 * x["four"] for x in c)}


def controls(r: dict, layers: list[dict]) -> dict:
    pl = r["plant"]
    plant = all(v["recovered"] >= PLANT_RECOVERED and v[f] <= PLANT_MARGIN * v["noise_rel"] for f, v in pl.items())
    tw = [1 - t["gauss_bal"]["weight"] / t["gauss_tern"]["weight"] for x in layers for t in x["twin"]]
    return {"plant": plant, "identity": all(x["identity"]["block"] <= IDENTITY for x in layers), "twin_gain": [min(tw), max(tw)],
            "twin": all(TWIN_GAIN[0] <= g <= TWIN_GAIN[1] for g in tw)}


def decide(r: dict, p: dict) -> dict:
    layers = [r["mini"][k] for k in sorted(r["mini"], key=int)]
    c = controls(r, layers)
    par = parity_ok(p)
    g = {o: {f: gain(layers, o, f) for f in FOUR} for o in ORGANS}
    best = {o: max(FOUR, key=lambda f: g[o][f]) for o in ORGANS}
    passed = [o for o in ORGANS if g[o][best[o]] >= BAR]
    kernel = all(par[k] for k in par if k != "cases")
    out = {"controls": c, "parity": par, "gain": g, "best": best, "passed": passed,
           "rotation_gain": {f: gain(layers, "attention", "rot_" + f, f) for f in ("tern", *FOUR)},
           "twin_real_vs_gauss_output": [{"tensor": t["tensor"], "real": 1 - t["real_bal"]["output"] / t["real_tern"]["output"],
                                          "real_asym": 1 - t["real_asym"]["output"] / t["real_tern"]["output"],
                                          "gauss": 1 - t["gauss_bal"]["output"] / t["gauss_tern"]["output"]} for x in layers for t in x["twin"]],
           "flash_weight_gain": {k: {o: {f: 1 - v[o][f] / v[o]["tern"] for f in FOUR} for o in ORGANS} for k, v in r.get("flash", {}).items()}}
    if not (c["plant"] and c["identity"] and c["twin"]):
        out["outcome"] = "VOID"
    elif not passed:
        out["outcome"] = "NOT AT THIS ERROR"
    elif not kernel:
        out["outcome"] = "THE KERNEL DOES NOT READ IT"
    else:
        out["outcome"] = "THE FOURTH CODE PAYS" if len(passed) == len(ORGANS) else "THE FOURTH CODE PAYS ON SOME"
    return out


def fake(att=0.7, sh=0.7, ex=0.7, recovered=1.0, ident=0.0, twin=0.2 / 0.32):
    def layer():
        o = {n: {"tern": {"block": 1.0}, "asym": {"block": 0.9}, "bal": {"block": v}} for n, v in (("attention", att), ("shared", sh), ("experts", ex))}
        o["attention"].update({"rot_tern": {"block": 1.0}, "rot_asym": {"block": 0.9}, "rot_bal": {"block": att}})
        t = {"tensor": "t", **{f"{a}_{f}": {"weight": 0.32 if f == "tern" else 0.32 * twin, "output": 0.3} for a in ("real", "gauss") for f in ("tern", "asym", "bal")}}
        return {"organs": o, "identity": {"block": ident}, "twin": [t]}
    pl = {f: {"noise_rel": 0.01, "recovered": recovered, f: 0.01} for f in ("tern", "asym", "bal")}
    return {"plant": pl, "mini": {"5": layer(), "10": layer()}}


def selftest() -> int:
    good = {"cases": [{"tern": 0.01, "four": 0.01, "neg": 0.01, "bal": 0.01, "wrong": 0.5}]}
    bad = {"cases": [{"tern": 0.01, "four": 0.3, "neg": 0.3, "bal": 0.3, "wrong": 0.5}]}
    cases = [("all pass", fake(), good, "THE FOURTH CODE PAYS"), ("one organ", fake(att=1.3, sh=1.0), good, "THE FOURTH CODE PAYS ON SOME"),
             ("none", fake(0.95, 0.95, 0.95), good, "NOT AT THIS ERROR"), ("kernel", fake(), bad, "THE KERNEL DOES NOT READ IT"),
             ("plant", fake(recovered=0.9), good, "VOID"), ("identity", fake(ident=1e-3), good, "VOID"), ("twin", fake(twin=0.9), good, "VOID")]
    ok = True
    for name, r, p, want in cases:
        got = decide(r, p)["outcome"]
        print("ok " if got == want else "BAD", name, got)
        ok &= got == want
    print("selftest", "PASS" if ok else "FAIL", len(cases))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    for x in ("run", "parity", "json"):
        ap.add_argument(f"--{x}", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = decide(json.loads(a.run.read_text(encoding="utf-8")), json.loads(a.parity.read_text(encoding="utf-8")))
    print(json.dumps(out, indent=1))
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
