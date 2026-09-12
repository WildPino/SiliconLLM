#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E37 addendum B -- is the fitted router load-bearing on BPB, or only on group mass?

Brief: BRIEF_E37_ADDENDUM_B_IS_THE_ROUTER_LOAD_BEARING.md (341b505), pushed before this ran.
THE VERDICT CELL IS ALREADY COMMITTED AND THIS CANNOT CHANGE IT.

G-E37D showed the fitted router recovers 0.5075 of the oracle's top-3 groups against 0.0102
random and captures 0.7132 of its mass against 0.0608.  That is a RANK result about predicting
mass, standing in for a SCORE result about BPB that it does not imply -- the mirror of E14 s3.
This builds the same artifact with the SYNTHETIC router and sweeps the same k.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e37_sparsity_cost as E37                                       # noqa: E402

KS = [64, 32, 16, 3, 1]
DECIDE_K, BAR = 3, 0.05
OUT = os.path.join(E37.RES, "e37_router_control.json")


def main():
    d = "D:/_ktmp/e37"
    syn = os.path.join(d, "e37_carved_syn.bin")
    t0 = time.time()
    print("== E37 addendum B: the SYNTHETIC-router control ==", flush=True)
    if not (os.path.exists(syn) and os.path.exists(syn + ".json")):
        cmd = [sys.executable, os.path.join(HERE, "qwen_export.py"),
               "--model", E37.HF, "--revision", E37.REV, "--rule", E37.RULE, "--head-ternary",
               "--calib-seqs", str(E37.CALIB_SEQS), "--fold", "none", "--out", syn,
               "--quant", "carved",
               "--carve-labels", os.path.abspath(os.path.join(
                   HERE, "..", "density", "results", "d0c_labels", "labels_E256.npz")),
               "--carve-seed", "26", "--carve-k", "256"]
        print("  building the synthetic-router twin ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit("export failed")
        for ln in r.stdout.splitlines():
            if "carved" in ln or ln.startswith("wrote") or "GATE" in ln:
                print("        " + ln.strip(), flush=True)

    fit = json.load(open(E37.OUT, encoding="utf-8"))
    qa = fit["quality_arms"]
    dense = fit["S15_DENSE_NF"]["bpb"]
    out = {"brief": "BRIEF_E37_ADDENDUM_B (341b505)", "bar": BAR, "decide_k": DECIDE_K,
           "dense_bpb": dense, "chance_bpb": E37.CHANCE_BPB, "arms": {}}
    print("  %-5s %12s %12s %10s" % ("k", "fitted", "synthetic", "d(syn-fit)"), flush=True)
    for k in KS:
        r = E37.run_bpb(syn, ["--carve-k", str(k)])
        f = qa["K%d" % k]["bpb"]
        out["arms"]["K%d" % k] = {"synthetic_bpb": r["bpb"], "fitted_bpb": f,
                                  "delta": r["bpb"] - f, "seconds": r["seconds"]}
        print("  %-5d %12.6f %12.6f %+10.6f%s"
              % (k, f, r["bpb"], r["bpb"] - f,
                 "   <- decides" if k == DECIDE_K else ""), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    dk = out["arms"]["K%d" % DECIDE_K]["delta"]
    out["verdict"] = {"delta_at_decide_k": dk, "bar": BAR,
                      "name": "FITTED-MATTERS" if dk > BAR else "ROUTER-IS-NOT-THE-CONSTRAINT"}
    out["nonmonotone_synthetic"] = bool(
        max(out["arms"]["K%d" % k]["synthetic_bpb"] for k in KS)
        > out["arms"]["K%d" % KS[-1]]["synthetic_bpb"] + 1e-9)
    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("", flush=True)
    print("  VERDICT  synthetic - fitted at k=%d = %+.6f BPB (bar %.2f) -> %s"
          % (DECIDE_K, dk, BAR, out["verdict"]["name"]), flush=True)
    print("  synthetic curve non-monotone: %s (prediction 3)" % out["nonmonotone_synthetic"],
          flush=True)
    print("  wrote %s  [%.0fs]" % (OUT, out["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
