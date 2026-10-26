"""C50-T2 trained stitch: the verdict, from c50_t2_tstitch.py's record only (protocol C50_T2_TSTITCH_PROTOCOL_20261026).

Per v0 MoE layer l, at its depth-matched F0 layer j: l STITCHES iff the chosen trained arm reads, on the test rows,
at least as well as F0's own state with Gaussian noise of R^2 = 0.5, on both measures:
    test top-k overlap >= noise_0.5 overlap  and  test routed-output cosine >= noise_0.5 cosine.
Cell: PASS if >= 10 of the 19 layers stitch; PARTIAL if 1-9; FAIL if none.  On FAIL, the cell is DATA-LIMITED if the
median over layers of (test cosine, full fit) - (test cosine, half fit) of the linear arm is >= DATA_GAIN.
STOP (no verdict) if a check or a planted control fails, on any layer:
  - tool: the bf16 block on F0's own test states vs llama.cpp: top-k overlap >= 0.97 and median cosine >= 0.99;
  - ridge: the ridge arm reproduces the linear post-hoc: same lambda, |d overlap| <= 0.02, |d cosine| <= 0.02;
  - positive: the PCA-3584 map of F0's own state, trained from its PCA-1024 reconstruction, stitches by the same rule;
  - shuffled: the chosen shuffled arm does not stitch by the same rule.

    python -I benchmarks/chatbot50/c50_t2_tstitch_verdict.py RECORD --json OUT
    python -I benchmarks/chatbot50/c50_t2_tstitch_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

NOISE_KEY = "noise_r2_0.5"
PASS_MIN, N_LAYERS = 10, 19
TOOL_OVERLAP, TOOL_COS = 0.97, 0.99
RIDGE_TOL = 0.02
DATA_GAIN = 0.05


def stitches(m: dict, bars: dict) -> bool:
    return m["topk_overlap"] >= bars[NOISE_KEY]["topk_overlap"] and m["moe_cos_mean"] >= bars[NOISE_KEY]["moe_cos_mean"]


def decide(rec: dict) -> dict:
    layers = rec["layers"]
    checks, planted, per = {"layers": len(layers) == N_LAYERS, "tool": {}, "ridge": {}}, {"positive": {}, "shuffled": {}}, {}
    for l, o in layers.items():
        t, A, bars = o["tool"], o["arms"], o["bars"]
        checks["tool"][l] = t["topk_overlap"] >= TOOL_OVERLAP and t["cos_median"] >= TOOL_COS
        rt, ph = A["ridge"]["test"], o["posthoc"]
        checks["ridge"][l] = (ph["f0_layer"] == o["f0_layer"] and ph["lambda"] == A["ridge"]["lambda"]
                              and abs(rt["topk_overlap"] - ph["topk_overlap"]) <= RIDGE_TOL
                              and abs(rt["moe_cos_mean"] - ph["moe_cos_mean"]) <= RIDGE_TOL)
        planted["positive"][l] = stitches(o["positive"], bars)
        planted["shuffled"][l] = stitches(A[o["chosen_shuffled"]]["test"], bars)
        m = A[o["chosen"]]["test"]
        bar = bars[NOISE_KEY]
        gap = bar["moe_cos_mean"] - rt["moe_cos_mean"]
        per[l] = {"f0_layer": o["f0_layer"], "arm": o["chosen"], "overlap": m["topk_overlap"], "cos": m["moe_cos_mean"],
                  "ridge_overlap": rt["topk_overlap"], "ridge_cos": rt["moe_cos_mean"],
                  "bar_overlap": bar["topk_overlap"], "bar_cos": bar["moe_cos_mean"],
                  "cos_gap_closed": (m["moe_cos_mean"] - rt["moe_cos_mean"]) / gap if gap > 0 else None,
                  "half_minus_full_cos": A["half_linear"]["test"]["moe_cos_mean"] - A["linear"]["test"]["moe_cos_mean"],
                  "fit_minus_test_cos": A[o["chosen"]]["fit"]["moe_cos_mean"] - m["moe_cos_mean"] if "fit" in A[o["chosen"]] else None,
                  "stitches": stitches(m, bars)}
    ok = (checks["layers"] and all(checks["tool"].values()) and all(checks["ridge"].values())
          and all(planted["positive"].values()) and not any(planted["shuffled"].values()))
    n = sum(p["stitches"] for p in per.values())
    data_gain = statistics.median(-p["half_minus_full_cos"] for p in per.values()) if per else 0.0
    if not ok:
        verdict = "STOP: a check or a planted control failed; no verdict"
    elif n >= PASS_MIN:
        verdict = f"PASS: {n} of {N_LAYERS} layers stitch; option X proceeds to its training protocol"
    elif n >= 1:
        verdict = f"PARTIAL: {n} of {N_LAYERS} layers stitch; option X proceeds on those layers only"
    elif data_gain >= DATA_GAIN:
        verdict = f"FAIL, DATA-LIMITED: no layer stitches, and doubling the fit data raised the test cosine by {data_gain:.3f} (median)"
    else:
        verdict = f"FAIL: no layer stitches, and the cell is not data-limited (median gain from doubling the data {data_gain:.3f})"
    return {"checks": checks, "planted": planted, "layers": per, "stitching_layers": n,
            "median_data_gain_cos": data_gain, "verdict": verdict}


def selftest() -> int:
    bars = {NOISE_KEY: {"topk_overlap": 0.7, "moe_cos_mean": 0.8}}
    meas = lambda ov, cs: {"topk_overlap": ov, "moe_cos_mean": cs}

    def lay(ov, cs, half_gap=0.0):
        return {"f0_layer": 5, "tool": {"topk_overlap": 0.99, "cos_median": 0.999}, "bars": bars,
                "positive": meas(0.9, 0.95), "chosen": "linear", "chosen_shuffled": "shuffled_linear",
                "posthoc": {"f0_layer": 5, "lambda": 0.01, "topk_overlap": 0.5, "moe_cos_mean": 0.4},
                "arms": {"ridge": {"lambda": 0.01, "test": meas(0.5, 0.4)},
                         "linear": {"test": meas(ov, cs), "fit": meas(ov, cs + 0.05)},
                         "mlp": {"test": meas(0.4, 0.3)},
                         "shuffled_linear": {"test": meas(0.1, 0.2)},
                         "half_linear": {"test": meas(ov, cs - half_gap)}}}

    mk = lambda good, gap=0.0: {"layers": {str(l): lay(0.75, 0.85) if l < good else lay(0.6, 0.6, gap) for l in range(1, 20)}}
    cases = [("19 stitch", mk(20), "PASS"), ("10 stitch", mk(11), "PASS"), ("9 stitch", mk(10), "PARTIAL"),
             ("none, not data-limited", mk(1, 0.01), "FAIL"), ("none, data-limited", mk(1, 0.08), "FAIL, DATA-LIMITED")]
    r = mk(20)
    r["layers"]["4"]["tool"] = {"topk_overlap": 0.9, "cos_median": 0.999}
    cases.append(("tool fails", r, "STOP"))
    r = mk(20)
    r["layers"]["4"]["arms"]["ridge"]["test"] = meas(0.5, 0.45)
    cases.append(("ridge off the post-hoc", r, "STOP"))
    r = mk(20)
    r["layers"]["4"]["positive"] = meas(0.9, 0.7)
    cases.append(("positive fails", r, "STOP"))
    r = mk(20)
    r["layers"]["4"]["arms"]["shuffled_linear"]["test"] = meas(0.8, 0.9)
    cases.append(("a shuffled map stitches", r, "STOP"))
    r = mk(1)
    for o in r["layers"].values():
        o["arms"]["linear"]["test"] = meas(0.75, 0.7)
        o["arms"]["half_linear"]["test"] = meas(0.75, 0.7)
    cases.append(("overlap ok but cosine low", r, "FAIL"))
    r = mk(20)
    del r["layers"]["19"]
    cases.append(("18 layers", r, "STOP"))
    bad = 0
    for name, rec, want in cases:
        got = decide(rec)["verdict"].split(":")[0]
        print(f"{'ok ' if got == want else 'BAD'} {name}: {got} (want {want})")
        bad += got != want
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    raw = Path(a.record).read_bytes()
    out = {"cell": "C50-T2 trained stitch verdict", "record": a.record, "record_sha256": hashlib.sha256(raw).hexdigest(),
           **decide(json.loads(raw))}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
