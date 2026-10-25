"""C50-T2 stitching cell: the verdict, from c50_t2_stitch.py's record only (protocol C50_T2_STITCH_PROTOCOL_20261026).

Per v0 MoE layer l, at its chosen F0 layer j: l STITCHES iff F0's experts read the mapped state at least as well as
F0's own state with Gaussian noise of R^2 = 0.5, on both measures:
    mapped top-k overlap >= noise_0.5 top-k overlap  and  mapped routed-output cosine >= noise_0.5 cosine.
Cell: PASS if >= 10 of the 19 layers stitch; PARTIAL if 1-9; FAIL if none.
STOP (no verdict) if a check or a planted control fails:
  - tool: on every chosen j, numpy block vs llama.cpp: top-k overlap >= 0.97 and median cosine >= 0.99;
  - F0 -> F0 through the same ridge pipeline stitches by the same rule, on every chosen j;
  - shuffled pairs: no layer's shuffled map stitches by the same rule.

    python -I benchmarks/chatbot50/c50_t2_stitch_verdict.py RECORD --json OUT
    python -I benchmarks/chatbot50/c50_t2_stitch_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

NOISE_KEY = "noise_r2_0.5"
PASS_MIN, N_LAYERS = 10, 19
TOOL_OVERLAP, TOOL_COS = 0.97, 0.99


def stitches(overlap: float, cos: float, cal: dict) -> bool:
    return overlap >= cal[NOISE_KEY]["topk_overlap"] and cos >= cal[NOISE_KEY]["moe_cos_mean"]


def decide(rec: dict) -> dict:
    layers, cal, tool = rec["layers"], rec["calibration"], rec["tool"]
    checks = {"layers": len(layers) == N_LAYERS,
              "tool": {j: t["topk_overlap"] >= TOOL_OVERLAP and t["cos_median"] >= TOOL_COS for j, t in tool.items()}}
    planted = {"self_map_stitches": {j: stitches(c["self"]["topk_overlap"], c["self"]["moe_cos_mean"], c) for j, c in cal.items()},
               "shuffled_stitches": {l: stitches(o["shuffled_topk_overlap"], o["shuffled_moe_cos_mean"], cal[str(o["f0_layer"])])
                                     for l, o in layers.items()}}
    per = {l: {"f0_layer": o["f0_layer"], "test_r2": o["test_r2"], "overlap": o["mapped_topk_overlap"],
               "cos": o["mapped_moe_cos_mean"], "noise_overlap": cal[str(o["f0_layer"])][NOISE_KEY]["topk_overlap"],
               "noise_cos": cal[str(o["f0_layer"])][NOISE_KEY]["moe_cos_mean"],
               "stitches": stitches(o["mapped_topk_overlap"], o["mapped_moe_cos_mean"], cal[str(o["f0_layer"])])}
           for l, o in layers.items()}
    ok = (checks["layers"] and all(checks["tool"].values()) and all(planted["self_map_stitches"].values())
          and not any(planted["shuffled_stitches"].values()))
    n = sum(p["stitches"] for p in per.values())
    if not ok:
        verdict = "STOP: a check or a planted control failed; no verdict"
    elif n >= PASS_MIN:
        verdict = f"PASS: {n} of {N_LAYERS} layers stitch; option X proceeds to its training protocol"
    elif n >= 1:
        verdict = f"PARTIAL: {n} of {N_LAYERS} layers stitch; option X proceeds on those layers only"
    else:
        verdict = "FAIL: no layer stitches; option X is closed at linear stitching"
    return {"checks": checks, "planted": planted, "layers": per, "stitching_layers": n, "verdict": verdict}


def selftest() -> int:
    cal = {"5": {NOISE_KEY: {"topk_overlap": 0.4, "moe_cos_mean": 0.6}, "self": {"topk_overlap": 0.95, "moe_cos_mean": 0.99}}}
    lay = lambda ov, cs: {"f0_layer": 5, "test_r2": 0.7, "mapped_topk_overlap": ov, "mapped_moe_cos_mean": cs,
                          "shuffled_topk_overlap": 0.05, "shuffled_moe_cos_mean": 0.2}
    tool = {"5": {"topk_overlap": 0.99, "cos_median": 0.999}}
    mk = lambda good: {"layers": {str(l): lay(0.5, 0.7) if l < good else lay(0.3, 0.7) for l in range(1, 20)},
                       "calibration": cal, "tool": tool}
    cases = [("19 stitch", mk(20), "PASS"), ("10 stitch", mk(11), "PASS"), ("9 stitch", mk(10), "PARTIAL"),
             ("none", mk(1), "FAIL")]
    bad_tool = {**mk(20), "tool": {"5": {"topk_overlap": 0.9, "cos_median": 0.999}}}
    shuf = mk(20)
    shuf["layers"]["3"] = {**shuf["layers"]["3"], "shuffled_topk_overlap": 0.5, "shuffled_moe_cos_mean": 0.7}
    selfbad = {**mk(20), "calibration": {"5": {**cal["5"], "self": {"topk_overlap": 0.3, "moe_cos_mean": 0.99}}}}
    cos_only = {**mk(1), "layers": {str(l): lay(0.5, 0.5) for l in range(1, 20)}}
    cases += [("tool fails", bad_tool, "STOP"), ("a shuffled map stitches", shuf, "STOP"),
              ("self map fails", selfbad, "STOP"), ("overlap ok but cosine low", cos_only, "FAIL")]
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
    out = {"cell": "C50-T2 stitching verdict", "record": a.record, "record_sha256": hashlib.sha256(raw).hexdigest(),
           **decide(json.loads(raw))}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
