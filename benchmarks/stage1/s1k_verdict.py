"""Stage 1 (c) verdict: the ternary kernel rate at Ling-flash's expert shapes on this CPU (it chooses nothing beyond
the registered rule).

Per launch key (build | type | cell | stream or resident), the rate is the median over rounds of each launch's median
G weights/s; the spread is the min and max over rounds.  Only COMPLETE launches count; a key needs 2 of 3 rounds.
Controls, before any label:
    C1 wall:        llama.cpp q8_0 experts, streaming, reads at 29-45 GB/s (stage 0's A1 measured 35-41 at Ling-mini's
                    shapes);
    C2 streaming:   llama.cpp tq2_0 experts runs at least 1.15x faster resident than streaming, so the streaming cell
                    reads from RAM;
    C3 correctness: a format enters the verdict only if every round of its expert cells has rel_err < 0.02 and the wrong
                    rows' rel_err > 0.5 (iq2_bn_r4 cannot be checked row by row: reported, not judged).
A failed C1 or C2 makes the run VOID.
A ternary format's expert rate is the best of its streaming "experts" and "experts2d" cells.  With the best checked
ternary format's rate R:
    "PASS"     R >= 128 G weights/s;
    "FAIL"     R < 73 (the top of Q4_K's rate in stage 0);
    "BETWEEN"  otherwise.
Reported, not judged: the attention shape (4096 x 4096) with the same lines, every format's GB/s, resident rates.
--selftest runs planted reports through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage1/s1k_verdict.py --run benchmarks/stage1/results/s1k_run.json \
        --json benchmarks/stage1/results/s1k_verdict.json
    python -I benchmarks/stage1/s1k_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PASS_AT, FAIL_UNDER = 128.0, 73.0
WALL_GBS = (29.0, 45.0)
STREAM_OVER_RESIDENT = 1.15
ERR_MAX, WRONG_MIN = 0.02, 0.5
TERNARY = ("tq2_0", "tq1_0", "iq2_bn", "iq2_bn_r4", "iq1_bn", "i2_s", "tl2")
UNCHECKABLE = ("iq2_bn_r4",)


def summarize(report: dict) -> dict:
    by: dict[str, list[dict]] = {}
    for rec in report["launches"]:
        if rec.get("status") == "COMPLETE":
            by.setdefault(rec["key"], []).append(rec["result"])
    out = {}
    for k, rs in by.items():
        g = [r["median_gweights_per_s"] for r in rs]
        out[k] = {"rounds": len(rs), "gweights_per_s": statistics.median(g), "min": min(g), "max": max(g),
                  "gbytes_per_s": statistics.median([r["median_gbytes_per_s"] for r in rs]),
                  "bytes_per_weight": rs[0]["bytes_per_weight"],
                  "checked": all(r["rel_err"] >= 0 and r["rel_err"] < ERR_MAX and r["rel_err_wrong"] > WRONG_MIN
                                 for r in rs)}
    return out


def label(r: float) -> str:
    return "PASS" if r >= PASS_AT else "FAIL" if r < FAIL_UNDER else "BETWEEN"


def decide(s: dict) -> dict:
    def get(k: str) -> dict | None:
        v = s.get(k)
        return v if v and v["rounds"] >= 2 else None

    wall = get("llama.cpp|q8_0|experts|stream")
    st, rs = get("llama.cpp|tq2_0|experts|stream"), get("llama.cpp|tq2_0|experts|resident")
    c1 = wall is not None and WALL_GBS[0] <= wall["gbytes_per_s"] <= WALL_GBS[1]
    c2 = st is not None and rs is not None and rs["gweights_per_s"] >= STREAM_OVER_RESIDENT * st["gweights_per_s"]
    out = {"controls": {"C1_wall_gbs": wall and wall["gbytes_per_s"], "C1": c1,
                        "C2_resident_over_stream": st and rs and rs["gweights_per_s"] / st["gweights_per_s"], "C2": c2}}
    formats = {}
    for k, v in s.items():
        b, t, cell, mode = k.split("|")
        if t not in TERNARY or mode != "stream" or v["rounds"] < 2:
            continue
        shape = "attn" if cell == "attn" else "experts"
        f = formats.setdefault(f"{b}|{t}", {})
        cur = f.get(shape)
        if cur is None or v["gweights_per_s"] > cur["gweights_per_s"]:
            f[shape] = {**v, "cell": cell}
    out["formats"] = formats
    judged = {n: f["experts"] for n, f in formats.items()
              if "experts" in f and f["experts"]["checked"] and n.split("|")[1] not in UNCHECKABLE}
    if not (c1 and c2):
        out["outcome"] = "VOID"
        return out
    if not judged:
        out["outcome"] = "VOID"
        out["why"] = "no checked ternary format"
        return out
    best = max(judged, key=lambda n: judged[n]["gweights_per_s"])
    out["best"] = {"format": best, **judged[best]}
    out["outcome"] = label(judged[best]["gweights_per_s"])
    att = {n: f["attn"] for n, f in formats.items() if "attn" in f and f["attn"]["checked"]}
    if att:
        ab = max(att, key=lambda n: att[n]["gweights_per_s"])
        out["attn_best"] = {"format": ab, **att[ab], "label": label(att[ab]["gweights_per_s"])}
    return out


def selftest() -> int:
    def rec(k, g, gbs, err=0.006, wrong=1.3, bpw=0.2578):
        return {"key": k, "status": "COMPLETE", "result": {"median_gweights_per_s": g, "median_gbytes_per_s": gbs,
                                                           "bytes_per_weight": bpw, "rel_err": err,
                                                           "rel_err_wrong": wrong}}

    def report(tq2, iq2, tl2, wall_gbs=35.0, resident=140.0, iq2_err=0.008, r4=150.0):
        L = []
        for _ in range(3):
            L += [rec("llama.cpp|q8_0|experts|stream", 33.0, wall_gbs, bpw=1.0625),
                  rec("llama.cpp|tq2_0|experts|stream", tq2, tq2 * 0.2578),
                  rec("llama.cpp|tq2_0|experts|resident", resident, resident * 0.2578),
                  rec("ik_llama.cpp|iq2_bn|experts|stream", iq2, iq2 * 0.252, err=iq2_err),
                  rec("ik_llama.cpp|iq2_bn_r4|experts|stream", r4, r4 * 0.252, err=-1, wrong=-1),
                  rec("bitnet.cpp/tl2|tl2|experts2d|stream", tl2, tl2 * 0.21),
                  rec("llama.cpp|tq2_0|attn|stream", 120.0, 31.0)]
        return {"launches": L}

    cases = {  # name: (report, outcome)
        "between": (report(100, 115, 50), "BETWEEN"),
        "pass via iq2_bn": (report(100, 130, 50), "PASS"),
        "fail": (report(60, 70, 40, resident=80), "FAIL"),
        "unchecked r4 does not pass": (report(100, 115, 50, r4=200), "BETWEEN"),
        "wrong output does not pass": (report(100, 130, 50, iq2_err=0.5), "BETWEEN"),
        "wall control fails": (report(100, 130, 50, wall_gbs=20.0), "VOID"),
        "not streaming": (report(100, 115, 50, resident=105), "VOID"),
    }
    bad = 0
    for name, (rep, want) in cases.items():
        got = decide(summarize(rep))["outcome"]
        bad += got != want
        print("selftest", "ok  " if got == want else "FAIL", name, got)
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    report = json.loads(Path(a.run).read_text(encoding="utf-8"))
    if report.get("dry"):
        raise SystemExit("a dry run is not judged")
    s = summarize(report)
    out = decide(s)
    out["summary"] = s
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in sorted(s):
        v = s[k]
        print(f"{k:45s} {v['gweights_per_s']:7.2f} [{v['min']:6.2f},{v['max']:6.2f}] Gw/s {v['gbytes_per_s']:6.2f} GB/s "
              f"{v['bytes_per_weight']:.4f} B/w rounds {v['rounds']} checked {v['checked']}")
    print("controls", out["controls"])
    if "best" in out:
        print("best", out["best"]["format"], round(out["best"]["gweights_per_s"], 2))
    if "attn_best" in out:
        print("attention", out["attn_best"]["format"], round(out["attn_best"]["gweights_per_s"], 2),
              out["attn_best"]["label"])
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
