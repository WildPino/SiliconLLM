"""Stage 2 part 1, test 5 verdict: the engines' decode rates at context 2048 in one session (it chooses nothing beyond
the registered rule).

Per arm, the OK rate processes' tok/s: mean, min..max, n.  The best cell of ik_llama.cpp and of llama.cpp is the one
with the highest mean in this run (the choice favours them, not the engine).  The ratio engine / best cell gets a
CI95 from 10,000 draws that resample the processes of each arm (seed 20261103).
Control, before any label:
    C1 planted: in each pair, the engine with the head read twice over the engine; the median of the pairs' ratios
                <= 0.90 (S0 measured the head at 7.07 ms of 27.8: twice the head predicts 0.80).
A failed C1, or a run not COMPLETE, gives VOID.
Label, on engine / ik's best cell, against the cross-session 1.45 (35.94 / 24.71) with a band of +-0.07 (about 5%,
A1b's stability bar):
    "CONFIRMS ~1.45x" if the ratio is in [1.38, 1.52];
    "SMALLER" if the CI95's upper end is under 1.38; "LARGER" if its lower end is over 1.52; "UNRESOLVED" otherwise.
--selftest runs planted inputs through the rule.

Usage: python -I benchmarks/stage2/s2e_verdict.py --run benchmarks/stage2/results/s2e_run.json \
           --json benchmarks/stage2/results/s2e_verdict.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

REF, BAND, PLANT_MAX, DRAWS, SEED = 1.45, 0.07, 0.90, 10_000, 20261103


def ratio_ci(num: list[float], den: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    rs = sorted(statistics.fmean(rng.choices(num, k=len(num))) / statistics.fmean(rng.choices(den, k=len(den)))
                for _ in range(DRAWS))
    return statistics.fmean(num) / statistics.fmean(den), rs[int(0.025 * DRAWS)], rs[int(0.975 * DRAWS) - 1]


def decide(rates: dict[str, list[float]], pairs: list[tuple[float, float]], complete: bool) -> dict:
    plant = statistics.median(p / b for b, p in pairs) if pairs else float("nan")
    out = {"controls": {"complete": complete, "C1_plant_ratio": plant, "C1": plant <= PLANT_MAX}}
    if not (complete and out["controls"]["C1"]):
        out["outcome"] = "VOID"
        return out
    arms = {k: {"mean": statistics.fmean(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in rates.items()}
    best = {fam: max((k for k in rates if k.startswith(fam + "/")), key=lambda k: arms[k]["mean"])
            for fam in ("ik", "llama")}
    eng = next(k for k in rates if k.startswith("engine/"))
    r_ik = ratio_ci(rates[eng], rates[best["ik"]])
    r_ll = ratio_ci(rates[eng], rates[best["llama"]])
    if REF - BAND <= r_ik[0] <= REF + BAND:
        label = "CONFIRMS ~1.45x"
    elif r_ik[2] < REF - BAND:
        label = "SMALLER"
    elif r_ik[1] > REF + BAND:
        label = "LARGER"
    else:
        label = "UNRESOLVED"
    out.update(arms=arms, best=best, engine_over_ik=r_ik, engine_over_llama=r_ll, outcome=label)
    return out


def selftest() -> int:
    def rates(eng, ik):
        return {"engine/s0": [eng * x for x in (0.99, 1.0, 1.01, 1.0, 0.995)],
                "ik/fa0_moe": [ik * x for x in (0.99, 1.0, 1.01, 1.0, 1.005)], "ik/fa0": [ik * 0.95] * 5,
                "llama/fa_off": [17.5] * 5, "llama/fa_on": [16.9] * 5}

    good = [(36.0, 28.7), (35.8, 28.9)]
    cases = {
        "as before": ((rates(36.0, 24.8), good, True), "CONFIRMS ~1.45x"),
        "smaller": ((rates(30.0, 25.0), good, True), "SMALLER"),
        "larger": ((rates(42.0, 25.0), good, True), "LARGER"),
        "plant not seen": ((rates(36.0, 24.8), [(36.0, 35.5), (35.8, 35.9)], True), "VOID"),
        "not complete": ((rates(36.0, 24.8), good, False), "VOID"),
    }
    bad = 0
    for name, (args, want) in cases.items():
        got = decide(*args)["outcome"]
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
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    ok = [p for p in run["processes"] if p["status"] == "OK"]
    rates: dict[str, list[float]] = {}
    for p in ok:
        if p["phase"] == "rate":
            rates.setdefault(f"{p['engine']}/{p['cell']}", []).append(p["tok_s"])
    plant = [p for p in ok if p["phase"] == "plant"]
    pairs = [(next(p["tok_s"] for p in plant if p["round"] == i and p["cell"] == "s0"),
              next(p["tok_s"] for p in plant if p["round"] == i and p["cell"] == "plant"))
             for i in sorted({p["round"] for p in plant})]
    out = decide(rates, pairs, run.get("verdict_input") == "COMPLETE" and not run.get("dry"))
    out["rates"] = rates
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.get("arms", {}).items():
        print(f"{k:16s} {v['mean']:.2f} [{v['min']:.2f}..{v['max']:.2f}] n={v['n']}")
    print("controls", out["controls"])
    print("engine/ik", out.get("engine_over_ik"), "engine/llama", out.get("engine_over_llama"))
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
