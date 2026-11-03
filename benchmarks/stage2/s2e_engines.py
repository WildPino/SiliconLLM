"""Stage 2 part 1, test 5: c50_engine.c, ik_llama.cpp and llama.cpp on the same Ling-mini Q4_K_M file in one session,
decode at context 2048.  It confirms or refutes the engine's ~1.45x over ik_llama.cpp, read across sessions (S0, 19
October: 35.94 tok/s; wall map, 29 October: 24.71).  A timing: idle box, the user's go.

Reused as they were run, so each number is comparable with its session's:
    llama.cpp and ik_llama.cpp: the wall map's binaries, cells, llama-bench arguments, page-in and correctness check
        (benchmarks/wallmap/w0_ref_speed.py; Ling-mini only);
    c50_engine: S0's pinned binary and process (benchmarks/chatbot50/c50_s0_run.py), A1b's adopted arm B0 (map, normal
        priority), context 2048, 128 steps; and S0's planted binary (the head read twice), the trivial control.
Order:
    pins (binaries, S0's source blob, the model's sha256), the wall map's perplexity check on Ling-mini (every
    llama.cpp / ik cell within 6% of llama.cpp fa_off, the planted 4-expert model beyond it)
    pre-flight: 45 s, mean foreign occupancy < 4.39%
    planted: 2 pairs (engine, engine with the head twice), fresh processes, context 2048
    5 rounds; each round visits the 7 arms (the engine and the wall map's 6 cells) once, rotated by the round
A process at or above the bar is VOID and replaced (at most 5 per arm).  Numbers only: s2e_verdict.py.

Usage: python -I benchmarks/stage2/s2e_engines.py --out benchmarks/stage2/results/s2e_run.json [--dry]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "wallmap"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE.parents[0] / "donor_adaptation" / "engine"))
import c50_s0_run as S0  # noqa: E402
import e44_interval as E44  # noqa: E402
import w0_ref_speed as W0  # noqa: E402

CTX, ROUNDS, PLANT_PAIRS = 2048, 5, 2
MEM, PRIO = S0.A1B_ARMS["B0"]
ARMS = [("engine", "s0")] + [(e, n) for e, n, *_ in W0.CELLS]


def visit(engine: str, cell: str) -> dict:
    if engine == "engine":
        rec = S0.one_process(cell, CTX, False, MEM, PRIO)
        if rec["status"] != "FAILED":
            rec["tok_s"] = rec["median_tok_s"]
    else:
        _, _, flags, want, _ = next(c for c in W0.CELLS if c[:2] == (engine, cell))
        rec = W0.one_process(engine, flags, "mini", want)
        if rec["status"] != "FAILED":
            rec["tok_s"] = next(r["avg_ts"] for r in rec["rows"] if r["depth"] == CTX)
    rec["engine"], rec["cell"] = engine, cell
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="pins, the check and one process per arm, no pre-flight")
    a = ap.parse_args()
    out_path = Path(a.out)
    report = {"cell": "S2E-engines", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ctx": CTX, "rounds": ROUNDS,
              "mem": MEM, "prio": PRIO, "arms": ARMS, "occ_bar_pct": W0.OCC_BAR, "dry": a.dry,
              "exe_sha256": {**{f"{k}/{t}": W0.sha256(v[t]) for k, v in W0.ENGINES.items() for t in ("bench", "ppl")},
                             **{f"engine/{k}": S0.sha256(p) for k, p in S0.EXE.items()}},
              "source_sha256": S0.source_blob_sha256(), "model": W0.FILES["mini"][0].as_posix(), "processes": []}

    def save():
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    def stop(why: str) -> int:
        report["verdict_input"] = why
        save()
        print(why)
        return 1

    print("CONFIG", json.dumps({k: report[k] for k in ("ctx", "rounds", "mem", "prio", "arms", "model", "dry")}), flush=True)
    if {k[7:]: v for k, v in report["exe_sha256"].items() if k.startswith("engine/")} != S0.EXE_SHA256_PIN \
            or report["source_sha256"] != S0.SOURCE_SHA256_PIN:
        return stop("VOID: the engine's binaries or source differ from S0's pins")
    if W0.FILES["mini"][0] != S0.MODEL or W0.FILES["mini"][1] != S0.MODEL_SHA256:
        return stop("VOID: the wall map's and S0's Ling-mini files differ")
    report["model_sha256"] = W0.sha256(S0.MODEL)
    if report["model_sha256"] != S0.MODEL_SHA256:
        return stop("VOID: the model differs from its pin")
    if not W0.check(["mini"], report, save):
        return stop("VOID: the correctness check failed")
    if not a.dry:
        pre = E44.watch(W0.PREFLIGHT_SECONDS)
        mean = sum(pre) / len(pre) if pre else float("nan")
        report["preflight"] = {"seconds": W0.PREFLIGHT_SECONDS, "samples": pre, "mean": mean,
                               "admissible": bool(pre) and mean < W0.OCC_BAR}
        print(f"preflight mean={mean:.2f}% admissible={report['preflight']['admissible']}", flush=True)
        if not report["preflight"]["admissible"]:
            return stop("VOID: pre-flight occupancy at or above the bar")
    sched = [("plant", "engine", k, i) for i in range(1 if a.dry else PLANT_PAIRS) for k in ("s0", "plant")]
    for r in range(1 if a.dry else ROUNDS):
        sched += [("rate", e, c, r) for e, c in ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]]
    replacements: dict[str, int] = {}
    for phase, e, c, r in sched:
        key = f"{phase}/{e}/{c}"
        while True:
            rec = visit(e, c)
            rec["phase"], rec["round"] = phase, r
            report["processes"].append(rec)
            save()
            print(f"{phase} r{r} {e}/{c} {rec['status']} foreign={rec['foreign_pct']:.2f}% "
                  f"tok_s={rec.get('tok_s', float('nan')):.3f} wall={rec['wall_s']:.0f}s", flush=True)
            if rec["status"] == "FAILED":
                return stop(f"VOID: process failed or did not confirm its cell ({key})")
            if rec["status"] == "OK" or a.dry:
                break
            replacements[key] = replacements.get(key, 0) + 1
            if replacements[key] > W0.MAX_REPLACEMENTS:
                return stop(f"VOID: more than {W0.MAX_REPLACEMENTS} replacements ({key})")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "DRY" if a.dry else "COMPLETE"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
