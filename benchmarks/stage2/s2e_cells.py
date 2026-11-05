"""Stage 2 test 5, amendment 1: three cells that close the desk budget's uncertainty, in the idle session, after s2e_engines.py.
One build of c50_engine.c with default-off options (bin/c50_engine_s2e.exe), Ling-mini Q4_K_M, context 2048, 128 steps, as test 5:

    base    the build with every option off: the reference of every cell here
    bar1    --barrier-extra 1: every job followed by one empty run()       the cost of a barrier
    bar3    --barrier-extra 3
    kvq8    --kv q8_0: K and V read as Q8_0                                the KV read
    k4, k3  --experts-used 4, 3: the time only, weights not renormalized   the routed read

Order: pre-flight, then 5 rounds visiting the 6 arms once, rotated.  A process at or above the occupancy bar is VOID and replaced,
at most 5 per arm.  CONFIG must confirm the cell.  `report` prints the derived numbers.  It labels nothing, but the run is VOID
unless the cells order as they must: bar3 slower than bar1 slower than base, k3 faster than k4 faster than base.

Usage: python -I benchmarks/stage2/s2e_cells.py run --out benchmarks/stage2/results/s2e_cells.json
       python -I benchmarks/stage2/s2e_cells.py report --run benchmarks/stage2/results/s2e_cells.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE.parents[0] / "donor_adaptation" / "engine"))
import c50_s0_run as S0  # noqa: E402
import e44_interval as E44  # noqa: E402

EXE = HERE.parents[0] / "chatbot50" / "bin" / "c50_engine_s2e.exe"
CTX, STEPS, ROUNDS = 2048, 128, 5
MEM, PRIO = S0.A1B_ARMS["B0"]
ARMS = {"base": ([], "kv=f16 barrier_extra=0 experts_used=8 "), "bar1": (["--barrier-extra", "1"], "kv=f16 barrier_extra=1 experts_used=8 "),
        "bar3": (["--barrier-extra", "3"], "kv=f16 barrier_extra=3 experts_used=8 "),
        "kvq8": (["--kv", "q8_0"], "kv=q8_0 barrier_extra=0 experts_used=8 "), "k4": (["--experts-used", "4"], "kv=f16 barrier_extra=0 experts_used=4 "),
        "k3": (["--experts-used", "3"], "kv=f16 barrier_extra=0 experts_used=3 ")}


def one(arm: str) -> dict:
    extra, want = ARMS[arm]
    split = E44.Split()
    t0 = time.time()
    cmd = [str(EXE), "--model", str(S0.MODEL), "--bench", str(S0.IDS), "--ctx", str(CTX), "--steps", str(STEPS), "--mem", MEM, "--prio", PRIO] + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    text = out.decode(errors="replace")
    config = [x for x in text.splitlines() if x.startswith("CONFIG ")]
    rows = [json.loads(x) for x in text.splitlines() if x.startswith("{")]
    rec = {"arm": arm, "cmd": cmd, "returncode": proc.returncode, "config": config, "stderr_tail": err.decode(errors="replace")[-300:],
           "system_pct": system, "foreign_pct": foreign, "wall_s": time.time() - t0}
    if proc.returncode or len(config) != 1 or len(rows) != 1 or want not in config[0]:
        rec["status"] = "FAILED"
        return rec
    rec.update(rows[0])
    rec["status"] = "OK" if foreign == foreign and foreign < S0.OCC_BAR else "VOID"
    return rec


def run(a: argparse.Namespace) -> int:
    report = {"cell": "S2E-cells", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "exe_sha256": S0.sha256(EXE), "rounds": ROUNDS, "ctx": CTX,
              "processes": [], "occ_bar_pct": S0.OCC_BAR}

    def save():
        a.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("CONFIG", json.dumps({k: report[k] for k in ("exe_sha256", "rounds", "ctx")}), list(ARMS), flush=True)
    names = list(ARMS)
    if not a.dry:
        pre = E44.watch(S0.PREFLIGHT_SECONDS)
        mean = sum(pre) / len(pre) if pre else float("nan")
        report["preflight"] = {"mean": mean, "admissible": bool(pre) and mean < S0.OCC_BAR}
        if not report["preflight"]["admissible"]:
            report["verdict_input"] = "VOID: pre-flight occupancy"
            save()
            return 1
    repl: dict[str, int] = {}
    for r in range(1 if a.dry else ROUNDS):
        for arm in names[r % len(names):] + names[:r % len(names)]:
            while True:
                rec = one(arm)
                rec["round"] = r
                report["processes"].append(rec)
                save()
                print(f"r{r} {arm} {rec['status']} foreign={rec.get('foreign_pct', float('nan')):.2f}% tok_s={rec.get('median_tok_s', float('nan')):.3f}", flush=True)
                if rec["status"] == "FAILED":
                    report["verdict_input"] = f"VOID: {arm} failed"
                    save()
                    return 1
                if rec["status"] == "OK" or a.dry:
                    break
                repl[arm] = repl.get(arm, 0) + 1
                if repl[arm] > S0.MAX_REPLACEMENTS:
                    report["verdict_input"] = f"VOID: more than {S0.MAX_REPLACEMENTS} replacements ({arm})"
                    save()
                    return 1
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "DRY" if a.dry else "COMPLETE"
    save()
    return 0


def report(a: argparse.Namespace) -> int:
    r = json.loads(a.run.read_text(encoding="utf-8"))
    ms = {x: [1e3 * p["median_step_s"] for p in r["processes"] if p["arm"] == x and p["status"] == "OK"] for x in ARMS}
    if not all(ms.values()):
        print(json.dumps({"status": r.get("verdict_input"), "arms_without_ok_process": [x for x, v in ms.items() if not v]}))
        return 1
    med = {x: st.median(v) for x, v in ms.items()}
    jobs = st.median(p["jobs_per_step"] for p in r["processes"] if p["arm"] == "base" and p["status"] == "OK")
    out = {"status": r.get("verdict_input"), "ms_median": med, "ms_min_max": {x: [min(v), max(v)] for x, v in ms.items()}, "jobs_per_step": jobs,
           "us_per_barrier": {"bar1": 1e3 * (med["bar1"] - med["base"]) / jobs, "bar3": 1e3 * (med["bar3"] - med["base"]) / (3 * jobs)},
           "barriers_ms_per_step": (med["bar3"] - med["base"]) / 3, "kv_saved_ms": med["base"] - med["kvq8"],
           "k4_saved_ms": med["base"] - med["k4"], "k3_saved_ms": med["base"] - med["k3"],
           "order_holds": med["bar3"] > med["bar1"] > med["base"] > med["k4"] > med["k3"]}
    print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--dry", action="store_true", help="one process per arm, no pre-flight")
    r.set_defaults(fn=run)
    q = sp.add_parser("report")
    q.add_argument("--run", type=Path, required=True)
    q.set_defaults(fn=report)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
