"""C50-A2b external reference rate: llama-bench decode on the Ling-mini Q4_K_M artifact.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md.
Same clean-box rules as C50-A1: 45 s pre-flight, five fresh processes, a process whose
foreign occupancy is at or above 4.39% is VOID (kept) and replaced, at most five times.
Records numbers only; there is no gate.

Usage:
    python -I benchmarks/chatbot50/c50_a2b_rate.py --model <q4_k_m.gguf> --out benchmarks/chatbot50/results/c50_a2b_rate_<tag>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "donor_adaptation" / "engine"))
import e44_interval as E44  # noqa: E402

BENCH = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4/build-avx2/bin/llama-bench.exe")
ARGS = ["-t", "6", "-p", "0", "-n", "128", "-d", "512,2048,4096", "-r", "3", "-fa", "off", "-o", "json"]
OCC_BAR = 4.39
PROCESSES = 5
MAX_REPLACEMENTS = 5
PREFLIGHT_SECONDS = 45


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def one_process(model: Path) -> dict:
    split = E44.Split()
    t0 = time.time()
    proc = subprocess.Popen([str(BENCH), "-m", str(model), *ARGS], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    if proc.returncode != 0:
        raise SystemExit(f"llama-bench failed rc={proc.returncode}: {err.decode(errors='replace')[-800:]}")
    rows = json.loads(out.decode())
    for r in rows:   # CONFIG check: the registered arguments must be the ones that ran
        if r["n_threads"] != 6 or r["n_prompt"] != 0 or r["n_gen"] != 128 or r.get("flash_attn") not in (0, False):
            raise SystemExit(f"llama-bench row does not confirm the registered configuration: {r}")
    return {"rows": [{"n_depth": r["n_depth"], "avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"],
                      "samples_ts": r.get("samples_ts"), "build_commit": r.get("build_commit"),
                      "cpu_info": r.get("cpu_info"), "type_k": r.get("type_k")} for r in rows],
            "system_pct": system, "foreign_pct": foreign, "wall_s": time.time() - t0,
            "timing": split.record(), "void": not (foreign == foreign and foreign < OCC_BAR)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = Path(args.model)
    report = {"cell": "C50-A2b-rate", "bench": str(BENCH), "bench_sha256": sha256(BENCH), "args": ARGS,
              "model": str(model), "model_sha256": sha256(model), "occ_bar_pct": OCC_BAR,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pre = E44.watch(PREFLIGHT_SECONDS)
    # Addendum 2 of the A1 protocol: the pre-flight is judged on its mean, the estimand the E52 bar was
    # calibrated on and the one the per-process gate uses; the max is recorded, not gated.
    mean = sum(pre) / len(pre) if pre else float("nan")
    report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre, "rule": "mean < bar (A1 addendum 2)",
                           "mean": mean, "max": max(pre) if pre else None,
                           "samples_at_or_above_bar": sum(x >= OCC_BAR for x in pre),
                           "admissible": bool(pre) and mean < OCC_BAR}
    print(f"preflight mean={mean:.2f}% max={report['preflight']['max']:.2f}% admissible={report['preflight']['admissible']}")
    report["processes"] = []
    if not report["preflight"]["admissible"]:
        report["verdict_input"] = "VOID: pre-flight occupancy at or above the bar"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 1
    admissible = replacements = 0
    while admissible < PROCESSES and replacements <= MAX_REPLACEMENTS:
        rec = one_process(model)
        report["processes"].append(rec)
        print(f"process foreign={rec['foreign_pct']:.2f}% void={rec['void']} wall={rec['wall_s']:.0f}s "
              + " ".join(f"d{r['n_depth']}={r['avg_ts']:.2f}" for r in rec["rows"]))
        replacements += rec["void"]
        admissible += not rec["void"]
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    ok = [p for p in report["processes"] if not p["void"]]
    report["verdict_input"] = "COMPLETE" if len(ok) >= PROCESSES else "VOID: too few admissible processes"
    if ok:
        report["summary"] = {}
        for depth in sorted({r["n_depth"] for p in ok for r in p["rows"]}):
            vals = [r["avg_ts"] for p in ok for r in p["rows"] if r["n_depth"] == depth]
            report["summary"][str(depth)] = {"median_tok_s": statistics.median(vals), "min": min(vals),
                                             "max": max(vals), "max_over_min": max(vals) / min(vals)}
            print(f"depth {depth}: median {statistics.median(vals):.2f} tok/s [{min(vals):.2f}..{max(vals):.2f}]")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict_input"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
