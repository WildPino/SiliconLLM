"""C50-A1 driver: pre-flight, five fresh interleaved processes, foreign-occupancy split.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md
(and its pre-run addendum).  This runner records numbers only; c50_a1_verdict.py
applies the registered gates to the file it writes.

Usage:
    python -I benchmarks/chatbot50/c50_a1_run.py --out benchmarks/chatbot50/results/c50_a1_run_<tag>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "donor_adaptation" / "engine"))
import e44_interval as E44  # noqa: E402

OCC_BAR = 4.39          # E52, foreign occupancy in percent; a process at or above it is VOID
PROCESSES = 5
MAX_REPLACEMENTS = 5
PREFLIGHT_SECONDS = 45
N_CELLS = 14
ORDER_STEP = 3          # process i starts at cell (3 * i) mod 14


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_process(exe: Path, order: int, seed: int) -> dict:
    split = E44.Split()
    t0 = time.time()
    proc = subprocess.Popen([str(exe), "--run", "--order", str(order), "--seed", str(seed)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    text = out.decode(errors="replace")
    if proc.returncode != 0 or "DONE" not in text:
        raise SystemExit(f"process failed rc={proc.returncode}: {err.decode(errors='replace')[-800:]}")
    config = [ln for ln in text.splitlines() if ln.startswith("CONFIG ")]
    cells = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    if len(config) != 1 or f"order={order}" not in config[0] or "quick=0" not in config[0]:
        raise SystemExit(f"CONFIG line does not confirm the registered run: {config}")
    return {"order": order, "seed": seed, "config": config[0], "cells": cells,
            "system_pct": system, "foreign_pct": foreign, "wall_s": time.time() - t0,
            "timing": split.record(), "void": not (foreign == foreign and foreign < OCC_BAR)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(HERE / "bin" / "c50_a1_roofline.exe"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exe = Path(args.exe)
    report = {"cell": "C50-A1", "exe": str(exe), "exe_sha256": sha256(exe),
              "source_sha256": {p: sha256(HERE / p) for p in
                                ("c50_a1_roofline.c", "c50_kquants.h", "c50_iq_tables.h")},
              "occ_bar_pct": OCC_BAR, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    st = subprocess.run([str(exe), "--selftest"], capture_output=True, text=True)
    report["selftest"] = st.stdout
    print(st.stdout.strip().splitlines()[-1])
    if st.returncode != 0:
        report["verdict_input"] = "VOID: selftest failed"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 1
    pre = E44.watch(PREFLIGHT_SECONDS)
    report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre,
                           "max": max(pre) if pre else None,
                           "admissible": bool(pre) and max(pre) < OCC_BAR}
    print(f"preflight max={report['preflight']['max']:.2f}% admissible={report['preflight']['admissible']}")
    report["processes"] = []
    if not report["preflight"]["admissible"]:
        report["verdict_input"] = "VOID: pre-flight occupancy at or above the bar"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 1
    admissible, replacements, slot = 0, 0, 0
    while admissible < PROCESSES and replacements <= MAX_REPLACEMENTS:
        order = (ORDER_STEP * slot) % N_CELLS
        rec = one_process(exe, order, seed=slot + 1)
        report["processes"].append(rec)
        print(f"process slot={slot} order={order} foreign={rec['foreign_pct']:.2f}% "
              f"void={rec['void']} wall={rec['wall_s']:.1f}s")
        if rec["void"]:
            replacements += 1
        else:
            admissible += 1
            slot += 1
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["admissible_processes"] = admissible
    report["replacements"] = replacements
    report["verdict_input"] = "COMPLETE" if admissible >= PROCESSES else "VOID: too few admissible processes"
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict_input"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
