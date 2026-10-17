"""C50-A1b driver: pre-flight, then 10 rounds of fresh engine bench processes, arms interleaved inside a round.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A1B_TIMING_STABILITY_PROTOCOL_20261017.md.
Arms (bin/c50_engine_a1b.exe --bench, P0, ctx 2048, 128 steps):
    B0  --mem map     --prio normal   (the A3 engine path)
    B1  --mem private --prio high     (weights copied into private memory locked in the working set)
    B2  --mem large   --prio high     (large pages; run only if the engine can take SeLockMemoryPrivilege)
Each process is split into own / foreign occupancy (E44); a process at or above the E52 bar is VOID and
replaced (at most 5 replacements per arm).  Numbers only: c50_a1b_verdict.py applies the gate.

Usage:
    python -I benchmarks/chatbot50/c50_a1b_run.py --out benchmarks/chatbot50/results/c50_a1b_run_<tag>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "donor_adaptation" / "engine"))
import e44_interval as E44  # noqa: E402

EXE = HERE / "bin" / "c50_engine_a1b.exe"
# Addendum 1 (17 October, before any A1b process): S0 extends c50_engine.c, so the A1b source is the frozen
# byte copy c50_engine_a1b.c (sha256 f874031b..., the 7ab2048 c50_engine.c) and the binary must be the pinned one.
SOURCE = HERE / "c50_engine_a1b.c"
EXE_SHA256_PIN = "6cbafb9071ed8635612aa17e926293dd4a9494a1e7c1814bdfb0b6e811868b60"
MODEL = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-mini-2.0-a810f64-q4_k_m.gguf")
IDS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs/docs.tsv")
ARMS = {"B0": ["--mem", "map", "--prio", "normal"], "B1": ["--mem", "private", "--prio", "high"],
        "B2": ["--mem", "large", "--prio", "high"]}
OCC_BAR, ROUNDS, MAX_REPLACEMENTS, PREFLIGHT_SECONDS = 4.39, 10, 5, 45


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def one_process(arm: str) -> dict:
    split = E44.Split()
    t0 = time.time()
    cmd = [str(EXE), "--model", str(MODEL), "--bench", str(IDS), "--ctx", "2048", "--steps", "128"] + ARMS[arm]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    text = out.decode(errors="replace")
    config = [ln for ln in text.splitlines() if ln.startswith("CONFIG ")]
    rows = [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]
    rec = {"arm": arm, "cmd": cmd, "returncode": proc.returncode, "config": config, "stderr_tail": err.decode(errors="replace")[-400:],
           "system_pct": system, "foreign_pct": foreign, "wall_s": time.time() - t0, "timing": split.record()}
    want = f"mem={ARMS[arm][1]} prio={ARMS[arm][3]}"
    if proc.returncode != 0 or len(config) != 1 or len(rows) != 1 or want not in config[0] \
            or "ctx=2048 steps=128 threads=6 " not in config[0] or f"model={MODEL.as_posix()} " not in config[0].replace("\\", "/"):
        rec["status"] = "FAILED"
        return rec
    rec.update(rows[0])
    rec["status"] = "VOID" if not (foreign == foreign and foreign < OCC_BAR) else "OK"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if sha256(EXE) != EXE_SHA256_PIN:
        print("VOID: bin/c50_engine_a1b.exe is not the pinned A1b binary", file=sys.stderr)
        return 1
    report = {"cell": "C50-A1b", "exe": str(EXE), "exe_sha256": sha256(EXE), "source": SOURCE.name, "source_sha256": sha256(SOURCE),
              "model_sha256_pin": "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4",
              "ids_sha256": sha256(IDS), "occ_bar_pct": OCC_BAR, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # B2 needs SeLockMemoryPrivilege; the engine exits 4 without it.  Probe once with a tiny allocation-free check:
    probe = subprocess.run([str(EXE), "--model", str(MODEL), "--bench", str(IDS), "--ctx", "1", "--steps", "1"] + ARMS["B2"],
                           capture_output=True, text=True)
    arms = ["B0", "B1"] + (["B2"] if probe.returncode == 0 else [])
    report["b2_probe"] = {"returncode": probe.returncode, "stderr": probe.stderr[-300:], "included": "B2" in arms}
    print("arms", arms, "(B2 unavailable: " + probe.stderr.strip()[-80:] + ")" if "B2" not in arms else "")
    pre = E44.watch(PREFLIGHT_SECONDS)
    mean = sum(pre) / len(pre) if pre else float("nan")
    report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre, "rule": "mean < bar (A1 addendum 2)", "mean": mean,
                           "max": max(pre) if pre else None, "admissible": bool(pre) and mean < OCC_BAR}
    print(f"preflight mean={mean:.2f}% admissible={report['preflight']['admissible']}", flush=True)
    report["processes"] = []
    if not report["preflight"]["admissible"]:
        report["verdict_input"] = "VOID: pre-flight occupancy at or above the bar"
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 1
    replacements = {a: 0 for a in arms}
    for rnd in range(ROUNDS):
        for arm in arms:
            while True:
                rec = one_process(arm)
                rec["round"] = rnd
                report["processes"].append(rec)
                Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
                print(f"round {rnd} {arm} {rec['status']} foreign={rec['foreign_pct']:.2f}% "
                      f"median_tok_s={rec.get('median_tok_s', float('nan')):.3f} wall={rec['wall_s']:.0f}s", flush=True)
                if rec["status"] == "FAILED":
                    report["verdict_input"] = f"VOID: process failed ({arm})"
                    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
                    return 1
                if rec["status"] == "OK":
                    break
                replacements[arm] += 1
                if replacements[arm] > MAX_REPLACEMENTS:
                    report["verdict_input"] = f"VOID: more than {MAX_REPLACEMENTS} replacements ({arm})"
                    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
                    return 1
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "COMPLETE"
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
