"""Stage 2, second idle session: ik_llama.cpp's attention at flash's shape, and the fourth code's correction.  Timing cells, reported, no label.

Arms: ik flash attention on f16 and q8_0 at context 2048, ik without flash attention on f16, f16 at 512 as the control; `corr`, the engine's single-thread bound.
3 rounds, rotated, occupancy and replacements as s2k_cells.py.  VOID if an arm's check fails (rel_err >= 0.05, wrong rows <= 0.5), or context 512 takes 0.6 of 2048 or more.

Usage: python -I benchmarks/stage2/s2l_cells.py run --out benchmarks/stage2/results/s2l_cells.json [--dry]
       python -I benchmarks/stage2/s2l_cells.py report --run benchmarks/stage2/results/s2l_cells.json
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

ATTN, ENGINE = HERE / "bin" / "s2l_attn.exe", HERE / "bin" / "s2k_engine.exe"
ROUNDS = 3
NULL_MASK_MS = 33.56  # the generic path, first run: ik's kernel must beat it
ARMS = {"fa_f16": ("fa", "f16", 2048), "fa_q8": ("fa", "q8_0", 2048), "mm_f16": ("mm", "f16", 2048), "fa_f16_512": ("fa", "f16", 512), "corr": None}


def one(arm: str, dry: bool) -> dict:
    if ARMS[arm]:
        path, kv, ctx = ARMS[arm]
        cmd = [str(ATTN), "--path", path, "--kv", kv, "--ctx", str(ctx), "--threads", "6"] + (["--tokens", "2", "--reps", "1"] if dry else [])
        want = f'"path": "{path}", "kv": "{kv}", "ctx": {ctx}'
    else:
        cmd, want = [str(ENGINE), "--cell", "corr", "--steps", "256"], '"cell": "corr"'
    split = E44.Split()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    system, foreign = split.close(E44._process_times(int(proc._handle)))
    lines = out.decode(errors="replace").splitlines()
    config = [x for x in lines if x.startswith("CONFIG ")]
    rows = [json.loads(x[7:]) for x in lines if x.startswith("RESULT ")]
    rec = {"arm": arm, "cmd": cmd, "returncode": proc.returncode, "config": config, "stderr_tail": err.decode(errors="replace")[-300:],
           "system_pct": system, "foreign_pct": foreign, "wall_s": time.time() - t0}
    if proc.returncode or len(config) != 1 or len(rows) != 1 or want not in config[0]:
        rec["status"] = "FAILED"
        return rec
    rec.update(rows[0])
    rec["status"] = "OK" if foreign == foreign and foreign < S0.OCC_BAR else "VOID"
    return rec


def run(a: argparse.Namespace) -> int:
    rep = {"cell": "S2L-cells", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "attn_sha256": S0.sha256(ATTN), "engine_sha256": S0.sha256(ENGINE),
           "rounds": ROUNDS, "processes": [], "occ_bar_pct": S0.OCC_BAR}

    def save():
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("CONFIG", json.dumps({k: rep[k] for k in ("attn_sha256", "engine_sha256", "rounds")}), list(ARMS), flush=True)
    if not a.dry:
        pre = E44.watch(S0.PREFLIGHT_SECONDS)
        mean = sum(pre) / len(pre) if pre else float("nan")
        rep["preflight"] = {"mean": mean, "admissible": bool(pre) and mean < S0.OCC_BAR}
        if not rep["preflight"]["admissible"]:
            rep["verdict_input"] = "VOID: pre-flight occupancy"
            save()
            return 1
    names, repl = list(ARMS), {}
    for r in range(1 if a.dry else ROUNDS):
        for arm in names[r % len(names):] + names[:r % len(names)]:
            while True:
                rec = one(arm, a.dry)
                rec["round"] = r
                rep["processes"].append(rec)
                save()
                print(f"r{r} {arm} {rec['status']} foreign={rec.get('foreign_pct', float('nan')):.2f}% {rec.get('ms_median')} wall={rec['wall_s']:.0f}", flush=True)
                if rec["status"] == "FAILED":
                    rep["verdict_input"] = f"VOID: {arm} failed"
                    save()
                    return 1
                if rec["status"] == "OK" or a.dry:
                    break
                repl[arm] = repl.get(arm, 0) + 1
                if repl[arm] > S0.MAX_REPLACEMENTS:
                    rep["verdict_input"] = f"VOID: more than {S0.MAX_REPLACEMENTS} replacements ({arm})"
                    save()
                    return 1
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rep["verdict_input"] = "DRY" if a.dry else "COMPLETE"
    save()
    return 0


def report(a: argparse.Namespace) -> int:
    r = json.loads(a.run.read_text(encoding="utf-8"))
    by = {x: [p for p in r["processes"] if p["status"] == "OK" and p["arm"] == x] for x in ARMS}
    if not all(by.values()):
        print(json.dumps({"status": r.get("verdict_input"), "arms_without_ok_process": [x for x, v in by.items() if not v]}))
        return 1
    ms = {x: st.median(p["ms_median"] for p in v) for x, v in by.items()}
    checks = all(p["rel_err"] < 0.05 and p["rel_err_wrong"] > 0.5 for x in ARMS if ARMS[x] for p in by[x])
    moves = ms["fa_f16_512"] < 0.6 * ms["fa_f16"] and ms["fa_f16"] < NULL_MASK_MS
    out = {"status": r.get("verdict_input") if checks and moves else "VOID: a control failed", "ms": ms, "min_max": {x: [min(p["ms_median"] for p in v), max(p["ms_median"] for p in v)] for x, v in by.items()},
           "checks": checks, "moves_with_context": moves, "line_ms": 5.15, "engine_ms": {"f16": 8.96, "q8_0": 11.67}}
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--dry", action="store_true", help="one short process per arm, no pre-flight")
    r.set_defaults(fn=run)
    q = sp.add_parser("report")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--json", type=Path)
    q.set_defaults(fn=report)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
