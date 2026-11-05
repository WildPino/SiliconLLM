"""Stage 2 test 5, amendment 2: the engine and kernel cells that close the desk budget at flash's shapes, in the idle session.

Kernel arms run s2k_bench.exe, ik_llama.cpp's kernels; engine arms run s2k_engine.exe, c50_engine.c's own code at flash's shapes.  3 rounds,
rotated.  A process at or above the occupancy bar is VOID and replaced, at most 5 per arm; CONFIG must confirm the cell.
`report` prints the numbers, it labels nothing.  VOID if a kernel's check is off (rel_err >= 0.05, or the wrong rows under 0.5), if the KV at context 512
does not take under 0.6 of the KV at 2048, or if the fixed term did not run 7 jobs per layer.  q8_0 against f16 is a result, not a control.

Usage: python -I benchmarks/stage2/s2k_cells.py run --out benchmarks/stage2/results/s2k_cells.json [--dry]
       python -I benchmarks/stage2/s2k_cells.py report --run benchmarks/stage2/results/s2k_cells.json
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

BENCH, ENGINE = HERE / "bin" / "s2k_bench.exe", HERE / "bin" / "s2k_engine.exe"
ROUNDS = 3
FF = "1024x4096:{t},1024x4096:{t},4096x1024:{t}"
KERNEL = {"kv_q4k": ("512x4096:q4_K", 0), "kv_bn": ("512x4096:iq2_bn", 0),                                     # 1
          "qkv_one": ("5120x4096:iq2_bn", 0), "qkv_two": ("4096x4096:iq2_bn,1024x4096:iq2_bn", 0),            # 2
          "qkv_mix": ("4096x4096:iq2_bn,1024x4096:q4_K", 0),
          "f_iq2ks": ("4096x4096:iq2_ks", 0), "f_iq3ks": ("4096x4096:iq3_ks", 0), "f_q3k": ("4096x4096:q3_K", 0),  # 3
          "f_iq2kl": ("4096x4096:iq2_kl", 0),
          "shared": (FF.format(t="iq2_bn"), 0),                                                                # 4; also cell 7's ternary pair
          "router": ("256x4096:q8_0", 0),                                                                      # 6
          "sq_tern": ("4096x4096:iq2_bn", 0), "sq_four": ("4096x4096:iq2_bn", 1), "ff_four": (FF.format(t="iq2_bn"), 1)}   # 7
ENGINE_ARMS = {"fixed": ["--cell", "fixed", "--steps", "256"],                                                          # 4
               "kv_f16": ["--cell", "kv", "--kv", "f16", "--ctx", "2048", "--steps", "64"],                               # 5
               "kv_q8": ["--cell", "kv", "--kv", "q8_0", "--ctx", "2048", "--steps", "64"],
               "kv_f16_512": ["--cell", "kv", "--kv", "f16", "--ctx", "512", "--steps", "64"]}                            # 5, its control
JOBS_PER_STEP = 7 * 32
ARMS = list(KERNEL) + list(ENGINE_ARMS)


def one(arm: str, dry: bool) -> dict:
    if arm in KERNEL:
        mats, four = KERNEL[arm]
        cmd = [str(BENCH), "--mats", mats, "--threads", "6"] + (["--four"] if four else []) + (["--tokens", "4", "--reps", "2", "--footprint-mb", "128"] if dry else [])
        want = f'"mats": "{mats}", "four": {four}'
    else:
        cmd = [str(ENGINE)] + ENGINE_ARMS[arm]
        c = ENGINE_ARMS[arm]
        want = '"cell": "fixed"' if arm == "fixed" else f'"cell": "kv", "kv": "{c[3]}", "ctx": {c[5]}'
    split = E44.Split()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
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
    rep = {"cell": "S2K-cells", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "bench_sha256": S0.sha256(BENCH), "engine_sha256": S0.sha256(ENGINE),
           "rounds": ROUNDS, "processes": [], "occ_bar_pct": S0.OCC_BAR}

    def save():
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("CONFIG", json.dumps({k: rep[k] for k in ("bench_sha256", "engine_sha256", "rounds")}), ARMS, flush=True)
    if not a.dry:
        pre = E44.watch(S0.PREFLIGHT_SECONDS)
        mean = sum(pre) / len(pre) if pre else float("nan")
        rep["preflight"] = {"mean": mean, "admissible": bool(pre) and mean < S0.OCC_BAR}
        if not rep["preflight"]["admissible"]:
            rep["verdict_input"] = "VOID: pre-flight occupancy"
            save()
            return 1
    repl: dict[str, int] = {}
    for r in range(1 if a.dry else ROUNDS):
        for arm in ARMS[r % len(ARMS):] + ARMS[:r % len(ARMS)]:
            while True:
                rec = one(arm, a.dry)
                rec["round"] = r
                rep["processes"].append(rec)
                save()
                print(f"r{r} {arm} {rec['status']} foreign={rec.get('foreign_pct', float('nan')):.2f}% {rec.get('ms_per_32', rec.get('ms_median'))} wall={rec['wall_s']:.0f}", flush=True)
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
    ok = [p for p in r["processes"] if p["status"] == "OK"]
    by = {x: [p for p in ok if p["arm"] == x] for x in ARMS}
    if not all(by.values()):
        print(json.dumps({"status": r.get("verdict_input"), "arms_without_ok_process": [x for x, v in by.items() if not v]}))
        return 1
    key = lambda p: p.get("ms_per_32", p.get("ms_median"))  # noqa: E731
    ms = {x: st.median(key(p) for p in v) for x, v in by.items()}
    checks = all(p["rel_err"] < 0.05 and p["rel_err_wrong"] > 0.5 for x in KERNEL for p in by[x])
    kv_moves = ms["kv_f16_512"] < 0.6 * ms["kv_f16"]
    jobs = all(abs(p["jobs"] - JOBS_PER_STEP) < 1 for p in by["fixed"])
    status = r.get("verdict_input") if checks and kv_moves and jobs else "VOID: a control failed"
    out = {"status": status, "ms_at_32_layers": ms, "min_max": {x: [min(map(key, v)), max(map(key, v))] for x, v in by.items()},
           "gweights_per_s": {x: st.median(p["gweights_per_s"] for p in by[x]) for x in KERNEL},
           "gbytes_per_s": {x: st.median(p["gbytes_per_s"] for p in by[x]) for x in list(KERNEL) + ["kv_f16", "kv_q8", "kv_f16_512"]},
           "split_cost_ms": ms["qkv_two"] - ms["qkv_one"], "four_over_tern": {"square": ms["sq_four"] / ms["sq_tern"], "ffn": ms["ff_four"] / ms["shared"]},
           "kernel_checks": checks, "kv_moves_with_context": kv_moves, "fixed_jobs": jobs}
    print(json.dumps(out, indent=1))
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
    q.set_defaults(fn=report)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
