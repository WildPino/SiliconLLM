"""Reference decode speeds of llama.cpp and ik_llama.cpp on the original Ling-mini-2.0 and Ling-flash-2.0 Q4_K_M files.

Protocol: docs/research/WALL_MAP_20261029/WALL_MAP_PROTOCOL_20261029.md (item 6). Idle box, the user's go.
Order:
    pins (engine binaries, model sha256)
    correctness check, per file and cell: llama-perplexity at one token per ubatch (the decode path) on a fixed text;
        every cell within PPL_BAR of llama.cpp fa_off on the same file, and a planted wrong model (Ling-mini with 4
        experts per token instead of 8) beyond it, or the run stops
    pre-flight: 45 s, mean foreign occupancy < 4.39% (E44, as in A2b and S0)
    3 rounds; each round visits every cell (engine, flags, file) once, rotated by the round; one process per visit.
        Before each process the file is read once end to end (page-in: -rtr turns mmap off and its private copy of
        Ling-flash evicts the page cache). Each process runs `llama-bench -t 6 -p 0 -n 128 -r 5 -o json` at depths 0
        and 2048 with the cell's flags; the depth is `-d 0,2048` in llama.cpp and `-gp 2048,128` in ik_llama.cpp
        (which has no -d): both time only the 128 generated tokens after the 2048-token prompt is in the cache
A process at or above the bar is VOID and replaced (at most 5 per cell). The process's JSON must confirm the file, the
thread count, the depths and the cell's flags, or the run stops. Numbers only: w0_ref_read.py.

    python -I benchmarks/wallmap/w0_ref_speed.py --out benchmarks/wallmap/results/w0_ref_speed_1.json [--dry]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "donor_adaptation" / "engine"))
import e44_interval as E44  # noqa: E402

GCC_BIN = Path(os.environ["LOCALAPPDATA"]) / ("Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft."
                                               "Winget.Source_8wekyb3d8bbwe/mingw64/bin")
LLAMA_BIN = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/llama.cpp-5b335f4/build-native/bin")
# ik_llama.cpp is built with -Wa,-muse-unaligned-vector-move: without it its flash-attention kernel crashes (access
# violation) under MinGW on both files (29 October)
IK_BIN = Path("E:/SiliconLLM_Artifacts/wallmap/tools/ik_llama.cpp/build-native-ua/bin")
ENGINES = {
    "llama": {"bench": LLAMA_BIN / "llama-bench.exe", "ppl": LLAMA_BIN / "llama-perplexity.exe", "commit": "5b335f4"},
    "ik": {"bench": IK_BIN / "llama-bench.exe", "ppl": IK_BIN / "llama-perplexity.exe", "commit": "d4c1f75"},
}
FILES = {
    "mini": (Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/ling-mini-2.0-a810f64-q4_k_m.gguf"),
             "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"),
    "flash": (Path("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf"),
              "5348c73fe6497c5152e16fd792d6c7b5aeefa15de8b15b8c8e18bef691704b8e"),
}
# the grid of the protocol: (engine, cell, llama-bench flags, the JSON fields that must confirm them, perplexity flags).
# ik_llama.cpp's fused MoE is on by default, so its plain cells turn it off explicitly
CELLS = [
    ("llama", "fa_off", ["-fa", "off"], {"flash_attn": False}, ["-fa", "off"]),
    ("llama", "fa_on", ["-fa", "on"], {"flash_attn": True}, ["-fa", "on"]),
    ("ik", "fa0", ["-fa", "0", "-fmoe", "0", "-rtr", "0"], {"flash_attn": False, "fused_moe": False, "repack": False},
     ["-fa", "0", "-no-fmoe"]),
    ("ik", "fa1", ["-fa", "1", "-fmoe", "0", "-rtr", "0"], {"flash_attn": True, "fused_moe": False, "repack": False},
     ["-fa", "1", "-no-fmoe"]),
    ("ik", "fa0_moe", ["-fa", "0", "-fmoe", "1", "-rtr", "1"], {"flash_attn": False, "fused_moe": True, "repack": True},
     ["-fa", "0", "-rtr"]),
    ("ik", "fa1_moe", ["-fa", "1", "-fmoe", "1", "-rtr", "1"], {"flash_attn": True, "fused_moe": True, "repack": True},
     ["-fa", "1", "-rtr"]),
]
COMMON = ["-t", "6", "-p", "0", "-n", "128", "-r", "5", "-o", "json"]
DEPTH_ARGS = {"llama": ["-d", "0,2048"], "ik": ["-gp", "2048,128"]}
ROUNDS, OCC_BAR, MAX_REPLACEMENTS, PREFLIGHT_SECONDS = 3, 4.39, 5, 45
# amendment of 29 October (run 1 VOID): Ling-flash processes raise the box's foreign occupancy by themselves, with or
# without mmap (6.20% / 6.26%, w0_ref_diag_loadmode.json); six fa_off processes read 5.70-8.30% while their rate moved
# 4% (depth 0) and 2% (depth 2048). The flash cells get their own bar; the pre-flight and Ling-mini keep 4.39
OCC_BAR_FILE = {"mini": OCC_BAR, "flash": 9.0}
# correctness check: 4 chunks of 256 tokens, one token per ubatch; the bar sits above the spread of valid
# configurations measured on 29 October on Ling-mini (3.564..3.690, 3.4% from the reference) and below the planted
# wrong model (+14.4%)
PPL_TEXT = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf/imatrix_fit_input.txt")
PPL_ARGS = ["-c", "256", "--chunks", "4", "-b", "256", "-ub", "1", "-t", "6"]
PPL_BAR = 0.06
PLANTED = ["--override-kv", "bailingmoe2.expert_used_count=int:4"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def page_in(path: Path) -> float:
    t0 = time.time()
    with path.open("rb", buffering=0) as f:
        while f.read(1 << 26):
            pass
    return time.time() - t0


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = str(GCC_BIN) + os.pathsep + e["PATH"]          # the WinLibs runtime DLLs the binaries link
    return e


def truth(v) -> bool:
    """A flag as either engine's JSON writes it: a boolean, a number, or a string such as "0"."""
    return bool(int(v)) if isinstance(v, str) and v.strip().lstrip("-").isdigit() else bool(v)


def depth(engine: str, r: dict) -> int:
    """The KV depth a JSON row was generated at: llama.cpp's n_depth; in ik_llama.cpp a tg128@pp2048 row."""
    if engine == "llama":
        return int(r.get("n_depth", -1))
    return int(r["n_prompt"]) if r.get("test", "").startswith("tg128@pp") else (0 if r.get("test") == "tg128" else -1)


def perplexity(engine: str, file: str, flags: list[str]) -> dict:
    cmd = [str(ENGINES[engine]["ppl"]), "-m", str(FILES[file][0]), "-f", str(PPL_TEXT)] + PPL_ARGS + flags
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, env=env())
    text = (p.stdout + p.stderr).decode(errors="replace")
    m = re.search(r"Final estimate: PPL(?: over \d+ chunks for n_ctx=\d+)? = ([0-9.]+)", text)
    return {"engine": engine, "file": file, "flags": flags, "returncode": p.returncode,
            "ppl": float(m.group(1)) if m else None, "wall_s": time.time() - t0, "tail": text[-400:]}


def check(files: list[str], report: dict, save) -> bool:
    out = report["check"] = {"bar": PPL_BAR, "text": str(PPL_TEXT), "args": PPL_ARGS, "runs": []}
    ok = True
    for f in files:
        ref = None
        for e, n, _, _, pfl in CELLS:
            r = perplexity(e, f, pfl)
            r["cell"] = n
            ref = r["ppl"] if (e, n) == ("llama", "fa_off") else ref
            r["rel"] = (r["ppl"] / ref - 1) if (r["ppl"] and ref) else None
            r["pass"] = r["rel"] is not None and abs(r["rel"]) <= PPL_BAR
            ok &= r["pass"]
            out["runs"].append(r)
            save()
            print(f"check {f} {e}/{n} ppl={r['ppl']} rel={r['rel']} pass={r['pass']} wall={r['wall_s']:.0f}s", flush=True)
        if f == "mini":
            r = perplexity("llama", f, ["-fa", "off"] + PLANTED)
            r["cell"] = "PLANTED_top4"
            r["rel"] = (r["ppl"] / ref - 1) if (r["ppl"] and ref) else None
            r["fires"] = r["rel"] is not None and abs(r["rel"]) > PPL_BAR
            ok &= r["fires"]
            out["runs"].append(r)
            save()
            print(f"check {f} PLANTED ppl={r['ppl']} rel={r['rel']} fires={r['fires']}", flush=True)
    out["pass"] = ok
    return ok


def one_process(engine: str, flags: list[str], file: str, want: dict) -> dict:
    page_s = page_in(FILES[file][0])
    split = E44.Split()
    t0 = time.time()
    cmd = [str(ENGINES[engine]["bench"]), "-m", str(FILES[file][0])] + COMMON + DEPTH_ARGS[engine] + flags
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env())
    out, err = proc.communicate()
    child = E44._process_times(int(proc._handle))
    system, foreign = split.close(child)
    rec = {"engine": engine, "file": file, "flags": flags, "cmd": cmd, "returncode": proc.returncode,
           "page_in_s": page_s, "stderr_tail": err.decode(errors="replace")[-600:], "system_pct": system,
           "foreign_pct": foreign, "wall_s": time.time() - t0, "timing": split.record()}
    try:
        rows = json.loads(out.decode(errors="replace"))
    except json.JSONDecodeError:
        rows = None
    rec["rows"] = rows
    ok = proc.returncode == 0 and isinstance(rows, list) and len(rows) == 2
    if ok:
        for r in rows:
            r["depth"] = depth(engine, r)
            ok &= Path(r["model_filename"]).resolve() == FILES[file][0].resolve()
            ok &= int(r["n_threads"]) == 6 and int(r["n_gen"]) == 128
            ok &= int(r["n_prompt"]) == (0 if engine == "llama" else r["depth"])
            ok &= ENGINES[engine]["commit"] in str(r.get("build_commit", ""))
            ok &= all(k in r and truth(r[k]) == v for k, v in want.items())
            ok &= len(r.get("samples_ts", [])) == 5
        ok &= sorted(r["depth"] for r in rows) == [0, 2048]
    if not ok:
        rec["status"] = "FAILED"
        return rec
    rec["status"] = "VOID" if not (foreign == foreign and foreign < OCC_BAR_FILE[file]) else "OK"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="pins, the check and one process per cell on Ling-mini, no "
                                                       "pre-flight; not a run")
    a = ap.parse_args()
    out_path = Path(a.out)
    report = {"cell": "wall map item 6, reference speeds", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "occ_bar_pct": OCC_BAR, "occ_bar_by_file": OCC_BAR_FILE, "common": COMMON, "depth_args": DEPTH_ARGS, "cells": [c[:3] for c in CELLS],
              "dry": a.dry, "exe_sha256": {f"{k}/{t}": sha256(v[t]) for k, v in ENGINES.items() for t in ("bench", "ppl")},
              "ppl_text_sha256": sha256(PPL_TEXT), "processes": []}

    def save():
        out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    files = ["mini"] if a.dry else list(FILES)
    report["file_sha256"] = {f: sha256(FILES[f][0]) for f in files}
    if any(report["file_sha256"][f] != FILES[f][1] for f in files):
        report["verdict_input"] = "VOID: a model file differs from its pin"
        save()
        print(report["verdict_input"])
        return 1
    if not check(files, report, save):
        report["verdict_input"] = "VOID: the correctness check failed (a cell off the reference, or the planted model not detected)"
        save()
        print(report["verdict_input"])
        return 1
    if not a.dry:
        pre = E44.watch(PREFLIGHT_SECONDS)
        mean = sum(pre) / len(pre) if pre else float("nan")
        report["preflight"] = {"seconds": PREFLIGHT_SECONDS, "samples": pre, "mean": mean,
                               "admissible": bool(pre) and mean < OCC_BAR}
        print(f"preflight mean={mean:.2f}% admissible={report['preflight']['admissible']}", flush=True)
        if not report["preflight"]["admissible"]:
            report["verdict_input"] = "VOID: pre-flight occupancy at or above the bar"
            save()
            return 1
    visits = [(e, n, fl, w, f) for f in files for e, n, fl, w, _ in CELLS]
    replacements: dict[str, int] = {}
    for rnd in range(1 if a.dry else ROUNDS):
        for e, n, fl, w, f in visits[rnd % len(visits):] + visits[:rnd % len(visits)]:
            key = f"{e}/{n}/{f}"
            while True:
                rec = one_process(e, fl, f, w)
                rec["phase"], rec["round"], rec["cell"] = "run", rnd, n
                report["processes"].append(rec)
                save()
                ts = [round(r["avg_ts"], 2) for r in rec["rows"]] if rec["status"] != "FAILED" else None
                print(f"r{rnd} {key} {rec['status']} foreign={rec['foreign_pct']:.2f}% tok/s(d0,d2048)={ts} "
                      f"page_in={rec['page_in_s']:.0f}s wall={rec['wall_s']:.0f}s", flush=True)
                if rec["status"] == "FAILED":
                    report["verdict_input"] = f"VOID: process failed or its JSON did not confirm the cell ({key})"
                    save()
                    return 1
                if rec["status"] == "OK" or a.dry:
                    break
                replacements[key] = replacements.get(key, 0) + 1
                if replacements[key] > MAX_REPLACEMENTS:
                    report["verdict_input"] = f"VOID: more than {MAX_REPLACEMENTS} replacements ({key})"
                    save()
                    return 1
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["verdict_input"] = "DRY" if a.dry else "COMPLETE"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
