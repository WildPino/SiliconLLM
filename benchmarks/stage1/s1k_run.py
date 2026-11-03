"""Stage 1 (c) runner: the registered kernel cells of s1k_bench, in three rounds, on an idle machine.  It prints and
records numbers, never a verdict: s1k_verdict.py decides.

Each launch is one process: its CONFIG and RESULT lines must confirm the registered build, type, cell and shape, or the
launch is VOID.  Before each launch the machine's total CPU load is sampled three times over 3 s, after 1.5 s that
let the sampling shell itself settle (Win32_PerfFormattedData_PerfOS_Processor, _Total); above LOAD_MAX the launch is
VOID (busy) and is retried once at the end.  Binary hashes are pinned in PIN (from s1k_build.sh's output) and checked
before the first launch.

Usage:
    python -I benchmarks/stage1/s1k_run.py --bin C:/SiliconLLM_fast/s1k --out benchmarks/stage1/results/s1k_run.json
    python -I benchmarks/stage1/s1k_run.py --bin C:/SiliconLLM_fast/s1k --out <json> --dry   (small shapes, 1 round)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROUNDS = 3
LOAD_MAX = 10.0  # percent, total CPU, sampled before each launch
STREAM = ["--layers", "16", "--experts", "16", "--k", "4", "--tokens", "128", "--reps", "9", "--threads", "6"]
ATTN = ["--layers", "64", "--tokens", "64", "--reps", "9", "--threads", "6"]
RESIDENT = ["--resident", "--k", "4", "--tokens", "512", "--reps", "9", "--threads", "6"]
DRY = ["--layers", "2", "--experts", "6", "--k", "4", "--tokens", "2", "--reps", "1", "--threads", "6"]
BINS = {"llama.cpp": "s1k_bench_main.exe", "ik_llama.cpp": "s1k_bench_ik.exe", "bitnet.cpp": "s1k_bench_bitnet.exe",
        "bitnet.cpp/tl2": "s1k_bench_tl2.exe"}
PIN = {
    "s1k_bench_main.exe": "4f4477b5963e7c570da5b615ad5cb7330b3f3cf6123bfbdb357f62d023d30a80",
    "s1k_bench_ik.exe": "33929705fd90a1ee2b6e090887cf84dc8f080f6e416349b199b2cb92ccb9d575",
    "s1k_bench_bitnet.exe": "1312b3a60968656bd1aca0c20e0c845d1a163b8cbc9eb04723f1a9b64bd65f97",
    "s1k_bench_tl2.exe": "c77a2c455bddcaff873db0ef873bd78f73a03730a6a5be2be47cae4eee73babe",
}


def cells() -> list[dict]:
    """The registered cells: (build, type, cell, shape set)."""
    c = []
    for t in ("tq2_0", "tq1_0", "q4_K", "q2_K", "q8_0"):
        c.append(("llama.cpp", t, "experts", STREAM))
    for t in ("iq2_bn", "iq2_bn_r4", "iq1_bn", "q4_K"):
        c.append(("ik_llama.cpp", t, "experts", STREAM))
    c += [("llama.cpp", "tq2_0", "experts2d", STREAM), ("ik_llama.cpp", "iq2_bn", "experts2d", STREAM),
          ("bitnet.cpp", "i2_s", "experts2d", STREAM), ("bitnet.cpp/tl2", "tl2", "experts2d", STREAM)]
    for b, t in (("llama.cpp", "tq2_0"), ("llama.cpp", "tq1_0"), ("llama.cpp", "q4_K"), ("ik_llama.cpp", "iq2_bn"),
                 ("ik_llama.cpp", "iq1_bn"), ("bitnet.cpp", "i2_s"), ("bitnet.cpp/tl2", "tl2")):
        c.append((b, t, "attn", ATTN))
    # resident: 4 experts' matrices, 13 MB at 2 bits, inside the L3 (Q8_0 would take 53 MB and is not resident)
    for b, t in (("llama.cpp", "tq2_0"), ("llama.cpp", "tq1_0"), ("ik_llama.cpp", "iq2_bn"), ("ik_llama.cpp", "iq1_bn")):
        c.append((b, t, "experts", RESIDENT))
    for b, t in (("bitnet.cpp", "i2_s"), ("bitnet.cpp/tl2", "tl2")):
        c.append((b, t, "experts2d", RESIDENT))
    return [{"build": b, "type": t, "cell": cl, "args": a} for b, t, cl, a in c]


def key(c: dict) -> str:
    return f"{c['build']}|{c['type']}|{c['cell']}|{'resident' if '--resident' in c['args'] else 'stream'}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pct() -> float:
    # Win32_PerfFormattedData_PerfOS_Processor: the counter names of Get-Counter are localized, this class is not
    ps = ("Start-Sleep -Milliseconds 1500; $v = 1..3 | ForEach-Object { (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
          "-Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Milliseconds 1000 }; "
          "($v | Measure-Object -Average).Average")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    try:
        return float(r.stdout.strip().replace(",", "."))
    except ValueError:
        return float("nan")


def launch(bindir: Path, c: dict, dry: bool, tl2_dir: Path) -> dict:
    exe = bindir / BINS[c["build"]]
    args = (DRY + (["--resident"] if "--resident" in c["args"] else [])) if dry else c["args"]
    cmd = [str(exe), "--type", c["type"], "--cell", c["cell"], *args]
    if c["build"] == "bitnet.cpp/tl2":
        cmd += ["--tl2-dir", str(tl2_dir)]
    load = load_pct()
    rec = {"key": key(c), "cmd": cmd, "load_pct_before": load, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not dry and not load <= LOAD_MAX:  # a dry run checks the plumbing, not the timing: load is recorded only
        rec["status"] = "VOID busy"
        return rec
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["returncode"] = r.returncode
    rec["stderr"] = r.stderr[-2000:]
    cfg = res = None
    reps = []
    for line in r.stdout.splitlines():
        if line.startswith("CONFIG "):
            cfg = json.loads(line[7:])
        elif line.startswith("RESULT "):
            res = json.loads(line[7:])
        elif line.startswith("REP "):
            reps.append(line)
    rec.update(config=cfg, result=res, reps=reps)
    build = "bitnet.cpp" if c["build"] == "bitnet.cpp/tl2" else c["build"]
    ok = (r.returncode == 0 and cfg is not None and res is not None and cfg["build"] == build
          and cfg["type"] == c["type"] and cfg["cell"] == c["cell"]
          and bool(cfg["resident"]) == ("--resident" in args) and cfg["threads"] == 6)
    rec["status"] = "COMPLETE" if ok else "VOID config"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tl2-dir", type=Path, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    tl2_dir = a.tl2_dir or a.bin / "tl2"
    hashes = {n: sha256(a.bin / n) for n in PIN}
    bad = [n for n in PIN if hashes[n] != PIN[n]]
    if bad and not a.dry:
        raise SystemExit(f"binary hash differs from its pin: {bad}")
    tl2 = {p.name: sha256(p) for p in sorted(tl2_dir.glob("*.tl2"))}
    report = {"cell": "S1K-kernel-rate", "dry": a.dry, "binaries_sha256": hashes, "tl2_weights_sha256": tl2,
              "rounds": 1 if a.dry else ROUNDS, "load_max_pct": LOAD_MAX, "launches": []}
    todo = [(rnd, c) for rnd in range(report["rounds"]) for c in cells()]
    retry = []
    for rnd, c in todo:
        rec = launch(a.bin, c, a.dry, tl2_dir)
        rec["round"] = rnd
        report["launches"].append(rec)
        res = rec.get("result") or {}
        print(f"{rec['status']:12s} r{rnd} {rec['key']:45s} load {rec['load_pct_before']:5.1f}% "
              f"{res.get('median_gweights_per_s', float('nan')):7.2f} Gw/s {res.get('median_gbytes_per_s', float('nan')):6.2f} "
              f"GB/s err {res.get('rel_err', float('nan')):.4f}", flush=True)
        if rec["status"] == "VOID busy":
            retry.append((rnd, c))
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for rnd, c in retry:
        rec = launch(a.bin, c, a.dry, tl2_dir)
        rec["round"] = rnd
        rec["retry"] = True
        report["launches"].append(rec)
        print(f"{rec['status']:12s} r{rnd} retry {rec['key']}", flush=True)
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
