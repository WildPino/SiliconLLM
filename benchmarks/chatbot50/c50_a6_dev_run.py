"""C50-A6 part 2 runner: the FIT candidate and R on DEV documents 13-32 and the A4 task panels, one runtime.

Arms: R (P0, KV f16) and C (the arm named by results/c50_a6_verdict.json "fit_candidate", with its build and KV
from c50_a6_run.py).  Runtime for both: c50_llama_score_a6_avx2 / c50_llama_generate_a6_avx2, flash attention on,
6 threads, ubatch 512; CONFIG asserted (AVX2, threads, ubatch, flash_attn=enabled, kv, stop IDs).  Numbers only:
c50_a4_checks.py scores each arm, c50_a6_dev_verdict.py applies the gates.  Resumable and budgeted like A4.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A6_PART2_DEV_TASKS_PROTOCOL_20261017.md

Usage:
    python -I benchmarks/chatbot50/c50_a6_dev_run.py --inputs E:/.../a6dev/inputs --work E:/.../a6dev \
        --out benchmarks/chatbot50/results/c50_a6_dev_run.json [--arms R,C] [--budget-h 9]
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
sys.path.insert(0, str(HERE))
from c50_a6_run import ARMS, gguf_path  # noqa: E402

STOP = "156892,156895"
PANELS = [("docs", "score", ["--ctx", "8192"]), ("piqa", "score", ["--ctx", "4096"]),
          ("arc", "score", ["--ctx", "4096"]), ("chat", "generate", ["--max-new", "256"]),
          ("gsm8k", "generate", ["--max-new", "384"]), ("rollout", "generate", ["--max-new", "256"]),
          ("ifeval", "generate", ["--max-new", "384"])]
MAX_PROJECTED_S = 3 * 3600
P0_PIN = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="R,C")
    ap.add_argument("--budget-h", type=float, default=9.0)
    args = ap.parse_args()
    inputs, work = Path(args.inputs), Path(args.work)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    for name, rec in manifest["files"].items():
        if sha256(inputs / f"{name}.tsv") != rec["tsv_sha256"]:
            raise SystemExit(f"input {name}.tsv differs from its manifest")
    verdict = json.loads((HERE / "results" / "c50_a6_verdict.json").read_text(encoding="utf-8"))
    cand = verdict["fit_candidate"]
    if cand in (None, "R"):
        raise SystemExit(f"FIT candidate {cand}: nothing to confirm (A6 stops at P0)")
    a6run = json.loads((HERE / "results" / "c50_a6_run.json").read_text(encoding="utf-8"))
    build, kv = ARMS[cand]
    arms = {"R": (gguf_path(None), P0_PIN, "f16"),
            "C": (gguf_path(build), P0_PIN if build is None else a6run["builds"][build]["sha256"], kv)}
    outp = Path(args.out)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "C50-A6 part 2", "fit_candidate": cand, "inputs_manifest_sha256": sha256(inputs / "manifest.json"), "arms": {}}
    if report["fit_candidate"] != cand:
        raise SystemExit("the FIT candidate changed since this run record was started")
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    for arm in args.arms.split(","):
        model, pin, kvt = arms[arm]
        a = report["arms"].setdefault(arm, {"model": str(model), "kv": kvt, "panels": {}})
        if "model_sha256" not in a:
            a["model_sha256"] = sha256(model)
        if a["model_sha256"] != pin:
            raise SystemExit(f"{arm}: model hash differs from its pin")
        (work / arm).mkdir(parents=True, exist_ok=True)
        for panel, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status") == "COMPLETE":
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            exe = HERE / "bin" / f"c50_llama_{tool}_a6_avx2.exe"
            flag = "--ids" if tool == "score" else "--prompts"
            cmd = [str(exe), "--model", str(model), flag, str(inputs / f"{panel}.tsv"), "--out", str(work / arm / panel),
                   "--threads", "6", "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S),
                   "--fa", "1", "--kv", kvt] + extra
            if tool == "generate":
                cmd += ["--stop", STOP]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = proc.stdout.splitlines()
            config = [ln for ln in lines if ln.startswith("CONFIG ")]
            rec = {"cmd": cmd, "exe_sha256": sha256(exe), "returncode": proc.returncode, "wall_s": time.time() - t0,
                   "config": config, "stdout_tail": lines[-3:], "stderr_tail": proc.stderr[-600:]}
            c = config[0] if len(config) == 1 else ""
            ok = (proc.returncode == 0 and "AVX2 = 1" in c and "threads=6 " in c and "ubatch=512 " in c
                  and "flash_attn=enabled " in c and f"kv={kvt} " in c and f"model={model} " in c
                  and (tool == "score" or f"stop={STOP} " in c))
            rec["status"] = ("COMPLETE" if ok else "STOPPED by the projected-time rule" if proc.returncode == 3
                             else "VOID: run failed or CONFIG does not confirm the registered settings")
            a["panels"][panel] = rec
            outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(arm, panel, rec["status"], f"{rec['wall_s']:.0f}s", flush=True)
    report["sessions"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    outp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
