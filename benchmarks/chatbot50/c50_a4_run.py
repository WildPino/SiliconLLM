"""C50-A4 runner: both arms (BF16 source GGUF, P0 Q4_K_M) through the same pinned llama.cpp tools on
the same A4 inputs.  Numbers only; c50_a4_checks.py scores each arm and c50_a4_verdict.py applies the
gates.  Resumable: a panel whose outputs and CONFIG are already recorded as COMPLETE is not re-run.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A4_QUALITY_PROTOCOL_20261017.md

Usage:
    python -I benchmarks/chatbot50/c50_a4_run.py --work E:/.../a4 --out benchmarks/chatbot50/results/c50_a4_run.json \
        [--arms p0,bf16] [--budget-h 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GGUF = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/gguf")
MODELS = {"p0": (GGUF / "ling-mini-2.0-a810f64-q4_k_m.gguf",
                 "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"),
          "bf16": (GGUF / "ling-mini-2.0-a810f64-bf16.gguf", None),  # BF16 hash recorded in c50_a2_gguf_check_bf16.json
          # planted positive (A4 protocol): Q2_K from the same BF16 and imatrix; the gates must fail it
          "pc": (GGUF / "ling-mini-2.0-a810f64-q2_k.gguf",
                 "278738d5cf27b34379d189d32dcd645f1ccc65f6e3d31240214e070b959e33d6")}
STOP = "156892,156895"
# (panel, tool, extra args)
PANELS = [("docs", "score", ["--ctx", "8192"]), ("piqa", "score", ["--ctx", "4096"]),
          ("arc", "score", ["--ctx", "4096"]), ("chat", "generate", ["--max-new", "256"]),
          ("gsm8k", "generate", ["--max-new", "384"]), ("rollout", "generate", ["--max-new", "256"]),
          ("ifeval", "generate", ["--max-new", "384"])]
MAX_PROJECTED_S = 3 * 3600  # per process: a runaway guard; the authorized budget is --budget-h


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="p0,bf16,pc")
    ap.add_argument("--budget-h", type=float, default=12.0)
    args = ap.parse_args()
    work = Path(args.work)
    inputs = work / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    for name, rec in manifest["files"].items():
        if sha256(inputs / f"{name}.tsv") != rec["tsv_sha256"]:
            raise SystemExit(f"input {name}.tsv differs from its manifest")
    outp = Path(args.out)
    report = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {
        "cell": "C50-A4", "inputs_manifest_sha256": sha256(inputs / "manifest.json"), "arms": {}}
    report.setdefault("sessions", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": args.arms,
                                              "budget_h": args.budget_h})
    t_start = time.time()
    for arm in args.arms.split(","):
        model, pin = MODELS[arm]
        a = report["arms"].setdefault(arm, {"model": str(model), "panels": {}})
        if "model_sha256" not in a:
            a["model_sha256"] = sha256(model)
        if pin and a["model_sha256"] != pin:
            raise SystemExit(f"{arm}: model hash differs from its pin")
        (work / arm).mkdir(exist_ok=True)
        for panel, tool, extra in PANELS:
            if a["panels"].get(panel, {}).get("status") == "COMPLETE":
                continue
            if time.time() - t_start > args.budget_h * 3600:
                a["panels"][panel] = {"status": "NOT STARTED: session budget spent"}
                continue
            exe = HERE / "bin" / f"c50_llama_{tool}_avx2.exe"
            flag = "--ids" if tool == "score" else "--prompts"
            cmd = [str(exe), "--model", str(model), flag, str(inputs / f"{panel}.tsv"), "--out", str(work / arm / panel),
                   "--threads", "6", "--ubatch", "512", "--max-projected-s", str(MAX_PROJECTED_S)] + extra
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
                  and "flash_attn=disabled" in c and (tool == "score" or f"stop={STOP} " in c))
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
