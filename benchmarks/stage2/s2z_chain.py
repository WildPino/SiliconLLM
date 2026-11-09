"""S2M's chains, resumable.  quant: IQ2_KS on the experts (s2z_ks.py).  cpu: ks_org's five panels, ks_stk4, then ks_all4 once quant has written its record.

Usage: python -I benchmarks/stage2/s2z_chain.py quant | cpu
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = HERE / "results"
Y = Path("E:/SiliconLLM_Artifacts/stage2/s2y")
SCORES = "E:/SiliconLLM_Artifacts/stage1/s1a"
A4, S2 = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs", "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs"
PY = sys.executable


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, "-I", *cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"stopped ({r.returncode}): {' '.join(cmd)}")


def panels(name: str, q4: Path, js: Path) -> None:
    sha = json.loads(js.read_text(encoding="utf-8"))["q4_sha256"]
    step("benchmarks/stage2/s2z_run.py", "--arm", name, "--model", str(q4), "--model-sha", sha, "--out", str(R / "s2z_run.json"))
    step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", A4, "--s2-inputs", S2, "--arm-a4", f"{SCORES}/{name}", "--arm-s2", f"{SCORES}/{name}", "--out", f"{SCORES}/scores/{name}.json")


def arm(name: str) -> None:
    js, q4 = R / f"s2z_export_{name}.json", Y / "q4" / f"{name}.gguf"
    if not js.exists():
        step("benchmarks/stage2/s2z_export.py", "--arm", name, "--out-q4", str(q4), "--json", str(js))
    panels(name, q4, js)


def quant() -> None:
    if not (R / "s2z_ks.json").exists():
        step("benchmarks/stage2/s2z_ks.py", "--json", str(R / "s2z_ks.json"))
    print("QUANT DONE", flush=True)


def cpu() -> None:
    panels("ks_org", Y / "q4" / "ks_org.gguf", R / "s2y_export_ks_org.json")
    arm("ks_stk4")
    while not (R / "s2z_ks.json").exists():
        time.sleep(60)
    if not json.loads((R / "s2z_ks.json").read_text(encoding="utf-8"))["control"]:
        raise SystemExit("stopped (s2z_ks control failed): ks_all4 not exported")
    arm("ks_all4")
    print("S2M CHAIN DONE", flush=True)


if __name__ == "__main__":
    {"quant": quant, "cpu": cpu}[sys.argv[1]]()
