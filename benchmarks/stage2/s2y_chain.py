"""S2L's two chains, resumable.  gpu: the two trainings, one after the other.  cpu: the arms that need no training, then each trained arm as it ends.

Usage: python -I benchmarks/stage2/s2y_chain.py gpu | cpu
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
TEACHER = "E:/SiliconLLM_Artifacts/stage2/s2s/teacher"
SCORES = "E:/SiliconLLM_Artifacts/stage1/s1a"
A4, S2 = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs", "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs"
PY = sys.executable
SEED2 = 20261107
FULL = ("as_both", "st_both2", "stk4")


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, "-I", *cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"stopped ({r.returncode}): {' '.join(cmd)}")


def train(arm: str, *extra: str) -> None:
    done = Y / arm / "DONE"
    if not done.exists():
        step("benchmarks/stage2/s2s_train.py", "train", "--arm", "st_both", "--dir", TEACHER, "--out", str(Y / arm), "--json", str(R / f"s2y_train_{arm}.json"), "--lr-scale", "3", *extra)
        done.write_text("done\n", encoding="ascii")


def gpu() -> None:
    train("as_both", "--form", "asym")
    train("st_both2", "--seed", str(SEED2))
    print("GPU CHAIN DONE", flush=True)


def arm(name: str, *extra: str) -> None:
    js, q4 = R / f"s2y_export_{name}.json", Y / "q4" / f"{name}.gguf"
    if not js.exists():
        step("benchmarks/stage2/s2y_export.py", "--arm", name, "--out-q4", str(q4), "--json", str(js), *extra)
    sha = json.loads(js.read_text(encoding="utf-8"))["q4_sha256"]
    step("benchmarks/stage2/s2y_run.py", "--arm", name, "--model", str(q4), "--model-sha", sha, "--out", str(R / "s2y_run.json"))
    if name in FULL:
        step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", A4, "--s2-inputs", S2, "--arm-a4", f"{SCORES}/{name}", "--arm-s2", f"{SCORES}/{name}", "--out", f"{SCORES}/scores/{name}.json")


def tool_check() -> None:
    """st_both's file through these binaries against test 2c's answers, on 30 nq items."""
    out = R / "s2y_tool_check.json"
    if out.exists():
        return
    sys.path.insert(0, str(HERE))
    from s2y_run import INPUTS, UA
    work = Path(SCORES)
    sub = Y / "tool_nq.tsv"
    sub.write_text("".join((INPUTS["s2"] / "nq.tsv").read_text(encoding="utf-8").splitlines(True)[:30]), encoding="utf-8", newline="")
    o = Y / "tool_nq"
    cmd = [str(UA["generate"]), "--model", "E:/SiliconLLM_Artifacts/stage2/s2s/q4/st_both.gguf", "--prompts", str(sub), "--out", str(o), "--threads", "6", "--ubatch", "512",
           "--max-projected-s", "32400", "--max-new", "32", "--moe-mode", "none", "--stop", "156892,156895"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"tool check failed ({r.returncode})")

    def rows(p: Path) -> dict:
        return {x["item"]: x["gen"] for x in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if x}
    new, old = rows(o.with_suffix(".jsonl")), rows(work / "st_both" / "nq.jsonl")
    equal = len(new) == 30 and all(old[k] == v for k, v in new.items())
    out.write_text(json.dumps({"items": len(new), "equal": equal}) + "\n", encoding="utf-8")
    print("TOOL CHECK", equal, flush=True)


def cpu() -> None:
    tool_check()
    arm("stk4")
    for name in ("ff_nodense", "ff_noearly", "ff_nolate", "ff_keepout", "ff_keeprnd"):
        arm(name)
    if not (R / "s2y_ks.json").exists():
        step("benchmarks/stage2/s2y_ks.py", "--json", str(R / "s2y_ks.json"))
    arm("ks_org", "--ks-bf16", str(Y / "ks-im-bf16.gguf"))
    for name in ("as_both", "st_both2"):
        while not (Y / name / "DONE").exists():
            time.sleep(60)
        arm(name)
    print("CPU CHAIN DONE", flush=True)


if __name__ == "__main__":
    {"gpu": gpu, "cpu": cpu}[sys.argv[1]]()
