"""Stage 2's last chains, as registered in S2G (test 2c) and S2K (the fourth code, step 2).  Resumable: a step whose output exists is skipped.

    gpu    after test 2c's trainings: ph_both's weights; the scale-10 reading on layer 2; bal's lr pick on stage 1's grid, the projection, the training,
           the exact export, the panels at k = 8 and 4.  It stops if the pick projects past 10 h of GPU, unless --no-limit (S2K, amendment 1).
    panels after test 2c's trainings: each arm's export on E:, then its panels.  ph_both waits for its weights.

Usage: python -I benchmarks/stage2/s2_tail_chain.py gpu [--no-limit] | panels
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
PY = sys.executable
ARMS_LOG = Path("C:/SiliconLLM_fast/s2s/resume_arms.out.log")
S2S = Path("C:/SiliconLLM_fast/s2s")
OUT = Path("E:/SiliconLLM_Artifacts/stage2")
PH = OUT / "s2s" / "ph_both"
PH_DONE = PH / "DONE"
WORK = "E:/SiliconLLM_Artifacts/stage1/s1a"
GPU_LIMIT_S = 36000
TERN_PICK_S, TERN_TRAIN_S = 2541.1, 29962.4


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run([PY, "-I", *cmd], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"stopped ({r.returncode}): {' '.join(cmd)}")


def wait_trainings() -> None:
    while "ARMS DONE" not in ARMS_LOG.read_text(encoding="utf-8", errors="replace"):
        if "ARMS STOPPED" in ARMS_LOG.read_text(encoding="utf-8", errors="replace"):
            raise SystemExit("test 2c's trainings stopped")
        time.sleep(60)


def gpu() -> None:
    wait_trainings()
    if not PH_DONE.exists():
        step("benchmarks/stage2/s2s_train.py", "posthoc", "--arm", "st_both", "--out", str(PH))
        PH_DONE.write_text("done\n", encoding="ascii")
    if not (R / "s2s_pick_10.json").exists():
        step("benchmarks/stage2/s2s_train.py", "train", "--arm", "st_both", "--dir", str(OUT / "s2s" / "teacher"), "--layers", "2-2", "--lr-scale", "10",
             "--pick-from", str(S2S / "pick_base"), "--out", str(S2S / "pick_10"), "--json", str(R / "s2s_pick_10.json"))
    pick = R / "s2v_lrpick.json"
    base = ["--k", "8", "--ternary", "256", "--bal"]
    if not pick.exists():
        step("benchmarks/stage1/s1t_train.py", *base, "--layers", "1-1", "--lr-grid", "1e-3,3e-3,1e-2", "--out", str(OUT / "s2v" / "pick"), "--json", str(pick))
    rec = json.loads(pick.read_text(encoding="utf-8"))["layers"]["1"]
    projected = rec["seconds"] * (1 + TERN_TRAIN_S / TERN_PICK_S)
    print("PROJECTED GPU", round(projected), "s; lr", rec["lr"], rec["lr_grid"], flush=True)
    if projected > GPU_LIMIT_S and "--no-limit" not in sys.argv:
        raise SystemExit("the pick projects past 10 h of GPU: stopped for the user's decision")
    train = R / "s2v_train.json"
    done = json.loads(train.read_text(encoding="utf-8")).get("layers", {}) if train.exists() else {}
    if len(done) < 19:
        step("benchmarks/stage1/s1t_train.py", *base, "--lr", str(rec["lr"]), "--out", str(OUT / "s2v" / "bal"), "--json", str(train))
    ex = R / "s2v_export.json"
    q4 = OUT / "s2v" / "bal-q4.gguf"
    if not ex.exists():
        step("benchmarks/stage1/s1t_export.py", "patch", "--trained", str(OUT / "s2v" / "bal"), "--bal", "--out-bf16", str(OUT / "s2v" / "bal-bf16.gguf"),
             "--out-q4", str(q4), "--json", str(ex))
    e = json.loads(ex.read_text(encoding="utf-8"))
    if not e["bal_exact"]["exact_to_bf16"]:
        raise SystemExit("the export is not exact")
    for arm in ("bal", "bal4"):
        step("benchmarks/stage2/s2x_run.py", "arm", "--arm", arm, "--out", str(R / "s2x_run.json"), "--model", str(q4), "--model-sha", e["q4_sha256"])
    print("GPU CHAIN DONE", flush=True)


def panels() -> None:
    wait_trainings()
    q = OUT / "s2s" / "q4"
    q.mkdir(parents=True, exist_ok=True)
    for arm, trained, layers in (("st_both", S2S / "st_both", ""), ("st_attn", S2S / "st_attn", ""), ("st_ffn", S2S / "st_ffn", ""), ("st_attn_lo", S2S / "st_attn", "0-9"),
                                 ("st_attn_hi", S2S / "st_attn_hi", ""), ("ph_both", PH, "")):
        if arm == "ph_both":
            while not PH_DONE.exists():
                time.sleep(60)
        js = R / f"s2s_export_{arm}.json"
        if not js.exists():
            step("benchmarks/stage2/s2s_export.py", "--trained", str(trained), *(["--layers", layers] if layers else []), "--out-q4", str(q / f"{arm}.gguf"), "--json", str(js))
        sha = json.loads(js.read_text(encoding="utf-8"))["q4_sha256"]
        step("benchmarks/stage2/s2s_run.py", "--work", WORK, "--arm", arm, "--model", str(q / f"{arm}.gguf"), "--model-sha", sha, "--out", str(R / "s2s_run.json"))
    print("PANELS CHAIN DONE", flush=True)


if __name__ == "__main__":
    {"gpu": gpu, "panels": panels}[sys.argv[1]]()
