"""C50-S4 Kaggle kernel: finds the datasets on the mount, prints the code manifest, runs TASK, and fails loudly.

TASK is set by `c50_s4_kaggle.py push <task>`.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

TASK = "__TASK__"


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


code = find("c50_s4_t0_gpu.py").parent
gguf = find("ling-mini-2.0-a810f64-q4_k_m.gguf")
print("TASK", TASK, "code", code, "gguf", gguf, flush=True)
print("MANIFEST", (code / "MANIFEST.json").read_text(encoding="utf-8"), flush=True)
subprocess.run(["nvidia-smi"], check=False)
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
           PYTHONPATH=os.pathsep.join([str(code), str(code / "gguf-py")]))
out = Path("/kaggle/working")
if TASK == "mve2":
    cmd = [sys.executable, "-u", str(code / "c50_s4_t0_gpu.py"), "--mve2", "--gguf", str(gguf), "--refs",
           str(code / "refs"), "--fingerprints", str(code / "refs/t0_dequant_fp32_sha256.json"),
           "--out", str(out / "mve2.json")]
elif TASK == "mve3":
    cmd = [sys.executable, "-u", str(code / "c50_s4_student.py"), "--mve3", "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--cap-min", "20", "--out", str(out / "mve3.json")]
elif TASK == "bank":
    cmd = [sys.executable, "-u", str(code / "c50_s4_bank.py"), "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "bank")]
elif TASK.startswith("train-"):
    # train-<arm>[+resume]: resumes from a `ckpt.pt` mounted from the previous session's kernel, if there is one;
    # with +resume a missing checkpoint fails at once instead of training silently from step 0.
    arm = TASK[len("train-"):]
    need_ck = arm.endswith("+resume")
    arm = arm.removesuffix("+resume")
    cmd = [sys.executable, "-u", str(code / "c50_s4_train.py"), "--arm", arm, "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "train")]
    ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True))
    assert len(ck) <= 1, ck
    assert ck or not need_ck, "a resumed session with no train/ckpt.pt mounted"
    if ck:
        cmd += ["--resume", ck[0]]
    if arm == "b":
        cmd += ["--bank", str(find("bank/bank_values.f16").parent)]
elif TASK.startswith("t1train-"):
    # t1train-<only>-<tau>[+resume] (C50-T1 part 2b): the anchored LoRA, resumed as the S4 trainings are.
    name = TASK[len("t1train-"):]
    need_ck = name.endswith("+resume")
    only, tau = name.removesuffix("+resume").split("-")
    cmd = [sys.executable, "-u", str(code / "c50_t1_train.py"), "--only", only, "--tau", tau, "--gguf", str(gguf),
           "--teacher", str(find("teacher_v1/manifest.jsonl").parent), "--corpus", str(find("teacher.tsv")),
           "--out", str(out / "train")]
    ck = sorted(glob.glob("/kaggle/input/**/train/ckpt.pt", recursive=True))
    assert len(ck) <= 1, ck
    assert ck or not need_ck, "a resumed session with no train/ckpt.pt mounted"
    if ck:
        cmd += ["--resume", ck[0]]
elif TASK.startswith("eval-"):
    # eval-<arm>[_<control>]: t0 (MVE-5), a, b, c48, c496, a_shuffle, b_remove, b_shuffle, b_mask.
    name = TASK[len("eval-"):]
    arm, _, ctrl = name.partition("_")
    cmd = [sys.executable, "-u", str(code / "c50_s4_eval.py"), "--gguf", str(gguf), "--inputs",
           str(code / "refs/inputs"), "--split", str(code / "refs/c50_s4_split.json"), "--arm", arm,
           "--out", str(out / "eval")]
    if arm != "t0":
        cmd += ["--weights", str(find("train/final.pt"))]
    if arm == "b":
        cmd += ["--bank", str(find("bank/bank_values.f16").parent), "--bank-mode", ctrl or "full"]
    if arm == "a":
        cmd += ["--a-mode", ctrl or "full"]
else:
    raise SystemExit(f"unknown TASK {TASK}")
print("CMD", json.dumps(cmd), flush=True)
subprocess.run(cmd, env=env, check=True)
print("DONE", TASK, flush=True)
