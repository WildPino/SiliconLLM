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
else:
    raise SystemExit(f"unknown TASK {TASK}")
print("CMD", json.dumps(cmd), flush=True)
subprocess.run(cmd, env=env, check=True)
print("DONE", TASK, flush=True)
