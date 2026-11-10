"""S2N's teacher: S3's teacher tool and arguments (top 32 log-probabilities, cuBLAS in fp32, context and ubatch 2,048, --ngl 99 --cpu-moe), run on mini's
Q4_K_M (P0, the panels' ref) over the first documents of S3's corpus v1 (the corpus flash's teacher F0 reads), until 1.6 M positions: flash's form of the
teacher signal at mini's scale.  1.5 M train, the rest held out, split by document.  Resumable as S3's run.

Usage: python -I benchmarks/stage2/s2n_teach.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
BIN = C / "tools/llama-b11040-win-cuda-12.4-x64/bin/c50_s3_teacher_cuda.exe"
BIN_SHA = "8a079d53118de71bae57d4144accb2925a38535c6d420730f2d6f8d5ddb906f1"
P0 = Path("C:/SiliconLLM_fast/s2t/m/p0.gguf")
P0_SHA = "44de73157f11165c0551fd06714f4f043913292a79b07683e6a34b11d3b0d2f4"
CORPUS = C / "s3/corpus_v1/teacher.tsv"
CORPUS_SHA = "b748733a0da0f631b26f428c220e48c116cafa0a8073d484041195fd233e6ac8"
OUT = Path("E:/SiliconLLM_Artifacts/stage2/s2n/teacher")
POSITIONS = 1_600_000
ENV = {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"}
# the tool was built with WinLibs (POSIX threads, MSVCRT; c50_build_teacher_gpu.sh) and loads its libwinpthread-1.dll from there
WINLIBS = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
ARGS = ["--ctx", "2048", "--ubatch", "2048", "--topk", "32", "--threads", "6", "--ngl", "99", "--cpu-moe"]
WANT = ["topk=32", "threads=6", "ubatch=2048", "ctx=2048", "ngl=99", "cpu_moe=1", "flash_attn=disabled", "kv=f16", "devices=[CUDA0,CPU]"]
SHARD_DOCS = "48"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = OUT / "ids.tsv"
    if not ids.exists():
        if sha256(CORPUS) != CORPUS_SHA:
            raise SystemExit("S3's corpus differs from its manifest")
        n, docs, lines = 0, 0, []
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                lines.append(line)
                n += len(line.rstrip("\n").split("\t", 1)[1].split(",")) - 1
                docs += 1
                if n >= POSITIONS:
                    break
        ids.write_text("".join(lines), encoding="utf-8")
        (OUT / "ids.json").write_text(json.dumps({"docs": docs, "positions": n, "corpus_sha256": CORPUS_SHA}) + "\n", encoding="utf-8")
    for p, want in ((BIN, BIN_SHA), (P0, P0_SHA)):
        if sha256(p) != want:
            raise SystemExit(f"{p}: sha256 differs from its pin")
    cmd = [str(BIN), "--model", str(P0), "--ids", str(ids), "--out", str(OUT), "--shard-docs", SHARD_DOCS] + ARGS
    rec = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": ENV, "cmd": cmd}
    print("CMD", " ".join(cmd), flush=True)
    if not (WINLIBS / "libwinpthread-1.dll").exists():
        raise SystemExit(f"libwinpthread-1.dll not in {WINLIBS}")
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GGML_")}, **ENV}
    env["PATH"] = f"{WINLIBS};{env.get('PATH', '')}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)
    with open(OUT / "teach.log", "a", encoding="utf-8") as log:
        for line in p.stdout:
            log.write(line)
            log.flush()
            if line.startswith("CONFIG"):
                rec["config"] = line.strip()
                missing = [w for w in WANT if f" {w} " not in f" {line.strip()} "]
                if missing:
                    p.kill()
                    raise SystemExit(f"CONFIG does not confirm {missing}: {line.strip()}")
            if line.startswith(("CONFIG", "SHARD", "DONE")) or "error" in line.lower():
                print(line.rstrip(), flush=True)
    rc = p.wait()
    rec.update(end=time.strftime("%Y-%m-%dT%H:%M:%S"), exit=rc)
    with open(OUT / "launches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("TEACH DONE" if rc == 0 else f"TEACH FAILED {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
