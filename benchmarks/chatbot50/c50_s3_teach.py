"""C50-S3 teacher run: F0's top-32 log-probabilities on the S3 corpus, resumable and in blocks.  Start it, stop it
with Ctrl+C, start the same command again: it resumes from the last finished shard (at most the shard in progress is
lost).  It stops by itself once the finished shards hold --until positions (default: block 1, 10 M).

Every start checks the pins (binary, corpus, model) and refuses to run otherwise, runs the teacher tool with the
protocol's arguments, asserts its CONFIG line, and appends one line per start to <out>/launches.jsonl (time, binary,
environment, arguments, exit code).  The first start hashes the 62 GB model (~4 min); later starts check its size and
modification time against that record.

The teacher is the protocol's GPU configuration (fp32 cuBLAS, ngl 99 with the routed experts on the CPU, ctx and
ubatch 2048).  It failed the fidelity rule on 20 October (addendum 1); its use was then decided explicitly, in blocks,
by addendum 2.  `--fidelity` reruns that check, for the record.

Usage (from a terminal outside the agent session; the machine can be used, nothing here reads a rate):
    python -I benchmarks/chatbot50/c50_s3_teach.py            # start or resume block 1 (stops at 10 M positions)
    python -I benchmarks/chatbot50/c50_s3_teach.py --until 49892578   # a later block, only once it is decided
    python -I benchmarks/chatbot50/c50_s3_teach.py --status   # progress only
    python -I benchmarks/chatbot50/c50_s3_teach.py --fidelity # the fidelity check of 20 October (FAIL), again
"""
from __future__ import annotations

import argparse
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
MODEL = Path("C:/SiliconLLM_stage/f0/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf")
MODEL_SHA, MODEL_BYTES = "5348c73fe6497c5152e16fd792d6c7b5aeefa15de8b15b8c8e18bef691704b8e", 62_425_484_640
CORPUS_MANIFEST = HERE / "results" / "c50_s3_corpus_v1_manifest.json"
CORPUS = C / "s3/corpus_v1/teacher.tsv"
OUT = C / "s3/teacher_v1"
FID_IDS = C / "s2/inputs/mmlu.tsv"
FID_OUT = C / "s3/fidelity_v1"
ENV = {"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f32"}
ARGS = ["--ctx", "2048", "--ubatch", "2048", "--topk", "32", "--threads", "6", "--ngl", "99", "--cpu-moe"]
SHARD_DOCS = "48"
WANT = ["topk=32", "threads=6", "ubatch=2048", "ctx=2048", "ngl=99", "cpu_moe=1", "flash_attn=disabled", "kv=f16",
        "devices=[CUDA0,CPU]"]
BLOCK_1 = 10_000_000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def check_model(out: Path) -> None:
    st = MODEL.stat()
    if st.st_size != MODEL_BYTES:
        raise SystemExit(f"{MODEL}: {st.st_size} bytes, pin {MODEL_BYTES}")
    rec = out / "model_verified.json"
    if rec.exists():
        r = json.loads(rec.read_text(encoding="utf-8"))
        if r["size"] == st.st_size and r["mtime_ns"] == st.st_mtime_ns and r["sha256"] == MODEL_SHA:
            return
    print(time.strftime("%H:%M:%S"), "hashing the model once (~4 min)", flush=True)
    got = sha256(MODEL)
    if got != MODEL_SHA:
        raise SystemExit(f"{MODEL}: sha256 {got} != pin {MODEL_SHA}")
    rec.write_text(json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": got}) + "\n", encoding="utf-8")


def done_positions(out: Path) -> tuple[int, int]:
    m = out / "manifest.jsonl"
    rows = [json.loads(x) for x in m.read_text(encoding="utf-8").splitlines()] if m.exists() else []
    return len(rows), sum(r.get("positions", 0) for r in rows)


def progress(out: Path) -> str:
    shards, pos = done_positions(out)
    man = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    return (f"{shards} shards done, {pos:,} of {man['positions']:,} positions "
            f"({100 * pos / man['positions']:.1f}%)")


def run(ids: Path, out: Path, shard_docs: str, until: int | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    if until is not None and done_positions(out)[1] >= until:
        print(f"block done: {done_positions(out)[1]:,} positions >= --until {until:,}; nothing to run", flush=True)
        return 0
    got = sha256(BIN)
    if got != BIN_SHA:
        raise SystemExit(f"{BIN}: sha256 {got} != pin {BIN_SHA}")
    check_model(out)
    cmd = [str(BIN), "--model", str(MODEL), "--ids", str(ids), "--out", str(out), "--shard-docs", shard_docs] + ARGS
    rec = {"start": time.strftime("%Y-%m-%dT%H:%M:%S"), "bin_sha256": got, "env": ENV, "cmd": cmd, "until": until}
    print("CMD", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                         env={**{k: v for k, v in os.environ.items() if not k.startswith("GGML_")}, **ENV})
    rc = None
    try:
        with open(out / "teach.log", "a", encoding="utf-8") as log:
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
                if until is not None and line.startswith("SHARD") and done_positions(out)[1] >= until:
                    p.terminate()
                    p.wait()
                    rc = "block_done"
                    print(f"block done: {done_positions(out)[1]:,} positions >= --until {until:,}", flush=True)
                    break
        if rc is None:
            rc = p.wait()
    except KeyboardInterrupt:
        p.wait()
        rc = "stopped"
        print("\nstopped; run the same command again to resume", flush=True)
    finally:
        rec["end"], rec["exit"] = time.strftime("%Y-%m-%dT%H:%M:%S"), rc if rc is not None else p.poll()
        with open(out / "launches.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return 0 if rc in (0, "stopped", "block_done") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--until", type=int, default=BLOCK_1, help="stop once the finished shards hold this many positions")
    a = ap.parse_args()
    if a.status:
        print(progress(OUT))
        return 0
    if a.fidelity:
        return run(FID_IDS, FID_OUT, "120")
    man = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if sha256(CORPUS) != man["teacher_tsv_sha256"]:
        raise SystemExit(f"{CORPUS}: sha256 differs from the corpus manifest")
    rc = run(CORPUS, OUT, SHARD_DOCS, a.until)
    print(progress(OUT))
    return rc


if __name__ == "__main__":
    sys.exit(main())
