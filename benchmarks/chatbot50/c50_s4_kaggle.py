"""C50-S4: the Kaggle side of the feasibility checks and of the S4 runs (T4 x2, one account).

Two private datasets and one kernel per task:
- `c50-s4-t0`: T0's GGUF alone (9.9 GB), uploaded once; the kernel checks its sha256.
- `c50-s4-code`: the S4 code, the pinned gguf-py, the reference files and a MANIFEST.json of sha256 per file, versioned
  when the code changes. The kernel prints the manifest, so each record names the code it ran.
- `c50-s4-block1`: S3 block 1 (494 shards, their manifest) and the corpus file, uploaded once.
- kernel `c50-s4-<task>`: `kaggle/c50_s4_kernel.py` with TASK set, on `NvidiaTeslaT4` (2 x T4).

Accounts go through `scripts/kaggle_ops.py` (per-account credentials, OAuth token hidden). Nothing here retries.

    python -I benchmarks/chatbot50/c50_s4_kaggle.py stage t0|code|block1  # hard links, or the code and its manifest
    python -I benchmarks/chatbot50/c50_s4_kaggle.py upload t0|code|block1 [--new]  # --new creates, else a new version
    python -I benchmarks/chatbot50/c50_s4_kaggle.py push mve2
    python -I benchmarks/chatbot50/c50_s4_kaggle.py status mve2
    python -I benchmarks/chatbot50/c50_s4_kaggle.py output mve2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import kaggle_ops  # noqa: E402

ACCT, OWNER = "acct1", "wildpino"
C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
STAGE = C / "s4/kaggle"
T0_GGUF = C / "gguf/ling-mini-2.0-a810f64-q4_k_m.gguf"
GGUF_PY = C / "tools/llama.cpp-5b335f4/gguf-py/gguf"
DATASETS = {"t0": "c50-s4-t0", "code": "c50-s4-code", "block1": "c50-s4-block1"}
S3 = C / "s3"
CODE = ["c50_s4_dequant.py", "c50_s4_t0_torch.py", "c50_s4_t0_gpu.py", "c50_s4_student.py", "c50_s1_expand.py",
        "c50_s4_bank.py"]
REFS = {"p0.docs.pos.bin": C / "a4/p0/docs.pos.bin", "bf16.docs.pos.bin": C / "a4/bf16/docs.pos.bin",
        "docs.tsv": C / "a4/inputs/docs.tsv", "t0_dequant_fp32_sha256.json": C / "s4/t0_dequant_fp32_sha256.json"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def meta(folder: Path, slug: str, title: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def cmd_stage(a: argparse.Namespace) -> int:
    if a.which == "t0":
        t0 = STAGE / "t0"
        t0.mkdir(parents=True, exist_ok=True)
        link = t0 / T0_GGUF.name
        if not link.exists():
            os.link(T0_GGUF, link)   # a hard link on the same volume: no 9.9 GB copy
        meta(t0, DATASETS["t0"], DATASETS["t0"])
        print("t0 ->", link)
        return 0
    if a.which == "block1":
        # The shards go in a subfolder, so that `--dir-mode zip` sends them as one file (496 separate uploads failed
        # on 22 October with a JSON error from the CLI); Kaggle extracts it on its side.
        blk = STAGE / "block1"
        if blk.exists():
            shutil.rmtree(blk)
        (blk / "teacher_v1").mkdir(parents=True)
        srcs = sorted((S3 / "teacher_v1").glob("shard_*.bin")) + [S3 / "teacher_v1/manifest.jsonl"]
        assert len(srcs) == 495, len(srcs)
        for src in srcs:
            os.link(src, blk / "teacher_v1" / src.name)
        os.link(S3 / "corpus_v1/teacher.tsv", blk / "teacher.tsv")
        srcs.append(S3 / "corpus_v1/teacher.tsv")
        meta(blk, DATASETS["block1"], DATASETS["block1"])
        print("block1:", len(srcs), "files,", round(sum(x.stat().st_size for x in srcs) / 2**30, 2), "GiB")
        return 0
    code = STAGE / "code"
    if code.exists():
        shutil.rmtree(code)
    (code / "refs").mkdir(parents=True)
    for n in CODE:
        shutil.copy2(HERE / n, code / n)
    for n, src in REFS.items():
        shutil.copy2(src, code / "refs" / n)
    shutil.copytree(GGUF_PY, code / "gguf-py/gguf", ignore=shutil.ignore_patterns("__pycache__", "scripts"))
    files = sorted(p for p in code.rglob("*") if p.is_file())
    man = {str(p.relative_to(code)).replace(os.sep, "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    (code / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    meta(code, DATASETS["code"], DATASETS["code"])
    for n in CODE:
        print(f"{n:24s} {man[n]['sha256'][:16]}")
    print(f"staged {len(man)} files in {code}")
    return 0


def cmd_upload(a: argparse.Namespace) -> int:
    folder = STAGE / a.which
    if a.new:
        args = ["datasets", "create", "-p", str(folder), "--dir-mode", "zip", "-q"]
    else:
        args = ["datasets", "version", "-p", str(folder), "--dir-mode", "zip", "-m", a.message, "-q"]
    r = kaggle_ops.kaggle(ACCT, *args, check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_push(a: argparse.Namespace) -> int:
    k = STAGE / f"kernel_{a.task}"
    if k.exists():
        shutil.rmtree(k)
    k.mkdir(parents=True)
    src = (HERE / "kaggle/c50_s4_kernel.py").read_text(encoding="utf-8")
    assert src.count('TASK = "__TASK__"') == 1
    (k / "c50_s4_kernel.py").write_text(src.replace('TASK = "__TASK__"', f'TASK = "{a.task}"'), encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/c50-s4-{a.task}", "title": f"c50-s4-{a.task}", "code_file": "c50_s4_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{OWNER}/{d}" for d in DATASETS.values()]}, indent=1), encoding="utf-8")
    r = kaggle_ops.kaggle(ACCT, "kernels", "push", "-p", str(k), check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_status(a: argparse.Namespace) -> int:
    r = kaggle_ops.kaggle(ACCT, "kernels", "status", f"{OWNER}/c50-s4-{a.task}", check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_output(a: argparse.Namespace) -> int:
    out = C / f"s4/kaggle_out/{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    r = kaggle_ops.kaggle(ACCT, "kernels", "output", f"{OWNER}/c50-s4-{a.task}", "-p", str(out), check=False)
    print(r.stdout, r.stderr, sep="\n")
    print("->", out)
    return r.returncode


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage")
    st.add_argument("which", choices=list(DATASETS))
    st.set_defaults(f=cmd_stage)
    u = sub.add_parser("upload")
    u.add_argument("which", choices=list(DATASETS))
    u.add_argument("--new", action="store_true")
    u.add_argument("-m", "--message", default="new version")
    u.set_defaults(f=cmd_upload)
    for n, f in (("push", cmd_push), ("status", cmd_status), ("output", cmd_output)):
        p = sub.add_parser(n)
        p.add_argument("task")
        p.set_defaults(f=f)
    a = ap.parse_args()
    raise SystemExit(a.f(a))
