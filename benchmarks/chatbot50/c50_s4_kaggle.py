"""C50-S4: the Kaggle side of the feasibility checks and of the S4 runs (T4 x2, one account).

Two private datasets and one kernel per task:
- `c50-s4-t0`: T0's GGUF alone (9.9 GB), uploaded once; the kernel checks its sha256.
- `c50-s4-code`: the S4 code, the pinned gguf-py, the reference files and a MANIFEST.json of sha256 per file, versioned
  when the code changes. The kernel prints the manifest, so each record names the code it ran.
- `c50-s4-block1`: S3 block 1 (494 shards, their manifest) and the corpus file, uploaded once.
- kernel `c50-s4-<name>`: `kaggle/c50_s4_kernel.py` with TASK set, on `NvidiaTeslaT4` (2 x T4); `--sources` adds
  other kernels' outputs (a previous training session to resume, the bank, a trained arm's weights).

Accounts go through `scripts/kaggle_ops.py` (per-account credentials, OAuth token hidden); `--acct` picks one, and each
account holds its own copy of the datasets (a private dataset is not readable from another account). Nothing here
retries.

    python -I benchmarks/chatbot50/c50_s4_kaggle.py stage t0|code|block1  # hard links, or the code and its manifest
    python -I benchmarks/chatbot50/c50_s4_kaggle.py upload t0|code|block1 [--new]  # --new creates, else a new version
    python -I benchmarks/chatbot50/c50_s4_kaggle.py push mve2
    python -I benchmarks/chatbot50/c50_s4_kaggle.py --acct acct2 push train-a-s2 --task train-a --sources giggio253/c50-s4-train-a-s1
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

ACCTS = {"acct1": "wildpino", "acct2": "giggio253", "acct3": "sirwildpino"}
ACCT, OWNER = "acct1", "wildpino"          # set from --acct
C = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016")
STAGE = C / "s4/kaggle"
T0_GGUF = C / "gguf/ling-mini-2.0-a810f64-q4_k_m.gguf"
GGUF_PY = C / "tools/llama.cpp-5b335f4/gguf-py/gguf"
DATASETS = {"t0": "c50-s4-t0", "code": "c50-s4-code", "block1": "c50-s4-block1"}
S3 = C / "s3"
CODE = ["c50_s4_dequant.py", "c50_s4_t0_torch.py", "c50_s4_t0_gpu.py", "c50_s4_student.py", "c50_s1_expand.py",
        "c50_s4_bank.py", "c50_s4_train.py", "c50_s4_eval.py", "c50_t1_train.py",   # C50-T1 part 2b: the anchored trainer
        "c50_t2_n.py", "c50_t2_n_eval.py",                                           # C50-T2 option N
        "c50_m_eval.py",                                                             # C50 cell M (the metric)
        "c50_inj_train.py", "c50_inj_eval.py"]                                       # C50 injection cell
REFS = {"p0.docs.pos.bin": C / "a4/p0/docs.pos.bin", "bf16.docs.pos.bin": C / "a4/bf16/docs.pos.bin",
        "docs.tsv": C / "a4/inputs/docs.tsv", "t0_dequant_fp32_sha256.json": C / "s4/t0_dequant_fp32_sha256.json",
        "c50_s4_split.json": HERE / "results/c50_s4_split.json"}
REFS.update({f"inputs/{n}.tsv": C / f"a4/inputs/{n}.tsv" for n in ("docs", "arc", "gsm8k", "chat")})
REFS.update({f"inputs/{n}.tsv": C / f"s2/inputs/{n}.tsv" for n in ("nq", "tqa", "mmlu")})
# C50-T2 option N: addendum 2's fresh items, and v0 as an extra dataset (mounted with --datasets OWNER/c50-t2-v0)
REFS.update({f"fresh/{n}": C / f"t1/fresh/inputs/{n}" for n in
             ("nq.tsv", "tqa.tsv", "mmlu.tsv", "arc.tsv", "manifest.json")})
# C50 injection cell: the inputs `c50_inj_data.py` writes (whichever exist when the code dataset is staged)
INJ = C / "inj"
REFS.update({f"inj/{n}": INJ / n for n in ("qa.tsv", "qa_new.tsv", "known.tsv", "known.jsonl", "manifest.json", "params.json",
                                           "split_typed.json", "train_E.jsonl", "train_S.jsonl", "train_QA1.jsonl",
                                           "train_dev_E.jsonl", "eval_rew.tsv", "eval_orig.tsv", "eval_cloze.tsv",
                                           "eval_open.tsv", "eval_open_rew.tsv")} if INJ.exists() else {})
REFS["a4bis_post/docs.tsv"] = C / "a4bis_post/inputs/docs.tsv"      # C50 cell M part 2: the 72 post-release documents
REFS.update({f"inj_smoke/{p.name}": p for p in sorted((C / "inj_smoke").glob("*")) if p.is_file()}
            if (C / "inj_smoke").exists() else {})
EXTRA = {"v0": ("c50-t2-v0", C / "t1/t1_a1_q4_k_m.gguf")}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def folder_of(which: str) -> Path:
    """Each account stages in its own folder (dataset id and upload caches differ)."""
    return STAGE / (which if OWNER == "wildpino" else f"{which}_{OWNER}")


def meta(folder: Path, slug: str, title: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def cmd_stage(a: argparse.Namespace) -> int:
    if a.which in EXTRA:
        slug, src = EXTRA[a.which]
        f = folder_of(a.which)
        f.mkdir(parents=True, exist_ok=True)
        link = f / src.name
        if not link.exists():
            os.link(src, link)
        meta(f, slug, slug)
        print(a.which, "->", link)
        return 0
    if a.which == "t0":
        t0 = folder_of("t0")
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
        blk = folder_of("block1")
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
    code = folder_of("code")
    if code.exists():
        shutil.rmtree(code)
    (code / "refs").mkdir(parents=True)
    for n in CODE:
        shutil.copy2(HERE / n, code / n)
    for n, src in REFS.items():
        if n.startswith("inj/") and not src.exists():
            continue
        (code / "refs" / n).parent.mkdir(parents=True, exist_ok=True)
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
    folder = folder_of(a.which)
    if a.new:
        args = ["datasets", "create", "-p", str(folder), "--dir-mode", "zip", "-q"]
    else:
        args = ["datasets", "version", "-p", str(folder), "--dir-mode", "zip", "-m", a.message, "-q"]
    r = kaggle_ops.kaggle(ACCT, *args, check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_verify(a: argparse.Namespace) -> int:
    """The account's current version of a dataset holds every file of the CANONICAL staging (the first account's folder),
    with its size; subfolders are zipped on upload and listed unzipped. Each account stages in its own folder, and on 28
    October two accounts uploaded folders staged days before (only the first account's had been restaged): a check
    against the account's own folder would have passed them."""
    folder = STAGE / a.which
    want = {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*")
            if p.is_file() and p.name != "dataset-metadata.json"}
    r = kaggle_ops.kaggle(ACCT, "datasets", "files", f"{OWNER}/{DATASETS.get(a.which) or EXTRA[a.which][0]}",
                          "--page-size", "1000", check=False)
    have = {}
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) >= 2 and f[1].isdigit():
            have[f[0]] = int(f[1])
    bad = sorted(k for k, s in want.items() if have.get(k) != s)
    print(f"{ACCT} {a.which}: {len(want)} staged, {len(have)} listed, {len(bad)} missing or of another size", bad[:20])
    return 0 if not bad else 1


def cmd_push(a: argparse.Namespace) -> int:
    k = STAGE / f"kernel_{OWNER}_{a.task}"
    task = a.run or a.task
    if k.exists():
        shutil.rmtree(k)
    k.mkdir(parents=True)
    src = (HERE / "kaggle/c50_s4_kernel.py").read_text(encoding="utf-8")
    assert src.count('TASK = "__TASK__"') == 1
    (k / "c50_s4_kernel.py").write_text(src.replace('TASK = "__TASK__"', f'TASK = "{task}"'), encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/c50-s4-{a.task}", "title": f"c50-s4-{a.task}", "code_file": "c50_s4_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{OWNER}/{d}" for d in DATASETS.values()] + [s for s in (a.datasets or "").split(",") if s],
        "kernel_sources": [s for s in (a.sources or "").split(",") if s]}, indent=1), encoding="utf-8")
    r = kaggle_ops.kaggle(ACCT, "kernels", "push", "-p", str(k), check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_status(a: argparse.Namespace) -> int:
    r = kaggle_ops.kaggle(ACCT, "kernels", "status", f"{OWNER}/c50-s4-{a.task}", check=False)
    print(r.stdout, r.stderr, sep="\n")
    return r.returncode


def cmd_output(a: argparse.Namespace) -> int:
    out = C / f"s4/kaggle_out/{a.task}" if OWNER == "wildpino" else C / f"s4/kaggle_out/{OWNER}_{a.task}"
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
    st.add_argument("which", choices=list(DATASETS) + list(EXTRA))
    st.set_defaults(f=cmd_stage)
    u = sub.add_parser("upload")
    u.add_argument("which", choices=list(DATASETS) + list(EXTRA))
    u.add_argument("--new", action="store_true")
    u.add_argument("-m", "--message", default="new version")
    u.set_defaults(f=cmd_upload)
    v = sub.add_parser("verify")
    v.add_argument("which", choices=list(DATASETS) + list(EXTRA))
    v.set_defaults(f=cmd_verify)
    for n, f in (("push", cmd_push), ("status", cmd_status), ("output", cmd_output)):
        p = sub.add_parser(n)
        p.add_argument("task", help="the kernel's name after c50-s4-")
        if n == "push":
            p.add_argument("--task", dest="run", help="the kernel's TASK, when it differs from the name")
            p.add_argument("--sources", help="comma-separated kernels whose outputs are mounted")
            p.add_argument("--datasets", help="comma-separated extra datasets to mount (a checkpoint moved to this account)")
        p.set_defaults(f=f)
    ap.add_argument("--acct", default="acct1", choices=list(ACCTS))
    a = ap.parse_args()
    ACCT, OWNER = a.acct, ACCTS[a.acct]
    raise SystemExit(a.f(a))
