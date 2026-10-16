"""C50-A2b runner: four llama.cpp arms on the same source-tokenizer IDs, plus a repeat of R0.

Protocol: docs/research/CHATBOT50_ROADMAP_20261016/C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md
(and its addenda).  Records numbers only; c50_a2b_floor.py computes the floor from the
file this writes.  Dumps (16-17 GB per arm) go to --work outside the repository.

Usage:
    python -I benchmarks/chatbot50/c50_a2b_floor_run.py --model <q4_k_m.gguf> --work E:/.../a2b \
        --out benchmarks/chatbot50/results/c50_a2b_floor_run.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/Ling-mini-2.0_a810f64/tokenizer.json"
IDS_SHA256 = "cf76819b8b5ee1636feea8badd7ae4b52d08b1bcc5dffcd87d8704722eeaa460"
MAX_PROJECTED_S = 3600
CTX = 8192
# (arm, build, threads, ubatch, dump)
ARMS = [("R0", "avx2", 6, 512, True), ("R3", "avx2", 6, 1, True), ("R1", "avx2", 1, 512, True),
        ("R2", "generic", 6, 512, True), ("R0b", "avx2", 6, 512, False)]
PAIRS = [("R0", "R1"), ("R0", "R2"), ("R0", "R3"), ("R0", "R0b")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ids = work / "a2b_ids.tsv"
    subprocess.run([sys.executable, "-I", str(HERE / "c50_make_ids.py"), "--holdout",
                    str(HERE / "corpus" / "c50_holdout_v1"), "--role", "fit", "--per-category", "4",
                    "--out", str(ids)], check=True, capture_output=True)
    if sha256(ids) != IDS_SHA256:
        raise SystemExit("A2b position file differs from the registered SHA-256")
    model = Path(args.model)
    report = {"cell": "C50-A2b", "model": str(model), "model_sha256": sha256(model), "ids_sha256": IDS_SHA256,
              "exe_sha256": {b: sha256(HERE / "bin" / f"c50_llama_score_{b}.exe") for b in ("avx2", "generic")},
              "source_sha256": sha256(HERE / "c50_llama_score.cpp"), "arms": {}, "pairs": {},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for arm, build, threads, ubatch, dump in ARMS:
        cmd = [str(HERE / "bin" / f"c50_llama_score_{build}.exe"), "--model", str(model), "--ids", str(ids),
               "--out", str(work / arm), "--threads", str(threads), "--ubatch", str(ubatch), "--ctx", str(CTX),
               "--max-projected-s", str(MAX_PROJECTED_S)] + (["--dump-logprobs"] if dump else [])
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        config = [ln for ln in lines if ln.startswith("CONFIG ")]
        rec = {"cmd": cmd, "returncode": proc.returncode, "wall_s": time.time() - t0,
               "config": config, "stdout_tail": lines[-3:], "stderr_tail": proc.stderr[-600:]}
        report["arms"][arm] = rec
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(arm, "rc", proc.returncode, f"{rec['wall_s']:.0f}s", lines[-1] if lines else "")
        if proc.returncode == 3:
            rec["status"] = "STOPPED by the projected-time rule"
            continue
        if proc.returncode != 0:
            raise SystemExit(f"{arm} failed")
        c = config[0] if len(config) == 1 else ""
        simd = "AVX2 = 1" in c
        if (f"threads={threads} " not in c or f"ubatch={ubatch} " not in c or f"dump={int(dump)} " not in c
                or simd != (build == "avx2") or "flash_attn=disabled" not in c):
            raise SystemExit(f"{arm}: CONFIG does not confirm the registered arm: {c}")
        rec["status"] = "COMPLETE"
    for a, b in PAIRS:
        if report["arms"][a].get("status") != "COMPLETE" or report["arms"][b].get("status") != "COMPLETE":
            report["pairs"][f"{a}-{b}"] = {"status": "not computed: an arm did not complete"}
            continue
        out = work / f"compare_{a}_{b}.json"
        subprocess.run([sys.executable, "-I", str(HERE / "c50_compare.py"), "--ref", str(work / a), "--other",
                        str(work / b), "--ids", str(ids), "--tokenizer", TOKENIZER, "--json", str(out)], check=True)
        report["pairs"][f"{a}-{b}"] = json.loads(out.read_text(encoding="utf-8"))
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
