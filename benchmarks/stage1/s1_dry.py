"""Stage 1 (a) dry run: the instrument checks registered before any arm runs.  Each check prints ok/FAIL; the run
PASSes only if all of them pass.

Scorer, on the first 2 documents of the panel (6,915 positions):
    1. the C50 scorer (c50_llama_score_avx2.exe, built before the C: reinstall) equals C50-A4's recorded P0 outputs
       for the same positions, bit for bit;
    2. s1 scorer without the callback equals the C50 scorer, bit for bit;
    3. s1 scorer with the callback in mode none equals it without, bit for bit;
    4. for k = 6, 4, 2: mode top equals llama.cpp's own top-k (--override-experts k), bit for bit;
    5. mode random at k = 4 differs from top at k = 4 (top-1 agreement with ref lower by >= 0.02): the control acts;
    6. every MOE line counts 8 slots x (18 layers x input tokens + the rows whose logits are requested, which the
       last layer alone runs on).  (Dry run 1 FAILed this check only: it expected 19 x input tokens.)
Generator, on the first 20 nq prompts (32 new tokens) and the first 5 gsm8k prompts (128 new tokens):
    7. without the callback equals the C50 generator, token for token;
    8. mode none equals without the callback;
    9. mode top k = 4 equals --override-experts 4 (prompt and every decode step);
   10. decode-only top k = 4 keeps 4.0 on decode tokens and 8 on prompt tokens, and its outputs differ from ref's;
   11. topw at k = 4 keeps 4 and differs from top at k = 4 (the biased and weight orders disagree).

Usage:
    python -I benchmarks/stage1/s1_dry.py --work E:/SiliconLLM_Artifacts/stage1/dry2 \
        --out benchmarks/stage1/results/s1a_dry.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_run import CAMPAIGN, HERE, MODEL, STOP, sha256  # noqa: E402

C50_BIN = HERE.parent / "chatbot50" / "bin"
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])


def first_lines(src: Path, n: int, dst: Path) -> Path:
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()][:n]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return dst


def run(cmd: list[str], log: dict, name: str) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    log[name] = {"cmd": [str(c) for c in cmd], "returncode": p.returncode,
                 "lines": [ln for ln in p.stdout.splitlines() if ln.startswith(("CONFIG", "MOE", "DONE"))]}
    if p.returncode != 0:
        raise SystemExit(f"{name} failed: {p.stderr[-400:]}")
    return p.stdout


def expected_slots(out: str, inputs: int) -> int:
    """llama.cpp runs the last layer only on the rows whose logits are requested: every position in the scorer, the
    last prompt token of each item and every generated token in the generator."""
    c = re.search(r"items=(\d+) prompt_tokens=(\d+)", out)
    rows = inputs if c is None else inputs - int(c.group(2)) + int(c.group(1))
    return 8 * (18 * inputs + rows)


def moe_ok(out: str) -> bool:
    m = re.search(r"n_expert_used=(\d+) layers=(\d+) input_tokens=(\d+) tensors=\d+ slots=(\d+)", out)
    return (bool(m) and int(m.group(1)) == 8 and int(m.group(2)) == 19
            and int(m.group(4)) == expected_slots(out, int(m.group(3))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = Path(args.work)
    w.mkdir(parents=True, exist_ok=True)
    docs = first_lines(CAMPAIGN / "a4" / "inputs" / "docs.tsv", 2, w / "docs2.tsv")
    nq = first_lines(CAMPAIGN / "s2" / "inputs" / "nq.tsv", 20, w / "nq20.tsv")
    gsm = first_lines(CAMPAIGN / "a4" / "inputs" / "gsm8k.tsv", 5, w / "gsm5.tsv")
    s1s, s1g = HERE / "bin" / "s1_llama_score_avx2.exe", HERE / "bin" / "s1_llama_generate_avx2.exe"
    c50s, c50g = C50_BIN / "c50_llama_score_avx2.exe", C50_BIN / "c50_llama_generate_avx2.exe"
    base = ["--model", MODEL[0], "--threads", "6", "--ubatch", "512"]
    log: dict = {"exe_sha256": {p.name: sha256(p) for p in (s1s, s1g, c50s, c50g)}}
    checks: dict[str, bool] = {}
    outs: dict[str, str] = {}

    sc = lambda name, exe, extra: run([exe] + base + ["--ids", docs, "--ctx", "8192", "--out", w / name] + extra, log, name)  # noqa: E731
    sc("c50", c50s, [])
    outs["nohook"] = sc("nohook", s1s, ["--no-hook"])
    outs["none"] = sc("none", s1s, ["--top", "32"])
    for k in (6, 4, 2):
        outs[f"native{k}"] = sc(f"native{k}", s1s, ["--no-hook", "--override-experts", str(k)])
        outs[f"top{k}"] = sc(f"top{k}", s1s, ["--moe-mode", "top", "--moe-k", str(k)])
    outs["rnd4"] = sc("rnd4", s1s, ["--moe-mode", "random", "--moe-k", "4", "--moe-seed", "20261030"])
    outs["topw4"] = sc("topw4", s1s, ["--moe-mode", "topw", "--moe-k", "4"])
    rb = lambda name: (w / f"{name}.pos.bin").read_bytes()  # noqa: E731
    a4 = (CAMPAIGN / "a4" / "p0" / "docs.pos.bin").read_bytes()[: len(rb("c50"))]
    checks["1 c50 scorer = recorded A4 P0"] = rb("c50") == a4
    checks["2 s1 no-hook = c50 scorer"] = rb("nohook") == rb("c50")
    checks["3 s1 none = no-hook"] = rb("none") == rb("nohook")
    for k in (6, 4, 2):
        checks[f"4 top{k} = native top-{k}"] = rb(f"top{k}") == rb(f"native{k}")
    r = {n: np.frombuffer(rb(n), dtype=REC) for n in ("none", "top4", "rnd4", "topw4")}
    agree = {n: float(np.mean(r[n]["top1"] == r["none"]["top1"])) for n in ("top4", "rnd4", "topw4")}
    checks["11 topw4 keeps 4 and differs from top4"] = ("mean_kept=4.0000" in outs["topw4"]
                                                         and rb("topw4") != rb("top4"))
    log["docs_top1_agreement"] = agree
    checks["5 random acts (top-1 agreement lower by >= 0.02)"] = agree["rnd4"] <= agree["top4"] - 0.02
    checks["6 scorer MOE counts"] = all(moe_ok(outs[n]) for n in outs if n not in ("nohook",) and "native" not in n)

    for tag, prompts, new in (("nq", nq, "32"), ("gsm", gsm, "128")):
        g = lambda name, exe, extra: run([exe] + base + ["--prompts", prompts, "--ctx", "4096", "--max-new", new,  # noqa: E731
                                                         "--stop", STOP, "--out", w / f"{tag}_{name}"] + extra, log, f"{tag}_{name}")
        g("c50", c50g, [])
        g("nohook", s1g, ["--no-hook"])
        o_none = g("none", s1g, [])
        g("native4", s1g, ["--no-hook", "--override-experts", "4"])
        o_top4 = g("top4", s1g, ["--moe-mode", "top", "--moe-k", "4"])
        o_dec = g("top4d", s1g, ["--moe-mode", "top", "--moe-k", "4", "--moe-decode-only"])
        jb = lambda name: (w / f"{tag}_{name}.jsonl").read_bytes()  # noqa: E731
        checks[f"7 {tag} s1 no-hook = c50 generator"] = jb("nohook") == jb("c50")
        checks[f"8 {tag} none = no-hook"] = jb("none") == jb("nohook")
        checks[f"9 {tag} top4 = native top-4 (decode path)"] = jb("top4") == jb("native4")
        m = re.search(r"mean_kept=([\d.]+) decode_slots=(\d+) decode_mean_kept=([\d.]+)", o_dec)
        checks[f"10 {tag} decode-only keeps 4 on decode, more overall, and acts"] = (
            bool(m) and abs(float(m.group(3)) - 4.0) < 1e-9 and int(m.group(2)) > 0 and float(m.group(1)) > 4.0
            and jb("top4d") != jb("none"))
        checks[f"6 {tag} generator MOE counts"] = moe_ok(o_none) and moe_ok(o_top4) and moe_ok(o_dec)

    for k, v in checks.items():
        print("dry", "ok  " if v else "FAIL", k)
    ok = all(checks.values())
    print("dry", "PASS" if ok else "FAIL")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"checks": checks, "pass": ok, **log}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
