"""Stage 1 (a) part 1b dry run (the mass-decoupled k = 4): the instrument checks registered before any arm runs.  Each
check prints ok/FAIL; the run PASSes only if all of them pass.  The part-1 binaries are kept in bin/s1a/ (their sha256
are in results/s1a_run.json).

Scorer, on the first 2 documents of the panel:
    1. new binary, mass off, modes none, top k = 4 and random k = 4: equal to the part-1 binary, bit for bit;
    2. mass 8 in mode none equals mode none, bit for bit (the factor is 1 and nothing is dropped);
    3. mass 8 at top k = 4 differs from top k = 4 (the hook acts after the norm) and its mean factor is 1;
    4. mass 16 at top k = 4: mean factor below 1, and differs from mass 8;
    5. every MASS line counts one row per MoE layer per token (slots / 8), and every MOE line counts as in part 1.
Generator, on the first 20 nq prompts (32 new tokens) and the first 5 gsm8k prompts (128 new tokens):
    6. new binary, mass off, modes none and top k = 4: equal to the part-1 binary, token for token;
    7. mass 8 in mode none equals mode none;
    8. mass 16 at top k = 4 runs the decode path (MASS rows = slots / 8) and differs from top k = 4;
    9. mass 16 decode-only keeps 4 on decode tokens and 8 on the prompt, and differs from part 1's decode-only top4;
   10. mass 16 decode-only leaves the prompt untouched: its first generated token equals mode none's on every item.
Every mass row is also checked inside the hook (it sums to 1 and equals probs[id] / S); a failure stops the tool.

Usage:
    python -I benchmarks/stage1/s1m_dry.py --work E:/SiliconLLM_Artifacts/stage1/drym \
        --out benchmarks/stage1/results/s1m_dry.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_dry import REC, first_lines, moe_ok, run  # noqa: E402
from s1_run import CAMPAIGN, HERE, MODEL, STOP, sha256  # noqa: E402


def mass(out: str) -> tuple[int, float] | None:
    m = re.search(r"MASS mass_k=\d+ rows=(\d+) mean_factor=([\d.]+)", out)
    return (int(m.group(1)), float(m.group(2))) if m else None


def rows_ok(out: str) -> bool:
    m, s = mass(out), re.search(r" slots=(\d+) ", out)
    return bool(m and s) and m[0] * 8 == int(s.group(1))


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
    news, newg = HERE / "bin" / "s1_llama_score_avx2.exe", HERE / "bin" / "s1_llama_generate_avx2.exe"
    olds, oldg = HERE / "bin" / "s1a" / "s1_llama_score_avx2.exe", HERE / "bin" / "s1a" / "s1_llama_generate_avx2.exe"
    base = ["--model", MODEL[0], "--threads", "6", "--ubatch", "512"]
    log: dict = {"exe_sha256": {str(p.relative_to(HERE)): sha256(p) for p in (news, newg, olds, oldg)}}
    checks: dict[str, bool] = {}
    outs: dict[str, str] = {}
    top4, rnd4 = ["--moe-mode", "top", "--moe-k", "4"], ["--moe-mode", "random", "--moe-k", "4", "--moe-seed", "20261030"]

    sc = lambda name, exe, extra: run([exe] + base + ["--ids", docs, "--ctx", "8192", "--out", w / name] + extra, log, name)  # noqa: E731
    for name, extra in (("none", []), ("top4", top4), ("rnd4", rnd4)):
        sc(f"old_{name}", olds, extra)
        outs[name] = sc(name, news, extra)
    outs["m8_none"] = sc("m8_none", news, ["--moe-mass", "8"])
    outs["m8_top4"] = sc("m8_top4", news, top4 + ["--moe-mass", "8"])
    outs["m16_top4"] = sc("m16_top4", news, top4 + ["--moe-mass", "16"])
    rb = lambda name: (w / f"{name}.pos.bin").read_bytes()  # noqa: E731
    checks["1 scorer, mass off: none, top4, rnd4 = part-1 binary"] = all(rb(n) == rb(f"old_{n}") for n in ("none", "top4", "rnd4"))
    checks["2 mass 8, none = none"] = rb("m8_none") == rb("none")
    m8, m16 = mass(outs["m8_top4"]), mass(outs["m16_top4"])
    checks["3 mass 8, top4 acts after the norm, factor 1"] = rb("m8_top4") != rb("top4") and bool(m8) and m8[1] == 1.0
    checks["4 mass 16, top4: factor < 1, differs from mass 8"] = bool(m16) and m16[1] < 1.0 and rb("m16_top4") != rb("m8_top4")
    checks["5 scorer MASS rows and MOE counts"] = (all(rows_ok(outs[n]) for n in ("m8_none", "m8_top4", "m16_top4"))
                                                   and all(moe_ok(o) for o in outs.values()))
    r = {n: np.frombuffer(rb(n), dtype=REC) for n in ("none", "top4", "m8_top4", "m16_top4")}
    log["docs_top1_agreement"] = {n: float(np.mean(r[n]["top1"] == r["none"]["top1"])) for n in ("top4", "m8_top4", "m16_top4")}
    log["mean_factor"] = {"m8_top4": m8[1] if m8 else None, "m16_top4": m16[1] if m16 else None}

    for tag, prompts, new in (("nq", nq, "32"), ("gsm", gsm, "128")):
        g = lambda name, exe, extra: run([exe] + base + ["--prompts", prompts, "--ctx", "4096", "--max-new", new,  # noqa: E731
                                                         "--stop", STOP, "--out", w / f"{tag}_{name}"] + extra, log, f"{tag}_{name}")
        go = {}
        for name, extra in (("none", []), ("top4", top4)):
            g(f"old_{name}", oldg, extra)
            go[name] = g(name, newg, extra)
        go["m8_none"] = g("m8_none", newg, ["--moe-mass", "8"])
        go["m16_top4"] = g("m16_top4", newg, top4 + ["--moe-mass", "16"])
        go["top4d"] = g("top4d", newg, top4 + ["--moe-decode-only"])
        go["m16_top4d"] = g("m16_top4d", newg, top4 + ["--moe-mass", "16", "--moe-decode-only"])
        jb = lambda name: (w / f"{tag}_{name}.jsonl").read_bytes()  # noqa: E731
        md = re.search(r"mean_kept=([\d.]+) decode_slots=(\d+) decode_mean_kept=([\d.]+)", go["m16_top4d"])
        checks[f"9 {tag} mass 16 decode-only: 4 on decode, more overall, differs from top4d"] = (
            bool(md) and abs(float(md.group(3)) - 4.0) < 1e-9 and int(md.group(2)) > 0 and float(md.group(1)) > 4.0
            and rows_ok(go["m16_top4d"]) and jb("m16_top4d") != jb("top4d"))
        first = lambda name: [json.loads(ln)["gen"][:1] for ln in jb(name).decode("utf-8").splitlines() if ln.strip()]  # noqa: E731
        checks[f"10 {tag} mass 16 decode-only: the prompt is untouched (first token = none's on every item)"] = (
            first("m16_top4d") == first("none"))
        checks[f"6 {tag} generator, mass off: none, top4 = part-1 binary"] = all(jb(n) == jb(f"old_{n}") for n in ("none", "top4"))
        checks[f"7 {tag} mass 8, none = none"] = jb("m8_none") == jb("none")
        checks[f"8 {tag} mass 16, top4: decode path, differs from top4"] = (rows_ok(go["m16_top4"]) and rows_ok(go["m8_none"])
                                                                              and jb("m16_top4") != jb("top4")
                                                                              and all(moe_ok(o) for o in go.values()))

    for k, v in checks.items():
        print("dry", "ok  " if v else "FAIL", k)
    ok = all(checks.values())
    print("dry", "PASS" if ok else "FAIL", json.dumps(log["docs_top1_agreement"]), json.dumps(log["mean_factor"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"checks": checks, "pass": ok, **log}, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
