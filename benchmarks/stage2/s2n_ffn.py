"""S2N amendment 5, a test of mechanism (localization): does the FFN hold sx19's remaining recall?  CPU only.  Not a recipe candidate.

    sxf  sx19 (e2e_rowx's file, top slot exact from P0 in layers 1-19) with the FFN organs exact: the dense FFN of layer 0 (ffn_gate, ffn_up, ffn_down)
         and the shared expert of layers 1-19 (ffn_*_shexp), P0's values dequantized and held in BF16 (relative rounding 0.14%), in place of
         e2e_rowx's IQ2_KS-with-scales.  Attention stays IQ2_KS with e2e_rowx's scales; routed experts stay bal with theirs.
Panels and slot exactly as sx19 (s2n_exact.py's command): generation at k = 4 (mass 16, decode only), scoring at k = 8.

export   writes sxf.gguf from e2e_rowx.gguf, then reads every tensor back: the FFN organs against bf16(dequant(P0)), every other tensor against
         e2e_rowx's bytes.  A layout check: each patched tensor's cosine with e2e_rowx's own must be >= 0.8 (a transposed or shuffled tensor gives ~0).
verdict  numbers and labels, chosen by nothing else (paired bootstrap, stage 1's seed and draws):
    FFN on recall, sxf - sx19 on recall (mean of nq and tqa): THE FFN HOLDS RECALL (lower > 0) / THE FFN COSTS RECALL (upper < 0) /
    NOT SHOWN (the recall is not in the FFN: attention).  All panels reported beside.
    Reported: stage 1's label and chat gate on sxf, recall against top4dm16 (what k = 4 alone loses), desk ms (not a recipe row).
    VOID if the export does not read back, the model differs from its pin, a panel is not COMPLETE, or a CONFIG or MIX line does not confirm the slot.

Usage: python -I benchmarks/stage2/s2n_ffn.py export --json benchmarks/stage2/results/s2n_ffn_export.json
       python -I benchmarks/stage2/s2n_ffn.py verdict --json benchmarks/stage2/results/s2n_ffn_verdict.json
       python -I benchmarks/stage2/s2n_ffn.py selftest
       python -I benchmarks/stage2/s2n_ffn.py chain
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2n_exact as X  # noqa: E402
import s2n_export as E  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

gguf = E.gguf
BASE, BASE_SHA = X.MODEL, X.MODEL_SHA
OUT = Path("E:/SiliconLLM_Artifacts/stage2/s2n/q4/sxf.gguf")
FFN = ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
COS_MIN = 0.8
DESK_FFN = 3.92        # S2_DELIVERY_20261105.md, section 5: FFN organs stored
NAMES = ("ref", "top4dm16", "e2e_rowx", "sx19", "sxf")


def is_ffn(name: str) -> bool:
    p = name.split(".")
    return len(p) == 4 and p[0] == "blk" and p[2] in FFN


def want(t_p0, t_base) -> tuple[np.ndarray, float]:
    a = dequantize(t_p0.data, t_p0.tensor_type).astype(np.float32)
    b = E.from_bf16(np.asarray(t_base.data).view(np.uint16)).reshape(a.shape)
    cos = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return E.to_bf16(a).view(np.uint8).reshape(-1), cos


def export(out: Path, p0: Path = X.P0, plant: str | None = None) -> dict:
    shutil.copyfile(BASE, out)
    rd, pr = gguf.GGUFReader(str(BASE)), gguf.GGUFReader(str(p0))
    P = {t.name: t for t in pr.tensors}
    mm = np.memmap(out, mode="r+")
    n = 0
    for t in rd.tensors:
        if is_ffn(t.name):
            if t.tensor_type.name != "BF16":
                raise SystemExit(f"{t.name}: {t.tensor_type.name}, expected BF16")
            w, _ = want(P[t.name], t)
            if plant == t.name:
                w = w.reshape(-1, 2)[np.random.default_rng(0).permutation(len(w) // 2)].reshape(-1)
            o = int(t.data_offset)
            mm[o:o + int(t.n_bytes)] = w
            n += 1
    mm.flush()
    del mm
    return verify(out, p0) | {"exported": n}


def verify(out: Path, p0: Path = X.P0) -> dict:
    rd, pr = gguf.GGUFReader(str(BASE)), gguf.GGUFReader(str(p0))
    P = {t.name: t for t in pr.tensors}
    src, dst = np.memmap(BASE, mode="r"), np.memmap(out, mode="r")
    if src.shape != dst.shape:
        return {"ok": False, "bad": ["file size"]}
    first = min(int(t.data_offset) for t in rd.tensors)
    bad = [] if np.array_equal(src[:first], dst[:first]) else ["header"]
    cos, patched = {}, 0
    for t in rd.tensors:
        o, nb = int(t.data_offset), int(t.n_bytes)
        if is_ffn(t.name):
            w, c = want(P[t.name], t)
            got = E.from_bf16(np.asarray(dst[o:o + nb]).view(np.uint16)).reshape(-1)
            ref = E.from_bf16(np.asarray(src[o:o + nb]).view(np.uint16)).reshape(-1)
            cos[t.name] = float((got * ref).sum() / np.sqrt((got * got).sum() * (ref * ref).sum()))
            patched += 1
            if not np.array_equal(w, dst[o:o + nb]) or cos[t.name] < COS_MIN:
                bad.append(t.name)
        elif not np.array_equal(src[o:o + nb], dst[o:o + nb]):
            bad.append(t.name)
    return {"ok": not bad and patched == 1 * 3 + 19 * 3, "bad": bad[:20], "patched": patched, "tensors": len(rd.tensors),
            "cos_min": min(cos.values()) if cos else None, "cos_mean": float(np.mean(list(cos.values()))) if cos else None}


def judge(arms: dict, deg: dict, valid: bool) -> dict:
    J = X.JUDGED
    n = {p: len(arms["ref"][p]) for p in J}
    for x in NAMES:
        if {p: len(arms[x][p]) for p in J} != n:
            raise SystemExit(f"{x} does not score the same items as ref")

    def rec(x, idx=None):
        return float(np.mean([100.0 * (arms[x][p] if idx is None else arms[x][p][idx[p]]).mean() for p in X.RECALL]))
    rng = np.random.default_rng(X.SEED)
    b = {k: [] for k in ("rec", "all", "d")}
    for _ in range(X.DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in J}
        b["rec"].append(rec("sxf", idx) - rec("sx19", idx))
        b["all"].append(X.points(arms["sxf"], idx) - X.points(arms["sx19"], idx))
        b["d"].append(X.points(arms["sxf"], idx) - X.points(arms["ref"], idx))
    a = {x: X.points(arms[x]) for x in NAMES}
    r = {x: rec(x) for x in NAMES}
    chat = {x: int(arms[x]["chat"].sum()) for x in NAMES}
    c = X.ci(b["d"])
    chat_ok = chat["sxf"] >= chat["ref"] - X.CHAT_MARGIN and deg["sxf"] <= deg["ref"] + X.DEGEN_MARGIN
    ms = X.DESK_STACK + X.DESK_SLOT["sx19"] + DESK_FFN
    return {"valid": valid, "all_vs_ref": {x: a[x] - a["ref"] for x in NAMES[1:]}, "recall_vs_ref": {x: r[x] - r["ref"] for x in NAMES[1:]},
            "chat": chat, "degenerate": deg,
            "ffn": {"recall_gain": r["sxf"] - r["sx19"], "ci95": X.ci(b["rec"]), "all_gain": a["sxf"] - a["sx19"], "all_ci95": X.ci(b["all"]),
                    "label": X.three(X.ci(b["rec"]), "THE FFN HOLDS RECALL", "THE FFN COSTS RECALL", "NOT SHOWN") if valid else "VOID"},
            "sxf": {"delta": a["sxf"] - a["ref"], "ci95": c, "label": X.label(c) if valid else "VOID", "chat_ok": chat_ok,
                    "recall_vs_top4dm16": r["sxf"] - r["top4dm16"], "desk_ms_not_a_recipe_row": ms, "desk_tok_s": 1000.0 / ms}}


def verdict(results: Path) -> dict:
    arms = {x: {p: X.vec(json.loads((X.SCORES / f"{x}.json").read_text(encoding="utf-8")), p) for p in X.JUDGED} for x in NAMES}
    deg = {x: X.degenerate_count(X.WORK / x) for x in NAMES}
    ex = json.loads((results / "s2n_ffn_export.json").read_text(encoding="utf-8"))
    ru = json.loads((results / "s2n_exact_run.json").read_text(encoding="utf-8"))["arms"].get("sxf", {})
    p = ru.get("panels", {})
    valid = bool(ex["ok"] and ru.get("model_sha256") == ex["out_sha256"] and ru.get("layers") == "1-19"
                 and all(p.get(x[0], {}).get("status") == "COMPLETE" for x in X.PANELS)
                 and all("exact_layers=1-19 " in p[x[0]]["config"][0] for x in X.PANELS))
    return judge(arms, deg, valid)


def selftest() -> int:
    rng = np.random.default_rng(2)
    n = {"nq": 300, "tqa": 300, "gsm8k": 100, "arc": 300, "mmlu": 300, "chat": 40}
    base = {p: rng.random(k) < 0.5 for p, k in n.items()}

    def up(v, share, val):
        out = {}
        for p, x in v.items():
            x = x.copy()
            x[rng.random(len(x)) < share] = val
            out[p] = x
        return out
    arms = {x: {p: v.copy() for p, v in base.items()} for x in NAMES}
    arms["ref"] = up(base, 0.2, True)
    deg = {x: 0 for x in NAMES}
    cases = [(judge(dict(arms, sxf=up(base, 0.2, True)), deg, True)["ffn"]["label"], "THE FFN HOLDS RECALL"),
             (judge(arms, deg, True)["ffn"]["label"], "NOT SHOWN"),
             (judge(dict(arms, sxf=up(base, 0.2, False)), deg, True)["ffn"]["label"], "THE FFN COSTS RECALL"),
             (judge(arms, deg, False)["ffn"]["label"], "VOID"), (judge(arms, deg, False)["sxf"]["label"], "VOID"),
             (abs(judge(arms, deg, True)["sxf"]["desk_ms_not_a_recipe_row"] - 50.89) < 1e-9, True),
             (judge(arms, dict(deg, sxf=9), True)["sxf"]["chat_ok"], False)]
    tmp = OUT.with_name("sxf_selftest.gguf")
    v = export(tmp)
    cases.append((v["ok"] and v["exported"] == 60, True))
    v2 = export(tmp, plant="blk.7.ffn_up_shexp.weight")      # a shuffled tensor: wrong bytes and a cosine near 0
    cases.append((v2["ok"], False))
    cases.append(("blk.7.ffn_up_shexp.weight" in v2["bad"], True))
    tmp.unlink()
    bad = [i for i, (x, y) in enumerate(cases) if x != y]
    print("selftest", "PASS" if not bad else f"FAIL {bad} {[cases[i] for i in bad]}", len(cases))
    return 0 if not bad else 1


def chain() -> None:
    R = HERE / "results"
    js = R / "s2n_ffn_export.json"
    if not js.exists():
        X.step("benchmarks/stage2/s2n_ffn.py", "export", "--json", str(js))
    rec = json.loads(js.read_text(encoding="utf-8"))
    if not rec["ok"]:
        raise SystemExit("stopped: sxf's export does not read back (VOID)")
    if not (X.SCORES / "sxf.json").exists():
        X.step("benchmarks/stage2/s2n_ffn.py", "run", "--model-sha", rec["out_sha256"])
        X.step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", X.A4, "--s2-inputs", X.S2, "--arm-a4", str(X.WORK / "sxf"),
               "--arm-s2", str(X.WORK / "sxf"), "--out", str(X.SCORES / "sxf.json"))
    X.step("benchmarks/stage2/s2n_ffn.py", "verdict", "--json", str(R / "s2n_ffn_verdict.json"))
    print("S2N FFN DONE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "run", "verdict", "selftest", "chain"))
    ap.add_argument("--json", type=Path)
    ap.add_argument("--model-sha", default="")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "chain":
        chain()
        return 0
    if a.cmd == "export":
        if X.sha256(BASE) != BASE_SHA or X.sha256(X.P0) != X.P0_SHA:
            raise SystemExit("a model differs from its pin")
        rec = {"base": str(BASE), "base_sha256": BASE_SHA, "p0": str(X.P0), "p0_sha256": X.P0_SHA, **export(OUT)}
        rec.update(out=str(OUT), out_sha256=X.sha256(OUT))
        a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("EXPORT", "OK" if rec["ok"] else "VOID", json.dumps({k: v for k, v in rec.items() if k != "bad"}), flush=True)
        return 0 if rec["ok"] else 1
    if a.cmd == "run":                       # sx19's panels and slot on sxf's file, into s2n_exact_run.json
        X.MODEL, X.MODEL_SHA, X.ARMS = OUT, a.model_sha, {"sxf": "1-19"}
        return X.run("sxf", HERE / "results" / "s2n_exact_run.json")
    out = verdict(HERE / "results")
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
