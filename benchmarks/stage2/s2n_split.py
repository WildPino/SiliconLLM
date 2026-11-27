"""S2N amendment 3: e2e_rowx split into its parts, CPU only.  Which part of the trained stack still holds the recall loss (nq, tqa)?

    rowx_org  e2e_rowx's organ scales on ks_org's file (organs IQ2_KS held in BF16, experts exact from P0): base ks_org4 (the same file at k = 4).
    rowx_exp  e2e_rowx's expert scales on bal-q4's file (bal experts in Q4_0, organs exact from P0): base bal4 (the same file at k = 4).
Both generate as ks_stk4 and e2e_rowx (k = 4, mass 16, decode only) through s2n_run.py; arc and mmlu score at k = 8.

export   patches only the part's tensors (s2n_export.patched: BF16 rows times the row scale; Q4_0 d times the row scale), reads every tensor back
         against its intended bytes, and every other tensor against the base's bytes.
verdict  numbers and labels, chosen by nothing else:
    Part's recall, rowx_P - base_P on recall (mean of nq and tqa), paired: GAINS (lower > 0) / LOSES (upper < 0) / NOT SHOWN.
    Who holds it, rowx_org - rowx_exp on recall: EXPERTS HOLD THE RECALL (lower > 0), ORGANS HOLD THE RECALL (upper < 0), NOT SEPARATED.
    Interaction, recall(e2e_rowx) - [recall(rowx_org) + recall(rowx_exp) - recall(top4dm16)]: THE STACK LOSES MORE (upper < 0),
    THE STACK LOSES LESS (lower > 0), ADDS UP.  top4dm16 counts k = 4 once.
    VOID if a part's export does not read back or its panels are incomplete on the exported file.

Usage: python -I benchmarks/stage2/s2n_split.py export --arm rowx_org|rowx_exp --json F
       python -I benchmarks/stage2/s2n_split.py verdict --scores-dir D --json F
       python -I benchmarks/stage2/s2n_split.py selftest
       python -I benchmarks/stage2/s2n_split.py chain        (both exports, then each part's seven panels and scores, then the verdict; resumable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stage1"))
import s2n_export as E  # noqa: E402
from s1_verdict import DRAWS, PANELS, SEED, points, vec  # noqa: E402

gguf = E.gguf
SCALES = Path("E:/SiliconLLM_Artifacts/stage2/s2n/train/e2e_rowx/scales.pt")
OUT = Path("E:/SiliconLLM_Artifacts/stage2/s2n/q4")
BASES = {"rowx_org": (Path("E:/SiliconLLM_Artifacts/stage2/s2y/q4/ks_org.gguf"), E.ORGANS, "ks_org4"),
         "rowx_exp": (Path("E:/SiliconLLM_Artifacts/stage2/s2v/bal-q4.gguf"), E.EXPERTS, "bal4")}
RECALL = ("nq", "tqa")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def targets(rd, scales: dict, names: tuple):
    for t in rd.tensors:
        p = t.name.split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in names:
            want = {"BF16"} if p[2] in E.ORGANS else {"Q4_0"}
            if t.tensor_type.name not in want:
                raise SystemExit(f"{t.name}: {t.tensor_type.name}, the patch expects {want}")
            yield t, E.multiplier(scales, "row", int(p[1]), p[2])


def export(arm: str, scales: dict, out: Path) -> dict:
    base, names, _ = BASES[arm]
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base, out)
    rd = gguf.GGUFReader(str(base))
    src, mm = np.memmap(base, mode="r"), np.memmap(out, mode="r+")
    n = 0
    for t, s in targets(rd, scales, names):
        o = int(t.data_offset)
        mm[o:o + int(t.n_bytes)] = E.patched(t, s, np.asarray(src[o:o + int(t.n_bytes)]))
        n += 1
    mm.flush()
    del mm
    return verify(arm, scales, out) | {"exported": n}


def verify(arm: str, scales: dict, out: Path) -> dict:
    base, names, _ = BASES[arm]
    rd = gguf.GGUFReader(str(base))
    src, dst = np.memmap(base, mode="r"), np.memmap(out, mode="r")
    if src.shape != dst.shape:
        return {"ok": False, "bad": ["file size"]}
    want = {t.name: s for t, s in targets(rd, scales, names)}
    first = min(int(t.data_offset) for t in rd.tensors)
    bad = [] if np.array_equal(src[:first], dst[:first]) else ["header"]
    for t in rd.tensors:
        o, nb = int(t.data_offset), int(t.n_bytes)
        exp = E.patched(t, want[t.name], np.asarray(src[o:o + nb])) if t.name in want else src[o:o + nb]
        if not np.array_equal(exp, dst[o:o + nb]):
            bad.append(t.name)
    return {"ok": not bad and len(want) > 0, "bad": bad[:20], "patched": len(want), "tensors": len(rd.tensors)}


def ci(v) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def three(c: list[float], pos: str, neg: str, mid: str) -> str:
    return pos if c[0] > 0 else neg if c[1] < 0 else mid


def verdict(scores: Path, results: Path) -> dict:
    names = ("ref", "top4dm16", "ks_org4", "bal4", "e2e_rowx", "rowx_org", "rowx_exp")
    arms = {x: {p: vec(json.loads((scores / f"{x}.json").read_text(encoding="utf-8")), p) for p in PANELS} for x in names}
    n = {p: len(arms["ref"][p]) for p in PANELS}

    def rec(x, idx=None):
        return float(np.mean([100.0 * (arms[x][p] if idx is None else arms[x][p][idx[p]]).mean() for p in RECALL]))
    rng = np.random.default_rng(SEED)
    b = {k: [] for k in ("org_gain", "exp_gain", "who", "inter", "org_all", "exp_all")}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = {x: rec(x, idx) for x in names}
        b["org_gain"].append(r["rowx_org"] - r["ks_org4"])
        b["exp_gain"].append(r["rowx_exp"] - r["bal4"])
        b["who"].append(r["rowx_org"] - r["rowx_exp"])
        b["inter"].append(r["e2e_rowx"] - (r["rowx_org"] + r["rowx_exp"] - r["top4dm16"]))
        b["org_all"].append(points(arms["rowx_org"], idx) - points(arms["ks_org4"], idx))
        b["exp_all"].append(points(arms["rowx_exp"], idx) - points(arms["bal4"], idx))
    r = {x: rec(x) for x in names}
    ref_all = points(arms["ref"])
    run = json.loads((results / "s2n_run.json").read_text(encoding="utf-8"))["arms"]
    valid = {}
    for x in ("rowx_org", "rowx_exp"):
        ex = json.loads((results / f"s2n_split_export_{x}.json").read_text(encoding="utf-8"))
        ru = run.get(x, {})
        valid[x] = bool(ex["ok"] and ru.get("model_sha256") == ex["out_sha256"]
                        and all(ru.get("panels", {}).get(p, {}).get("status") == "COMPLETE" for p in ru.get("panel_set", ["missing"])))
    ok = all(valid.values())
    out = {"valid": valid,
           "recall_vs_ref": {x: r[x] - r["ref"] for x in names[1:]},
           "all_vs_ref": {x: points(arms[x]) - ref_all for x in names[1:]},
           "chat": {x: int(arms[x]["chat"].sum()) for x in names},
           "org": {"recall_gain": r["rowx_org"] - r["ks_org4"], "ci95": ci(b["org_gain"]), "all_gain": points(arms["rowx_org"]) - points(arms["ks_org4"]),
                   "all_ci95": ci(b["org_all"]), "label": three(ci(b["org_gain"]), "GAINS", "LOSES", "NOT SHOWN") if ok else "VOID"},
           "exp": {"recall_gain": r["rowx_exp"] - r["bal4"], "ci95": ci(b["exp_gain"]), "all_gain": points(arms["rowx_exp"]) - points(arms["bal4"]),
                   "all_ci95": ci(b["exp_all"]), "label": three(ci(b["exp_gain"]), "GAINS", "LOSES", "NOT SHOWN") if ok else "VOID"},
           "who": {"org_minus_exp": r["rowx_org"] - r["rowx_exp"], "ci95": ci(b["who"]),
                   "label": three(ci(b["who"]), "EXPERTS HOLD THE RECALL", "ORGANS HOLD THE RECALL", "NOT SEPARATED") if ok else "VOID"},
           "interaction": {"value": r["e2e_rowx"] - (r["rowx_org"] + r["rowx_exp"] - r["top4dm16"]), "ci95": ci(b["inter"]),
                           "label": three(ci(b["inter"]), "THE STACK LOSES LESS", "THE STACK LOSES MORE", "ADDS UP") if ok else "VOID"}}
    return out


def selftest() -> int:
    cases = [(three([0.2, 3], "G", "L", "N"), "G"), (three([-3, -0.1], "G", "L", "N"), "L"), (three([-1, 1], "G", "L", "N"), "N")]
    rd = gguf.GGUFReader(str(BASES["rowx_org"][0]))
    sc = {}
    for t in rd.tensors:
        p = t.name.split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in E.ORGANS:
            sc[f"{p[1]}.{p[2]}"] = 1 + 0.05 * torch.randn(int(t.shape[1]))
    tmp = OUT / "split_selftest.gguf"
    v = export("rowx_org", sc, tmp)
    cases.append((v["ok"] and v["exported"] == v["patched"], True))
    sc2 = {k: v2.clone() for k, v2 in sc.items()}
    k0 = next(iter(sc2))
    sc2[k0][0] *= 1.5
    cases.append((verify("rowx_org", sc2, tmp)["ok"], False))         # a changed scale is caught
    tmp.unlink()
    bad = [i for i, (a, b) in enumerate(cases) if a != b]
    print("selftest", "PASS" if not bad else f"FAIL {bad}", len(cases))
    return 0 if not bad else 1


ROOT = HERE.parents[1]
SCORES = "E:/SiliconLLM_Artifacts/stage1/s1a"
A4, S2 = "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs", "E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/s2/inputs"


def step(*cmd: str) -> None:
    print(">", " ".join(cmd), flush=True)
    if subprocess.run([sys.executable, "-I", *cmd], cwd=ROOT).returncode:
        raise SystemExit(f"stopped: {' '.join(cmd)}")


def chain() -> None:
    R = HERE / "results"
    for arm in BASES:
        js = R / f"s2n_split_export_{arm}.json"
        if not js.exists():
            step("benchmarks/stage2/s2n_split.py", "export", "--arm", arm, "--json", str(js))
        rec = json.loads(js.read_text(encoding="utf-8"))
        if not rec["ok"]:
            raise SystemExit(f"stopped: {arm}'s export does not read back (VOID)")
        if not Path(f"{SCORES}/scores/{arm}.json").exists():
            step("benchmarks/stage2/s2n_run.py", "--arm", arm, "--model", rec["out"], "--model-sha", rec["out_sha256"], "--out", str(R / "s2n_run.json"))
            step("benchmarks/chatbot50/c50_s2_checks.py", "--a4-inputs", A4, "--s2-inputs", S2, "--arm-a4", f"{SCORES}/{arm}", "--arm-s2", f"{SCORES}/{arm}",
                 "--out", f"{SCORES}/scores/{arm}.json")
    step("benchmarks/stage2/s2n_split.py", "verdict", "--scores-dir", f"{SCORES}/scores", "--json", str(R / "s2n_split_verdict.json"))
    print("S2N SPLIT DONE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "verdict", "selftest", "chain"))
    ap.add_argument("--arm", choices=sorted(BASES))
    ap.add_argument("--json", type=Path)
    ap.add_argument("--scores-dir", type=Path)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "chain":
        chain()
        return 0
    if a.cmd == "export":
        out = OUT / f"{a.arm}.gguf"
        rec = {"arm": a.arm, "base": str(BASES[a.arm][0]), "base_sha256": sha256(BASES[a.arm][0]), "scales": str(SCALES),
               "scales_sha256": sha256(SCALES), **export(a.arm, torch.load(SCALES, map_location="cpu"), out)}
        rec.update(out=str(out), out_sha256=sha256(out))
        a.json.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print("EXPORT", "OK" if rec["ok"] else "VOID", json.dumps({k: v for k, v in rec.items() if k != "bad"}), flush=True)
        return 0 if rec["ok"] else 1
    out = verdict(a.scores_dir, a.results)
    a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
