"""Stage 1 (a) verdict: fewer routed experts per token on Ling-mini, no training (it chooses nothing beyond the
registered rule).

Points of an arm = the unweighted mean over the six item panels (nq, tqa, gsm8k, arc, mmlu, chat) of its accuracy in
percent, scored by c50_s2_checks.py.  Delta(arm) = points(arm) - points(ref), paired on the same items; CI95 from
10,000 bootstrap draws (seed 20261030) that resample items within each panel, the same draw for every arm.
Labels, on the CI95 of Delta (the 2.5-point share is the one declared for this lever, 5 points the whole budget):
    WITHIN SHARE   lower end >= -2.5
    WITHIN BUDGET  lower end >= -5 (and not WITHIN SHARE)
    OVER BUDGET    upper end < -5
    UNDECIDED      otherwise; every decision below treats it as not fitting
Chat gate (the budget also asks for a chat that stays usable): CHAT OK iff the arm passes at least as many chat items
as ref minus 4 (of 40), and its degenerate rollouts on chat + gsm8k (a 32-token run, an 8-token loop repeated three
times, or nothing generated: C50-A4's rollout_flags) are at most ref's plus 3.
An arm FITS iff its label is WITHIN SHARE or WITHIN BUDGET and CHAT OK; it FITS THE SHARE iff WITHIN SHARE and CHAT OK.
Treatment against its trivial control (top-k by biased score, and topw4 by weight, against k random slots of the
top-8; Top-P against as many random
slots): margin = Delta(treatment) - Delta(control), CI95 as above.  BEATS CONTROL iff the CI95 lies above 0.  If it
does not, ROUTER NO BETTER THAN CHANCE only when the control's own CI95 lies below 0 (the control measurably loses);
otherwise CONTROL UNINFORMATIVE.
Top-P against top4 (margin and CI95) only if Top-P's realized mean of kept experts over all its panels is <= 4.05;
otherwise NOT MATCHED.
Decisions the rule fixes:
    training_k = 4 unless top4, topw4 or top4d FITS THE SHARE; then 2 unless top2 FITS; then none.
    k4_form = "everywhere (biased score)" if top4 FITS THE SHARE, else "everywhere (weight)" if topw4 does, else
              "decode-only" if top4d does, else none.
Reported: topw4 against top4 (margin and CI95), at exactly matched compute.
Reported, not gated: per-panel accuracies; replay's Delta and CI95 (the noise of an equivalent run); on the 72
documents, per arm against ref: top-1 agreement, KL(ref || arm) on the reference's top-32 partition (32 ids plus one
bin for the rest, a lower bound of the full KL), the paired dBPB with a document bootstrap, and whether top-1 and KL lie
within replay's; the realized mean of kept experts, overall and on decode tokens; the share of tokens whose biased
slot order differs from weight order.
--selftest runs planted arms through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage1/s1_verdict.py --scores-dir <dir with <arm>.json from c50_s2_checks> \
        --work E:/SiliconLLM_Artifacts/stage1/s1a --run benchmarks/stage1/results/s1a_run.json --json verdict.json
    python -I benchmarks/stage1/s1_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SEED, DRAWS = 20261030, 10_000
PANELS = ("nq", "tqa", "gsm8k", "arc", "mmlu", "chat")
SHARE, BUDGET = 2.5, 5.0
CHAT_MARGIN, DEGEN_MARGIN = 4, 3
TOPP_MATCH = 4.05
PAIRS = {"top4": "rnd4", "topw4": "rnd4", "top6": "rnd6", "top2": "rnd2", "topp": "rndp"}
REC = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"),
                ("logp_target", "<f4"), ("logp_top1", "<f4"), ("lse", "<f8")])
TOP_K = 32
TOP = np.dtype([("doc", "<i4"), ("pos", "<i4"), ("target", "<i4"), ("top1", "<i4"), ("logp_target", "<f4"),
                ("k", "<i4"), ("lse", "<f8"), ("ids", "<i4", (TOP_K,)), ("logp", "<f4", (TOP_K,))])
CHATBOT50 = Path(__file__).resolve().parents[1] / "chatbot50"


def vec(scores: dict, panel: str) -> np.ndarray:
    return np.array([bool(i["correct"]) for i in sorted(scores[panel]["items"], key=lambda i: i["item"])])


def points(v: dict[str, np.ndarray], idx=None) -> float:
    return float(np.mean([100.0 * (v[p] if idx is None else v[p][idx[p]]).mean() for p in PANELS]))


def label(ci: list[float]) -> str:
    if ci[0] >= -SHARE:
        return "WITHIN SHARE"
    if ci[0] >= -BUDGET:
        return "WITHIN BUDGET"
    if ci[1] < -BUDGET:
        return "OVER BUDGET"
    return "UNDECIDED"


def decide(arms: dict[str, dict[str, np.ndarray]], degenerate: dict[str, int], mean_kept: dict[str, float]) -> dict:
    ref = arms["ref"]
    n = {p: len(ref[p]) for p in PANELS}
    for a, v in arms.items():
        if {p: len(v[p]) for p in PANELS} != n:
            raise SystemExit(f"{a} does not score the same items as ref")
    rng = np.random.default_rng(SEED)
    boot = {a: [] for a in arms}
    for _ in range(DRAWS):
        idx = {p: rng.integers(0, n[p], n[p]) for p in PANELS}
        r = points(ref, idx)
        for a, v in arms.items():
            boot[a].append(points(v, idx) - r)
    boot = {a: np.array(b) for a, b in boot.items()}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    out = {"ref_points": points(ref), "arms": {}}
    for a, v in arms.items():
        c = ci(boot[a])
        chat_ok = (int(v["chat"].sum()) >= int(ref["chat"].sum()) - CHAT_MARGIN
                   and degenerate[a] <= degenerate["ref"] + DEGEN_MARGIN)
        lab = label(c)
        out["arms"][a] = {"points": points(v), "delta": points(v) - points(ref), "ci95": c, "label": lab,
                          "chat_ok": chat_ok, "chat_passes": int(v["chat"].sum()), "degenerate": degenerate[a],
                          "fits": lab in ("WITHIN SHARE", "WITHIN BUDGET") and chat_ok,
                          "fits_share": lab == "WITHIN SHARE" and chat_ok, "mean_kept": mean_kept.get(a),
                          "panels": {p: 100.0 * float(v[p].mean()) for p in PANELS}}
    for t, c in PAIRS.items():
        if t in arms and c in arms:
            mci = ci(boot[t] - boot[c])
            if mci[0] > 0:
                outcome = "BEATS CONTROL"
            elif out["arms"][c]["ci95"][1] < 0:
                outcome = "ROUTER NO BETTER THAN CHANCE"
            else:
                outcome = "CONTROL UNINFORMATIVE"
            out["arms"][t]["control"] = {"arm": c, "margin": out["arms"][t]["delta"] - out["arms"][c]["delta"],
                                         "ci95": mci, "outcome": outcome}
    if "topp" in arms and "top4" in arms:
        mk = mean_kept.get("topp")
        if mk is not None and mk <= TOPP_MATCH:
            out["topp_vs_top4"] = {"mean_kept": mk, "margin": out["arms"]["topp"]["delta"] - out["arms"]["top4"]["delta"],
                                   "ci95": ci(boot["topp"] - boot["top4"])}
        else:
            out["topp_vs_top4"] = {"mean_kept": mk, "outcome": "NOT MATCHED"}
    fs = lambda a: a in out["arms"] and out["arms"][a]["fits_share"]  # noqa: E731
    fits = lambda a: a in out["arms"] and out["arms"][a]["fits"]  # noqa: E731
    out["training_k"] = 4 if not (fs("top4") or fs("topw4") or fs("top4d")) else 2 if not fits("top2") else None
    out["k4_form"] = ("everywhere (biased score)" if fs("top4") else "everywhere (weight)" if fs("topw4")
                      else "decode-only" if fs("top4d") else None)
    if "topw4" in arms and "top4" in arms:
        out["topw4_vs_top4"] = {"margin": out["arms"]["topw4"]["delta"] - out["arms"]["top4"]["delta"],
                                "ci95": ci(boot["topw4"] - boot["top4"])}
    return out


def docs_metrics(work: Path, arm: str) -> dict:
    sys.path.insert(0, str(CHATBOT50))
    from c50_compare import byte_lengths
    import c50_a4_checks as A4
    r = np.fromfile(work / "ref" / "docs.pos.bin", dtype=REC)
    a = np.fromfile(work / arm / "docs.pos.bin", dtype=REC)
    if len(r) != len(a) or np.any(r["doc"] != a["doc"]) or np.any(r["pos"] != a["pos"]):
        raise SystemExit(f"{arm}: documents positions differ from ref")
    blen = byte_lengths(Path(A4.TOKENIZER))[r["target"]].astype(np.float64)
    bpb = {}
    for name, x in (("ref", r), (arm, a)):
        bpb[name] = {int(d): float(-x["logp_target"][x["doc"] == d].astype(np.float64).sum() / np.log(2)
                                   / blen[x["doc"] == d].sum()) for d in np.unique(x["doc"])}
    docs = sorted(bpb["ref"])
    dd = np.array([bpb[arm][d] - bpb["ref"][d] for d in docs])
    rng = np.random.default_rng(SEED)
    bb = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(DRAWS)]
    out = {"top1_agreement": float(np.mean(r["top1"] == a["top1"])), "dbpb": float(dd.mean()),
           "dbpb_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], "documents": len(docs),
           "kl_top32_nats": 0.0}
    if arm != "ref" and (work / arm / "docs.top.bin").exists() and arm != "top4d":
        rt = np.fromfile(work / "ref" / "docs.top.bin", dtype=TOP)
        at = np.fromfile(work / arm / "docs.top.bin", dtype=TOP)
        if len(rt) != len(r) or len(at) != len(r) or np.any(rt["ids"] != at["ids"]):
            raise SystemExit(f"{arm}: top-32 partitions differ from ref")
        lr, la = rt["logp"].astype(np.float64), at["logp"].astype(np.float64)
        pr = np.exp(lr)
        rest_r = np.clip(1.0 - pr.sum(1), 1e-12, None)
        rest_a = np.clip(1.0 - np.exp(la).sum(1), 1e-12, None)
        kl = (pr * (lr - la)).sum(1) + rest_r * np.log(rest_r / rest_a)
        out.update({"kl_top32_nats": float(kl.mean()), "ref_rest_mass": float(rest_r.mean())})
    return out


def degenerate_count(arm_dir: Path) -> int:
    sys.path.insert(0, str(CHATBOT50))
    import c50_a4_checks as A4
    n = 0
    for panel in ("chat", "gsm8k"):
        for line in (arm_dir / f"{panel}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                n += int(A4.rollout_flags(json.loads(line)["gen"])["degenerate"])
    return n


def moe_totals(rec: dict) -> dict:
    tot = {"slots": 0, "kept": 0, "dslots": 0, "dkept": 0, "misordered": 0, "tokens": 0}
    for r in rec["panels"].values():
        for line in r.get("moe", []):
            g = lambda k: int(re.search(rf"\b{k}=(\d+)", line).group(1))  # noqa: E731
            mean_d = float(re.search(r"decode_mean_kept=([\d.]+)", line).group(1))
            tot["slots"] += g("slots"); tot["kept"] += g("kept"); tot["dslots"] += g("decode_slots")
            tot["dkept"] += round(mean_d * g("decode_slots") / 8)
            tot["misordered"] += g("misordered"); tot["tokens"] += g("tokens_seen")
    return {"mean_kept": 8.0 * tot["kept"] / tot["slots"] if tot["slots"] else None,
            "decode_mean_kept": 8.0 * tot["dkept"] / tot["dslots"] if tot["dslots"] else None,
            "misordered_share": tot["misordered"] / tot["tokens"] if tot["tokens"] else None}


def selftest() -> int:
    rng = np.random.default_rng(1)
    acc = {"nq": 0.29, "tqa": 0.49, "gsm8k": 0.81, "arc": 0.53, "mmlu": 0.49, "chat": 0.70}
    size = {"nq": 600, "tqa": 600, "gsm8k": 100, "arc": 600, "mmlu": 570, "chat": 40}
    ref = {p: rng.random(size[p]) < acc[p] for p in PANELS}

    def lose(base, share, panels=PANELS):   # break a share of base's right answers
        out = {}
        for p in PANELS:
            v = base[p].copy()
            if p in panels:
                right = np.flatnonzero(v)
                v[right[: int(round(share * len(right)))]] = False
            out[p] = v
        return out

    deg = {a: 2 for a in ("ref", "top6", "rnd6", "top4", "rnd4", "topw4", "top4d", "top2", "rnd2", "topp", "rndp")}
    arms = {"ref": ref, "top6": lose(ref, 0.0), "rnd6": lose(ref, 0.0), "top4": lose(ref, 0.01),
            "rnd4": lose(ref, 0.40), "top2": lose(ref, 0.30), "rnd2": lose(ref, 0.30)}
    o = decide(arms, deg, {"topp": 4.0})
    a = o["arms"]
    checks = {"identical arm WITHIN SHARE": a["top6"]["label"] == "WITHIN SHARE" and a["top6"]["delta"] == 0.0,
              "identical control is uninformative": a["top6"]["control"]["outcome"] == "CONTROL UNINFORMATIVE",
              "small loss within share": a["top4"]["label"] == "WITHIN SHARE",
              "big loss OVER BUDGET": a["top2"]["label"] == "OVER BUDGET",
              "treatment beats a much worse control": a["top4"]["control"]["outcome"] == "BEATS CONTROL",
              "equal loss to a losing control = chance": a["top2"]["control"]["outcome"] == "ROUTER NO BETTER THAN CHANCE",
              "training at 2 when 4 fits the share and 2 does not": o["training_k"] == 2,
              "k4 everywhere": o["k4_form"] == "everywhere (biased score)"}
    arms["top4"] = lose(ref, 0.15)
    arms["top4d"] = lose(ref, 0.01, ("nq", "tqa", "gsm8k", "chat"))
    arms["topp"], arms["rndp"] = lose(ref, 0.02), lose(ref, 0.10)
    o2 = decide(arms, deg, {"topp": 4.3})
    checks["topp compared at matched compute"] = "margin" in decide(arms, deg, {"topp": 4.0})["topp_vs_top4"]
    checks["top4 over budget"] = o2["arms"]["top4"]["label"] == "OVER BUDGET"
    checks["decode-only rescues k4"] = o2["k4_form"] == "decode-only" and o2["training_k"] == 2
    del arms["top4d"]
    o3 = decide(arms, deg, {"topp": 4.3})
    arms["topw4"] = lose(ref, 0.01)
    o3w = decide(arms, deg, {"topp": 4.3})
    checks["weight order rescues k4"] = (o3w["k4_form"] == "everywhere (weight)" and o3w["training_k"] == 2
                                         and "margin" in o3w["topw4_vs_top4"])
    del arms["topw4"]
    checks["training at 4 when neither k4 form fits the share"] = o3["training_k"] == 4 and o3["k4_form"] is None
    arms["top4"] = lose(ref, 0.01)
    deg2 = dict(deg, top4=9)
    o4 = decide(arms, deg2, {})
    checks["chat gate: degenerate rollouts block a within-share arm"] = (o4["arms"]["top4"]["label"] == "WITHIN SHARE"
                                                                        and not o4["arms"]["top4"]["fits"]
                                                                        and o4["training_k"] == 4)
    arms["top4"] = lose(ref, 0.20, ("chat",))
    o5 = decide(arms, deg, {})
    checks["chat gate: 8 lost chat items block"] = not o5["arms"]["top4"]["chat_ok"]
    checks["labels on the CI"] = (label([-2.0, 1.0]) == "WITHIN SHARE" and label([-4.0, -1.0]) == "WITHIN BUDGET"
                                  and label([-9.0, -5.5]) == "OVER BUDGET" and label([-7.0, -3.0]) == "UNDECIDED")
    checks["topp not matched above 4.05"] = o2["topp_vs_top4"].get("outcome") == "NOT MATCHED"
    for k, v in checks.items():
        print("selftest", "ok  " if v else "FAIL", k)
    bad = not all(checks.values())
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scores-dir")
    ap.add_argument("--work")
    ap.add_argument("--run")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    work = Path(args.work)
    arms, degenerate, moe = {}, {}, {}
    for a, rec in run["arms"].items():
        if any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in PANELS + ("docs",)):
            print(f"{a}: not every panel COMPLETE; left out")
            continue
        s = json.loads((Path(args.scores_dir) / f"{a}.json").read_text(encoding="utf-8"))
        arms[a] = {p: vec(s, p) for p in PANELS}
        degenerate[a] = degenerate_count(work / a)
        moe[a] = moe_totals(rec)
    if "ref" not in arms:
        raise SystemExit("ref is not complete")
    out = decide(arms, degenerate, {a: m["mean_kept"] for a, m in moe.items()})
    floor = docs_metrics(work, "replay") if "replay" in arms else None
    for a in arms:
        d = docs_metrics(work, a)
        if floor and a not in ("ref", "replay"):
            d["within_replay"] = (d["top1_agreement"] >= floor["top1_agreement"]
                                  and d["kl_top32_nats"] <= floor["kl_top32_nats"])
        out["arms"][a]["docs"] = d
        out["arms"][a]["moe"] = moe[a]
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for a, v in out["arms"].items():
        c = v.get("control", {})
        print(f"{a:6s} points {v['points']:6.2f} delta {v['delta']:+6.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] "
              f"{v['label']:13s} chat_ok={v['chat_ok']} {c.get('outcome', '')} "
              f"top1 {v['docs']['top1_agreement']:.4f} kl {v['docs']['kl_top32_nats']:.4f} kept {v['moe']['mean_kept']}")
    print("topp_vs_top4", out.get("topp_vs_top4"))
    print("training_k", out["training_k"], "k4_form", out["k4_form"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
