"""Stage 2 part 1, test 2 verdict: one organ of Ling-mini ternary (one scale per row), trained alone (it chooses
nothing beyond the registered rule).

Per arm, on stage 1's 72 documents against ref (s1_verdict.docs_metrics): top-1 agreement, KL(ref || arm) on ref's
top-32 partition, paired dBPB with its CI95.  The whole organs (org_dense, org_head) also on the 40 chat items (C50-A4's
chat_check and rollout_flags), against ref.
Controls, before any label:
    C1 planted: org_attn_shuf (the trained attention with its output projection's rows permuted) has top-1 < 0.90 and
                KL > 0.10: the rule below fires on a known damage of one layer;
    C2 exports: every export record has its tensors TQ2_0 and bit-exact, the layout checked against the HF checkpoint,
                every other tensor equal to P0's, P0's sha256 checked; the head's gain equals P0's over c (< 1e-6).
A failed control makes the run VOID.
The share of one organ is 1 point of the 5-point budget; on the documents it reads through stage 1's top6 arm (-0.96
points at top-1 0.898, KL 0.094), so the line is top-1 >= 0.90 and KL <= 0.10:
    one layer of a stacked organ (org_attn, org_shexp, org_exps, layer 10 of 19 MoE layers):
        "ONE LAYER WITHIN THE SHARE" at the line, "ONE LAYER OVER THE SHARE" otherwise;
    a whole organ (org_dense: the only dense FFN; org_head):
        "FITS" at the line with chat passes >= ref's - 4 and degenerate rollouts <= ref's + 3, "DOES NOT FIT" otherwise.
Stage flag: any OVER or DOES NOT FIT -> "AN ORGAN DOES NOT STAY TERNARY WITH THIS TRAINING" (the stage's case for
talking with the Consulente before part 2); else "EVERY ORGAN TESTED WITHIN ITS SHARE, ONE LAYER AT A TIME".
Reported, not judged: org_exps against stage 1 (b)'s tern_l10 (group 256), replay (the noise floor), the trainers'
held-out losses.  org_head is in the run only if test 1 read HEAD NEEDS TRAINING.
--selftest runs planted inputs through the rule.

Usage:
    python -I benchmarks/stage2/s2o_verdict.py --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --run benchmarks/stage2/results/s2o_run.json --exports benchmarks/stage2/results \
        --json benchmarks/stage2/results/s2o_verdict.json
    python -I benchmarks/stage2/s2o_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))
sys.path.insert(0, str(HERE))

LINE_TOP1, LINE_KL = 0.90, 0.10
CHAT_MARGIN, DEGEN_MARGIN = 4, 3
GAIN_TOL = 1e-6
LAYER_ARMS = ("org_attn", "org_shexp", "org_exps")
WHOLE_ARMS = ("org_dense", "org_head")
PLANTED = "org_attn_shuf"


def within(m: dict) -> bool:
    return m["top1_agreement"] >= LINE_TOP1 and m["kl_top32_nats"] <= LINE_KL


def export_ok(e: dict) -> bool:
    return (e["all_exact"] and all(e["layout_checked"].values()) and not e["other_tensors_differ"]
            and e["p0_sha256_checked"] and e.get("gain_ratio_err", 0.0) < GAIN_TOL)


def decide(docs: dict, chats: dict, exports: dict) -> dict:
    pl = docs[PLANTED]
    c1 = pl["top1_agreement"] < LINE_TOP1 and pl["kl_top32_nats"] > LINE_KL
    c2 = all(export_ok(e) for e in exports.values()) and set(exports) == set(docs) - {"replay", "tern_l10"}
    out = {"controls": {"C1_planted_top1": pl["top1_agreement"], "C1_planted_kl": pl["kl_top32_nats"], "C1": c1,
                        "C2": c2}}
    if not (c1 and c2):
        out["outcome"] = "VOID"
        return out
    labels = {}
    for arm in LAYER_ARMS:
        labels[arm] = "ONE LAYER WITHIN THE SHARE" if within(docs[arm]) else "ONE LAYER OVER THE SHARE"
    for arm in WHOLE_ARMS:
        if arm in docs:
            ca, cr = chats[arm], chats["ref"]
            ok = (within(docs[arm]) and ca["passes"] >= cr["passes"] - CHAT_MARGIN
                  and ca["degenerate"] <= cr["degenerate"] + DEGEN_MARGIN)
            labels[arm] = "FITS" if ok else "DOES NOT FIT"
    out["labels"] = labels
    bad = [a for a, v in labels.items() if v in ("ONE LAYER OVER THE SHARE", "DOES NOT FIT")]
    out["outcome"] = ("AN ORGAN DOES NOT STAY TERNARY WITH THIS TRAINING" if bad
                      else "EVERY ORGAN TESTED WITHIN ITS SHARE, ONE LAYER AT A TIME")
    out["over"] = bad
    return out


def selftest() -> int:
    def d(t, k):
        return {"top1_agreement": t, "kl_top32_nats": k, "dbpb": 0.0}

    ok = {"all_exact": True, "layout_checked": {"x": True}, "other_tensors_differ": [], "p0_sha256_checked": True}
    arms = LAYER_ARMS + WHOLE_ARMS + (PLANTED,)

    def case(over=None, planted=d(0.6, 1.0), exp=None, chat=(27, 0), head=True):
        docs = {a: d(0.94, 0.04) for a in arms if head or a != "org_head"}
        docs[PLANTED] = planted
        if over:
            docs[over] = d(0.88, 0.12)
        docs["replay"] = d(1.0, 0.0)
        ex = {a: dict(ok) for a in docs if a != "replay"}
        if exp:
            ex[exp[0]] = dict(ok, **exp[1])
        ch = {"ref": {"passes": 28, "degenerate": 0}, **{a: {"passes": chat[0], "degenerate": chat[1]}
                                                          for a in WHOLE_ARMS}}
        return docs, ch, ex

    every, flag = "EVERY ORGAN TESTED WITHIN ITS SHARE, ONE LAYER AT A TIME", "AN ORGAN DOES NOT STAY TERNARY WITH THIS TRAINING"
    cases = {
        "all within": (case(), every),
        "all within, no head arm": (case(head=False), every),
        "attn over": (case(over="org_attn"), flag),
        "dense over": (case(over="org_dense"), flag),
        "chat fails": (case(chat=(23, 0)), flag),
        "degenerate": (case(chat=(27, 4)), flag),
        "planted not seen": (case(planted=d(0.95, 0.03)), "VOID"),
        "planted top-1 only": (case(planted=d(0.85, 0.05)), "VOID"),
        "export not exact": (case(exp=("org_shexp", {"all_exact": False})), "VOID"),
        "head gain not P0 / c": (case(exp=("org_head", {"gain_ratio_err": 0.5})), "VOID"),
        "layout differs": (case(exp=("org_attn", {"layout_checked": {"x": False}})), "VOID"),
    }
    bad = 0
    for name, ((dc, ch, ex), want) in cases.items():
        got = decide(dc, ch, ex)["outcome"]
        bad += got != want
        print("selftest", "ok  " if got == want else "FAIL", name, got)
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--run")
    ap.add_argument("--exports")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s1_verdict import docs_metrics
    from s2h_verdict import chat
    work = Path(a.work)
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    arms = [x for x in LAYER_ARMS + WHOLE_ARMS + (PLANTED,) if x in run["arms"]]
    for arm in arms:
        rec = run["arms"][arm]
        if any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in rec["panel_set"]):
            raise SystemExit(f"{arm}: not every panel COMPLETE")
    missing = [x for x in LAYER_ARMS + ("org_dense", PLANTED) if x not in arms]
    if missing:
        raise SystemExit(f"registered arms missing: {missing}")
    exports = {arm: json.loads((Path(a.exports) / f"s2o_export_{arm}.json").read_text(encoding="utf-8"))
               for arm in arms}
    docs = {arm: docs_metrics(work, arm) for arm in arms + ["replay", "tern_l10"]}
    chats = {arm: chat(work, arm) for arm in ["ref"] + [x for x in WHOLE_ARMS if x in arms]}
    out = decide(docs, chats, exports)
    out.update(docs=docs, chat=chats)
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for arm, m in docs.items():
        print(f"{arm:14s} top1 {m['top1_agreement']:.4f} kl {m['kl_top32_nats']:.4f} dbpb {m['dbpb']:+.4f} "
              f"[{m['dbpb_ci95'][0]:+.4f},{m['dbpb_ci95'][1]:+.4f}]")
    print("chat", chats)
    print("controls", out["controls"], "labels", out.get("labels"))
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
