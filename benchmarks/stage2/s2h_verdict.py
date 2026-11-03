"""Stage 2 part 1, test 1 verdict: Ling-mini's head ternary with one scale per row, untrained (it chooses nothing beyond
the registered rule).

Per arm, on stage 1's 72 documents against ref (stage 1's s1_verdict.docs_metrics): top-1 agreement, KL(ref || arm) on
ref's top-32 partition, paired dBPB with its CI95.  On the 40 chat items (the primary arm head_tern_col and ref, scored by C50-A4's
chat_check): passes and degenerate rollouts (C50-A4's rollout_flags).
Controls, before any label:
    C1 planted:  head_shuf's top-1 agreement < 0.10, so the panel sees the head;
    C2 builds:   every build record has the head exact in TQ2_0, the final norm's gain as built and every other
                 tensor equal to P0's.
A failed control makes the run VOID.
The head's share of the 5-point budget is 1 point (one of five organs).  Stage 1's top6 arm cost -0.96 points with
document top-1 0.898 and KL 0.094, so the share reads on the documents as:
    "HEAD FITS ITS SHARE UNTRAINED"  head_tern_col: top-1 >= 0.90, KL <= 0.10, chat passes >= ref's - 4 and
                                     degenerate chat rollouts <= ref's + 3;
    "HEAD NEEDS TRAINING"            otherwise.
Reported, not judged: head_tern_im's metrics (the fit without the column scale), replay's (the noise floor of the
panel), the column-scale exponent, the builds' row statistics.
--selftest runs planted inputs through the rule and fails unless each gets its registered outcome.

Usage:
    python -I benchmarks/stage2/s2h_verdict.py --work E:/SiliconLLM_Artifacts/stage1/s1a \
        --run benchmarks/stage2/results/s2h_run.json --builds benchmarks/stage2/results \
        --json benchmarks/stage2/results/s2h_verdict.json
    python -I benchmarks/stage2/s2h_verdict.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
sys.path.insert(0, str(HERE.parents[0] / "chatbot50"))

FIT_TOP1, FIT_KL = 0.90, 0.10
PLANTED_TOP1 = 0.10
CHAT_MARGIN, DEGEN_MARGIN = 4, 3
ARMS = ("head_tern_col", "head_tern_im", "head_shuf")
BUILDS = ("tern_im", "tern_col", "shuf")
A4_INPUTS = Path("E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/a4/inputs")


def chat(work: Path, arm: str) -> dict:
    import c50_a4_checks as A4
    tok = A4.Tokenizer.from_file(A4.TOKENIZER)
    rows = A4.gen_rows(work / arm, "chat")
    passes = degenerate = 0
    for m in A4.meta(A4_INPUTS, "chat"):
        r = rows[m["item"]]
        passes += int(A4.chat_check(m, A4.decode(tok, r["gen"]), r["stop"] == "EOS")["pass"])
        degenerate += int(A4.rollout_flags(r["gen"])["degenerate"])
    return {"passes": passes, "degenerate": degenerate, "n": len(rows)}


def decide(docs: dict, chats: dict, builds: dict) -> dict:
    c1 = docs["head_shuf"]["top1_agreement"] < PLANTED_TOP1
    c2 = all(b["head_tq2_0_exact"] and b["output_norm_exact"] and not b["other_tensors_differ"]
             and b["p0_sha256_checked"] for b in builds.values()) and set(builds) == set(BUILDS)
    out = {"controls": {"C1_shuf_top1": docs["head_shuf"]["top1_agreement"], "C1": c1, "C2": c2}}
    if not (c1 and c2):
        out["outcome"] = "VOID"
        return out
    p, cr, ca = docs["head_tern_col"], chats["ref"], chats["head_tern_col"]
    gates = {"top1": p["top1_agreement"] >= FIT_TOP1, "kl": p["kl_top32_nats"] <= FIT_KL,
             "chat": ca["passes"] >= cr["passes"] - CHAT_MARGIN,
             "degenerate": ca["degenerate"] <= cr["degenerate"] + DEGEN_MARGIN}
    out["gates"] = gates
    out["outcome"] = "HEAD FITS ITS SHARE UNTRAINED" if all(gates.values()) else "HEAD NEEDS TRAINING"
    return out


def selftest() -> int:
    def d(top1, kl):
        return {"top1_agreement": top1, "kl_top32_nats": kl, "dbpb": 0.0}

    ok = {"head_tq2_0_exact": True, "output_norm_exact": True, "other_tensors_differ": [], "p0_sha256_checked": True}
    ok_b = {a: ok for a in BUILDS}
    bad_b = dict(ok_b, tern_col=dict(ok, head_tq2_0_exact=False))
    bad_g = dict(ok_b, tern_col=dict(ok, output_norm_exact=False))
    chats = lambda p, g: {"ref": {"passes": 28, "degenerate": 0}, "head_tern_col": {"passes": p, "degenerate": g}}  # noqa: E731
    docs = lambda t, k, s=0.001: {"head_tern_col": d(t, k), "head_tern_im": d(t - 0.05, k * 2), "head_shuf": d(s, 9.0)}  # noqa: E731
    cases = {  # name: (docs, chats, builds, outcome)
        "fits": (docs(0.93, 0.06), chats(27, 0), ok_b, "HEAD FITS ITS SHARE UNTRAINED"),
        "top-1 short": (docs(0.88, 0.06), chats(27, 0), ok_b, "HEAD NEEDS TRAINING"),
        "KL over": (docs(0.93, 0.15), chats(27, 0), ok_b, "HEAD NEEDS TRAINING"),
        "chat fails": (docs(0.93, 0.06), chats(23, 0), ok_b, "HEAD NEEDS TRAINING"),
        "degenerate": (docs(0.93, 0.06), chats(27, 4), ok_b, "HEAD NEEDS TRAINING"),
        "planted not seen": (docs(0.93, 0.06, s=0.5), chats(27, 0), ok_b, "VOID"),
        "build not exact": (docs(0.93, 0.06), chats(27, 0), bad_b, "VOID"),
        "gain not as built": (docs(0.93, 0.06), chats(27, 0), bad_g, "VOID"),
    }
    bad = 0
    for name, (dc, ch, bu, want) in cases.items():
        got = decide(dc, ch, bu)["outcome"]
        bad += got != want
        print("selftest", "ok  " if got == want else "FAIL", name, got)
    print("selftest", "FAIL" if bad else "PASS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--run")
    ap.add_argument("--builds")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    from s1_verdict import docs_metrics
    work = Path(a.work)
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    for arm in ARMS:
        rec = run["arms"].get(arm)
        need = ("docs", "chat") if arm == "head_tern_col" else ("docs",)
        if rec is None or any(rec["panels"].get(p, {}).get("status") != "COMPLETE" for p in need):
            raise SystemExit(f"{arm}: not every panel COMPLETE")
    builds = {b: json.loads((Path(a.builds) / f"s2h_build_{b}.json").read_text(encoding="utf-8"))
              for b in BUILDS}
    docs = {arm: docs_metrics(work, arm) for arm in ARMS + ("replay",)}
    chats = {arm: chat(work, arm) for arm in ("ref", "head_tern_col")}
    out = decide(docs, chats, builds)
    out.update(docs=docs, chat=chats, alpha=builds["tern_col"]["alpha"],
               row_stats={b: v["row_stats"] for b, v in builds.items()})
    Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for arm, m in docs.items():
        print(f"{arm:13s} top1 {m['top1_agreement']:.4f} kl {m['kl_top32_nats']:.4f} dbpb {m['dbpb']:+.4f} "
              f"[{m['dbpb_ci95'][0]:+.4f},{m['dbpb_ci95'][1]:+.4f}]")
    print("chat", chats)
    print("controls", out["controls"], "gates", out.get("gates"))
    print("outcome", out["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
