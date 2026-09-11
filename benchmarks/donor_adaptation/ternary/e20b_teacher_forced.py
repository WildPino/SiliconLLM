#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E20 part B -- separate one-step argmax fidelity from autoregressive drift.

UNREGISTERED. This metric was built AFTER reading part A and it REPORTS, it does not decide.
E14 section 6 forbids promoting a post-hoc metric to a gate, and nothing here is a gate. The
verdict of E20 is part A's, measured against bands fixed before run 1.

WHY. Every ranking number this programme has quoted since E7 is FREE-RUNNING greedy agreement:
the arm generates its own continuation and is scored against the donor's. One flipped token at
position k makes every later position a comparison between two different contexts, so the metric
conflates two things:

    (1) does the modified head still put the donor's token first, given the donor's own context?
    (2) does the model survive its own mistakes?

Part A found that the ONE prompt reproduced whole (prompt 2, by R2H and OPTH, 32/32) is the ONE
prompt whose donor continuation contains no near-tie: its minimum top-2 logit gap is 1.1630, while
prompts 0/1/3/4 dip to 0.0739 / 0.0453 / 0.2740 / 0.0081. That is consistent with (1) being mostly
intact and (2) doing the damage -- but free-running greedy cannot tell them apart.

This script teacher-forces the donor's own reference continuation through each arm, so each of the
160 positions is an INDEPENDENT test of the head's argmax with the context held identical.

Env: D_THREADS (6), E20B_ONLY (comma list)
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "density")))
sys.path.insert(0, HERE)
sys.path.insert(0, ENGDIR)

import common as C                                          # noqa: E402
import e20_rule_or_format as E20                            # noqa: E402
from t2b_organs import capture, apply_arm                   # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
ONLY = [x.strip() for x in os.environ.get("E20B_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e20b_teacher_forced.json")
ARMS = ["base", "ID", "R0H", "R1H", "R2H", "R3H", "R4H", "R5H", "OPTH", "GPTQH"]
N_NEW = 32


def log(*a):
    print(*a, flush=True)


def main():
    t_start = time.time()
    ref = json.load(open(E6REF, encoding="utf-8"))[HF]

    log("== loading %s ==" % HF)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()

    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    seqs = []
    for i in range(5):
        p = e6eng["prompt_ids"][i]
        tgt = ref[i]["ids"][-N_NEW:]
        seqs.append((p, tgt))
        assert ref[i]["ids"][:len(p)] == p

    ids_cal, _, _ = C.get_slice(tok, "calib", E20.NCAL, E20.SEQCAL, E20.SEEDCAL)
    d_model = model.config.hidden_size
    acc = {"H": torch.zeros(d_model, d_model), "n": 0}

    def hH(mod, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        acc["H"] += x.T @ x
        acc["n"] += x.shape[0]
    hh = model.lm_head.register_forward_hook(hH)
    log("== calibration pass (identical to part A) ==")
    t0 = time.time()
    act_rms = capture(model, ids_cal, None)
    hh.remove()
    H = acc["H"].double()
    rms = act_rms[E20.HEADKEY]
    log("   done in %.0fs, H over %d tokens" % (time.time() - t0, acc["n"]))

    out = {"note": "UNREGISTERED post-hoc metric. Reports, does not decide (E14 section 6).",
           "part_a": "results/e20_rules_on_the_head.json",
           "model": HF, "threads": THREADS, "positions": 5 * N_NEW, "arms": {}}

    donor_gap = None
    for tag in [a for a in ARMS if (not ONLY or a in ONLY or a == "base")]:
        t0 = time.time()
        if tag == "base":
            restore = (lambda: None)
        elif tag == "R3H":
            restore, _ = apply_arm(model, "H", act_rms, None)
        else:
            restore, _ = E20.apply_head(model, tag, H, rms)

        hits, ranks, gaps, per_prompt, by_pos = 0, [], [], [], []
        with torch.no_grad():
            for pi, (p, tgt) in enumerate(seqs):
                full = torch.tensor([p + tgt])
                lg = model(full).logits[0].float()
                h = 0
                for k in range(N_NEW):
                    row = lg[len(p) - 1 + k]
                    t = tgt[k]
                    top2 = torch.topk(row, 2)
                    am = int(top2.indices[0])
                    ok = (am == t)
                    hits += int(ok)
                    h += int(ok)
                    # rank of the donor's token under this arm
                    ranks.append(int((row > row[t]).sum()) + 1)
                    # signed margin of the donor's token: >0 iff it is on top
                    other = row.clone()
                    other[t] = float("-inf")
                    gaps.append(float(row[t] - other.max()))
                    by_pos.append(1 if ok else 0)
                miss = next((k for k in range(N_NEW)
                             if by_pos[pi * N_NEW + k] == 0), None)
                per_prompt.append({"prompt": pi, "top1_hits": h, "of": N_NEW,
                                   "first_teacher_forced_miss": miss,
                                   "by_pos": by_pos[pi * N_NEW:(pi + 1) * N_NEW]})
        restore()
        if tag == "base":
            donor_gap = list(gaps)

        n = len(gaps)
        srt = sorted(range(n), key=lambda i: donor_gap[i])
        terc = [srt[:n // 3], srt[n // 3:2 * n // 3], srt[2 * n // 3:]]
        rec = {"arm": tag, "teacher_forced_top1": hits, "of": n,
               "frac": hits / float(n),
               "median_rank_of_donor_token": sorted(ranks)[n // 2],
               "mean_rank_of_donor_token": sum(ranks) / float(n),
               "rank_le_5": sum(1 for r in ranks if r <= 5),
               "mean_signed_margin": sum(gaps) / float(n),
               "per_prompt": per_prompt,
               "by_donor_gap_tercile": [
                   {"tercile": j, "donor_gap_range":
                    [round(donor_gap[t[0]], 4), round(donor_gap[t[-1]], 4)],
                    "top1_hits": sum(by_pos[i] for i in t), "of": len(t)}
                   for j, t in enumerate(terc)],
               "seconds": time.time() - t0}
        out["arms"][tag] = rec
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  %-6s teacher-forced %3d/%d  %6.2f%%  | median rank %-5d  rank<=5 %3d  "
            "| terciles %s  [%.0fs]"
            % (tag, hits, n, 100.0 * hits / n, rec["median_rank_of_donor_token"],
               rec["rank_le_5"], [t["top1_hits"] for t in rec["by_donor_gap_tercile"]],
               rec["seconds"]))

    # the comparison the whole script exists for
    pa = json.load(open(os.path.join(ENGDIR, "results", "e20_rules_on_the_head.json"),
                        encoding="utf-8"))["arms"]
    tbl = []
    for tag in ARMS:
        if tag in out["arms"] and tag in pa:
            tbl.append({"arm": tag, "free_running": pa[tag]["matched"],
                        "teacher_forced": out["arms"][tag]["teacher_forced_top1"],
                        "bpb": pa[tag]["bpb"]})
    out["free_vs_forced"] = tbl

    # EXACT IDENTITY, not a prediction: if an arm matches the donor at every position < k with the
    # donor's context, then free-running it emits those same tokens, so the contexts agree and it
    # must first diverge at exactly k.  Any mismatch means one of the two harnesses is wrong.
    ident = []
    for tag in ARMS:
        if tag not in out["arms"] or tag not in pa:
            continue
        for pp_f, pp_a in zip(out["arms"][tag]["per_prompt"], pa[tag]["per_prompt"]):
            ident.append({"arm": tag, "prompt": pp_f["prompt"],
                          "first_teacher_forced_miss": pp_f["first_teacher_forced_miss"],
                          "free_running_diverges_at": pp_a["diverges_at"],
                          "agree": pp_f["first_teacher_forced_miss"] == pp_a["diverges_at"]})
    bad = [x for x in ident if not x["agree"]]
    out["G_B1_divergence_identity"] = {"checked": len(ident), "mismatches": len(bad),
                                       "holds": not bad, "detail": bad[:12]}
    log("\n  G-B1 divergence identity: %d/%d agree%s"
        % (len(ident) - len(bad), len(ident),
           "" if not bad else "  MISMATCHES: %s" % bad[:6]))
    out["seconds_total"] = time.time() - t_start
    log("\n  arm     BPB        free-running   teacher-forced   drift cost")
    for r in tbl:
        log("    %-6s %.6f  %3d/160        %3d/160          %+d"
            % (r["arm"], r["bpb"], r["free_running"], r["teacher_forced"],
               r["free_running"] - r["teacher_forced"]))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("\nwrote %s  [%.0fs]" % (OUT, out["seconds_total"]))


if __name__ == "__main__":
    main()
