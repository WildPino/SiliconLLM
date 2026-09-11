#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 step 4 (CPU, free) -- measure the H0 gate, and emit the probe the T4 job watches.

Plan: docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md s3 (H0).

GPU TRAINS, CPU MEASURES.  The T4 job reports an fp16 teacher-forced count so the run is
watchable, but the H0 gate is decided here, on this machine, in fp32, with the SAME instrument
every published number in this programme used: E12's chance line, E17/E18's free-running bands,
E20 part B's teacher-forced top-1, the frozen 24x512 heldout slice (ids sha a1a48dc9...) and the
five frozen E6 prompts.  Porting the instrument to a GPU I cannot re-verify would make H0's
number incomparable with E19, E21, E22 and E23, which is the whole reason the gate is a number
from this table and not a loss curve.

  --emit-probe   write the prompt/target ids the T4 job needs, and stop
  --factors F    measure the factored model built from bundle F (init or trained)
  (no --factors)  measure the intact donor -- the control that must read 160/160

THE GATE.  tf >= 48 out of 160, against E22's measured QO512-TB start of 28 and E21's fp32
ceiling of 144.  Both are re-measured here as anchors rather than quoted, so a harness drift
shows up as a failed anchor instead of a fake result.

Env: D_THREADS (6)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
TERNDIR = os.path.abspath(os.path.join(HERE, "..", "ternary"))
ENGDIR = os.path.abspath(os.path.join(HERE, "..", "engine"))
for _p in (DENSDIR, TERNDIR, ENGDIR):
    sys.path.insert(0, _p)

import common as C                                          # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402
import h0_qat as H0                                         # noqa: E402

torch.set_num_threads(int(os.environ.get("D_THREADS", "6")))

E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
E6ENG = os.path.join(ENGDIR, "results", "e6", "engine.json")
OUTDIR = os.path.join(HERE, "results", "h0")
EXPECT_IDS_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
LN2 = 0.6931471805599453
CHANCE = 4.069819
FLOOR, AT_FLOOR_MAX, RANKS_MIN = 12, 14, 80
TF_LO, TF_HI = 107, 119
GATE_TF = 48                                    # the H0 gate
E22_TB_TF, E21_FP32_TF = 28, 144                # start state and fp32 ceiling


def log(m):
    print(m, flush=True)


def band_free(m):
    if m >= RANKS_MIN:
        return "RANKS"
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    return "PARTIAL"


def band_tf(t):
    return "CHEAPER" if t > TF_HI else ("COMPARABLE" if t >= TF_LO else "WORSE")


def greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(torch.tensor([ids])).logits[0, -1]
            ids.append(int(torch.argmax(lg)))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--emit-probe", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    model, tok = C.load_model(dtype=torch.float32)
    model.eval()

    ref = json.load(open(E6REF, encoding="utf-8"))[C.MODEL_ID]
    e6eng = json.load(open(E6ENG, encoding="utf-8"))
    pids = []
    for i, p in enumerate(PROMPTS):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)
    tgts = [ref[i]["ids"][-N_NEW:] for i in range(len(PROMPTS))]

    if a.emit_probe:
        pth = os.path.join(OUTDIR, "h0_probe.json")
        json.dump({"what": "the five frozen E6 prompts and the donor's own greedy continuations; "
                           "the T4 job's watchable tf count, NOT the gate",
                   "model": C.MODEL_ID, "revision": C.REVISION,
                   "n_new": N_NEW, "prompt_ids": pids, "target_ids": tgts,
                   "counted": len(pids) * N_NEW},
                  open(pth, "w", encoding="utf-8"), indent=1)
        log("wrote %s  (%d prompts x %d = %d positions)"
            % (pth, len(pids), N_NEW, len(pids) * N_NEW))
        return 0

    tag = a.tag or (os.path.basename(a.factors) if a.factors else "base")
    if a.factors:
        n_layers = C.arch(model)["n_layers"]
        n = H0.build(model, a.factors, list(range(n_layers)), "cpu")
        log("installed %d factored organs from %s" % (n, a.factors))
    log("== H0 eval: %s ==" % tag)

    ids_ev, byts_ev, meta_ev = C.get_slice(tok, "heldout", 24, 512, 1234)
    if meta_ev["ids_sha256"] != EXPECT_IDS_SHA:
        raise SystemExit("SLICE HASH MISMATCH -- STOP")
    B_TOT = float(byts_ev.sum())

    nats = []
    with torch.no_grad():
        for i in range(ids_ev.shape[0]):
            ch = ids_ev[i:i + 1]
            lg = model(ch).logits
            lp = torch.nn.functional.log_softmax(lg[:, :-1], dim=-1)
            nats.append(-lp.gather(-1, ch[:, 1:].unsqueeze(-1)).squeeze(-1)[0].double())
    bpb = float(torch.stack(nats).sum() / (LN2 * B_TOT))

    allids, matched, per_prompt = [], 0, []
    for i in range(len(PROMPTS)):
        new = greedy(model, pids[i], N_NEW)[-N_NEW:]
        allids.append(new)
        m = sum(1 for x, y in zip(new, tgts[i]) if x == y)
        per_prompt.append({"prompt": i, "free": m})
        matched += m

    tf, ranks = 0, []
    with torch.no_grad():
        for i in range(len(PROMPTS)):
            full = torch.tensor([pids[i] + tgts[i]])
            lg = model(full).logits[0].float()
            h = 0
            for k in range(N_NEW):
                row = lg[len(pids[i]) - 1 + k]
                t = tgts[i][k]
                h += int(int(torch.argmax(row)) == t)
                ranks.append(int((row > row[t]).sum()) + 1)
            per_prompt[i]["tf"] = h
            tf += h

    counted = len(PROMPTS) * N_NEW
    rec = {"plan": "decisions/T4_HEALING_PROPOSAL.md s3 (H0)", "tag": tag,
           "factors": a.factors, "model": C.MODEL_ID, "revision": C.REVISION,
           "eval_slice": meta_ev, "bpb": bpb, "vs_chance": bpb - CHANCE,
           "free": matched, "counted": counted, "teacher_forced": tf,
           "mean_rank": sum(ranks) / float(len(ranks)),
           "rank_le5": sum(1 for x in ranks if x <= 5),
           "band_free": band_free(matched), "band_tf": band_tf(tf),
           "per_prompt": per_prompt, "text": [tok.decode(x) for x in allids],
           "gate": {"metric": "teacher_forced", "threshold": GATE_TF,
                    "start_E22_QO512_TB": E22_TB_TF, "ceiling_E21_fp32": E21_FP32_TF,
                    "passes": tf >= GATE_TF, "delta_from_start": tf - E22_TB_TF},
           "seconds": time.time() - t0}
    out = a.out or os.path.join(OUTDIR, "h0_eval_%s.json"
                                % tag.replace(".npz", "").replace(os.sep, "_"))
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)

    log("  BPB %.6f (%+.6f vs chance)  free %d/%d [%s]  tf %d/%d [%s]  mean rank %.2f"
        % (bpb, bpb - CHANCE, matched, counted, rec["band_free"], tf, counted,
           rec["band_tf"], rec["mean_rank"]))
    log("  per-prompt free %s  tf %s"
        % ([p["free"] for p in per_prompt], [p["tf"] for p in per_prompt]))
    if a.factors:
        log("  GATE tf >= %d : %s   (start %d, fp32 ceiling %d, delta %+d)"
            % (GATE_TF, "PASS" if tf >= GATE_TF else "FAIL", E22_TB_TF, E21_FP32_TF,
               tf - E22_TB_TF))
    log("  wrote %s  [%.0fs]" % (out, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
