#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E18 part B -- how much ternarization can a model that still RANKS survive?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E18_THE_RANKING_LADDER.md,
pushed before any arm ran (commit 2ac74f5).

E17 established that no converted donor ranks -- three scales, two rules, two folds, both head
settings, and E18 part A then showed every one of those arms scores at or BELOW a constant-token
predictor (floor 12/160 at 1.5B, 11/160 at 0.5B, the token being '\\n').

But every one of those arms converts EVERYTHING at once.  T2b measured an organ ladder in BPB
spanning -3.302 to -0.586 against the chance line, and FIVE of its seven rungs have never been
generated with -- only FA and FAH, the two worst, ever reached the engine:

  base   nothing            0.767595   -3.302224   harness control, must reproduce ref.json
  I      identity path      0.767595   -3.302224   instrument control, must be token-identical
  H      lm_head only       1.106584   -2.963235   EMPTY
  A      q,k,v,o only       1.903569   -2.166251   EMPTY
  F      gate,up,down only  2.476967   -1.592852   EMPTY
  FA     F+A  (= E1 TQ)     3.484251   -0.585568   12/160 on the engine
  FAH    FA+H (= TQH)       3.475706   -0.594113   10/160 on the engine

This runs in PyTorch, not the engine: the QWENDON1 header carries ONE global `quant` field, so a
mixed-precision organ arm is not expressible in the format without changing it.  The question is
about the MODEL, not the runtime, and E1 s2.1 measured the two at +1.5347e-05 BPB on an identical
arm.  FA and FAH are cross-instrument replications and are the reason the empty rungs can be
trusted: the harness is validated against two known engine values before any of them is read.

Nothing is re-derived.  capture() and apply_arm() are imported from t2b_organs -- the same
definitions that produced the BPB column above.  PROMPTS/N_NEW come from e6_generate.  The
reference is results/e6/ref.json, the same file part A computed the floor from.

Env: D_THREADS (6), E18_ONLY (comma list of arms), E18_SMOKE (1 = 2 prompts, 8 new tokens)
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
from t2b_organs import capture, apply_arm, ARM_ORGANS       # noqa: E402
from e6_generate import PROMPTS, N_NEW                      # noqa: E402

THREADS = int(os.environ.get("D_THREADS", "6"))
torch.set_num_threads(THREADS)
SMOKE = os.environ.get("E18_SMOKE", "0") == "1"
ONLY = [x.strip() for x in os.environ.get("E18_ONLY", "").split(",") if x.strip()]

HF = "Qwen/Qwen2.5-1.5B"
E6REF = os.path.join(ENGDIR, "results", "e6", "ref.json")
OUT = os.path.join(ENGDIR, "results", "e18_ranking_ladder%s.json" % ("_smoke" if SMOKE else ""))

ARMS = ["base", "I", "H", "A", "F", "FA", "FAH"]

# T2b's engine column, quoted and never re-measured here
BPB = {"base": 0.767595, "I": 0.767595, "H": 1.106584, "A": 1.903569,
       "F": 2.476967, "FA": 3.484251, "FAH": 3.475706}
CHANCE = 4.069819                       # log2(151936) / 4.229452

# brief s5, derived before the run
FLOOR = 12                              # part A: best constant predictor on the 1.5B reference
MARGIN = 2                              # E17: a change rewriting 49% of the output moved 2 tokens
AT_FLOOR_MAX = FLOOR + MARGIN           # 14
RANKS_MIN = 80                          # half the positions -- a registered convention
ENGINE_REPLICATION = {"FA": 12, "FAH": 10}

NCAL, SEQCAL, SEEDCAL = 32, 512, 42424


def label(m):
    if m <= AT_FLOOR_MAX:
        return "AT-FLOOR"
    if m >= RANKS_MIN:
        return "RANKS"
    return "ABOVE-FLOOR-DOES-NOT-RANK"


def greedy(model, prompt_ids, n_new):
    """Exactly e6_generate.stage_ref's loop: full forward each step, argmax, no KV cache, so the
    reference is reproduced by construction rather than by a second implementation of it."""
    ids = list(prompt_ids)
    gaps = []
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(torch.tensor([ids])).logits[0, -1].float()
            top = torch.topk(lg, 2)
            gaps.append(float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
    return ids, gaps


def main():
    t_start = time.time()
    n_new = 8 if SMOKE else N_NEW
    prompts = PROMPTS[:2] if SMOKE else PROMPTS

    ref = json.load(open(E6REF, encoding="utf-8"))[HF]
    refids = {r["prompt"]: r["ids"][-N_NEW:] for r in ref}

    print("== loading %s ==" % HF, flush=True)
    model, tok = C.load_model(dtype=torch.float32)
    model.eval()
    assert model.config.vocab_size == 151936, model.config.vocab_size

    # prompt ids, and the check that they are the ones E6 used
    e6eng = json.load(open(os.path.join(ENGDIR, "results", "e6", "engine.json"), encoding="utf-8"))
    pids = []
    for i, p in enumerate(prompts):
        q = tok(p)["input_ids"]
        assert q == e6eng["prompt_ids"][i], "prompt %d differs from E6's stored ids" % i
        pids.append(q)

    out = {"brief": "briefs/BRIEF_E18_THE_RANKING_LADDER.md (2ac74f5)",
           "instrument": "PyTorch fp32 eager, greedy, no KV cache -- e6_generate.stage_ref's loop",
           "why_not_the_engine": "QWENDON1 carries one global quant field; a mixed-precision organ "
                                 "arm is not expressible in the format without changing it",
           "model": HF, "smoke": SMOKE, "threads": THREADS,
           "n_new": n_new, "n_prompts": len(prompts),
           "bands": {"floor_from_part_A": FLOOR, "uninterpretable_margin_from_E17": MARGIN,
                     "at_floor_max": AT_FLOOR_MAX, "ranks_min": RANKS_MIN,
                     "ceiling": len(prompts) * n_new},
           "bpb_from_t2b": BPB, "chance_bpb": CHANCE, "arms": {}}
    if os.path.exists(OUT):
        try:
            out["arms"] = json.load(open(OUT, encoding="utf-8")).get("arms", {})
        except Exception:
            pass

    # ---- calibration, captured once on the untouched model (T2b's constants)
    ids_cal, _, meta_cal = C.get_slice(tok, "calib", NCAL, SEQCAL, SEEDCAL)
    if SMOKE:
        ids_cal = ids_cal[:2]
    out["calib_tokens_T"] = int(ids_cal.shape[0] * ids_cal.shape[1])
    print("== capturing activation RMS (T=%d) ==" % out["calib_tokens_T"], flush=True)
    t0 = time.time()
    act_rms = capture(model, ids_cal, None)
    print("   done in %.0fs, %d organ keys" % (time.time() - t0, len(act_rms)), flush=True)

    base_ids = None
    arms = [a for a in ARMS if (not ONLY or a in ONLY or a == "base")]
    for tag in arms:
        if tag in out["arms"] and "matched" in out["arms"][tag]:
            print("  %-4s CACHED  %d/%d" % (tag, out["arms"][tag]["matched"],
                                            out["arms"][tag]["counted"]), flush=True)
            if tag == "base":
                base_ids = out["arms"][tag]["ids"]
            continue
        t0 = time.time()
        if tag == "base":
            restore, st = (lambda: None), {"n": 0, "zero_frac": []}
        else:
            restore, st = apply_arm(model, tag, act_rms, None)

        allids, matched, counted, first_div, per_prompt = [], 0, 0, None, []
        for i in range(len(prompts)):
            ids, gaps = greedy(model, pids[i], n_new)
            new = ids[-n_new:]
            allids.append(new)
            theirs = refids[i][:n_new]
            m = sum(1 for a, b in zip(new, theirs) if a == b)
            d = next((k for k, (a, b) in enumerate(zip(new, theirs)) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "matched": m, "of": n_new, "diverges_at": d})
            matched += m
            counted += n_new
        restore()

        rec = {"matched": matched, "counted": counted, "agree": matched / float(counted),
               "first_div": first_div, "per_prompt": per_prompt, "ids": allids,
               "n_tensors_converted": st["n"],
               "mean_zero_frac": (sum(st["zero_frac"]) / len(st["zero_frac"])
                                  if st["zero_frac"] else None),
               "organs": list(ARM_ORGANS[tag][1]) + (["lm_head"] if
                                                     tag in ARM_ORGANS and ARM_ORGANS[tag][2]
                                                     else []) if tag != "base" else [],
               "bpb_t2b": BPB.get(tag), "vs_chance": (BPB.get(tag) - CHANCE
                                                      if BPB.get(tag) is not None else None),
               "seconds": time.time() - t0,
               "text": [tok.decode(x) for x in allids]}
        if tag == "base":
            base_ids = allids
            rec["G_L0"] = "FIRES" if matched == counted else "VOID"
        elif tag == "I":
            rec["G_L1_token_identical_to_base"] = (allids == base_ids)
            rec["G_L1"] = "FIRES" if allids == base_ids else "VOID"
        elif tag in ENGINE_REPLICATION and not SMOKE:
            exp = ENGINE_REPLICATION[tag]
            rec["engine_matched"] = exp
            rec["G_L2_within_margin"] = abs(matched - exp) <= MARGIN
        if tag not in ("base", "I"):
            # the bands in s5 are drawn for the FULL 160 positions.  A smoke has a different
            # denominator, so its label would be meaningless -- E16 printed exactly such a
            # label on a 2-sequence smoke and it read as a failing control.
            rec["G_L3"] = label(matched) if not SMOKE else "NOT-APPLICABLE (smoke)"
        out["arms"][tag] = rec

        print("  %-4s %3d/%-3d  %6.2f%%  first-div %-8s  BPB %s (%+.6f)  %s  [%.0fs]"
              % (tag, matched, counted, 100.0 * matched / counted, first_div,
                 ("%.6f" % BPB[tag]) if tag in BPB else "?",
                 (BPB[tag] - CHANCE) if tag in BPB else float("nan"),
                 rec.get("G_L3") or rec.get("G_L0") or rec.get("G_L1") or "", time.time() - t0),
              flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---- verdicts
    A = out["arms"]
    g = {}
    g["G_L0"] = A.get("base", {}).get("G_L0")
    g["G_L1"] = A.get("I", {}).get("G_L1")
    g["G_L2"] = {k: {"measured": A[k]["matched"], "engine": ENGINE_REPLICATION[k],
                     "within_margin": A[k].get("G_L2_within_margin")}
                 for k in ENGINE_REPLICATION if k in A and "matched" in A[k]}
    g["G_L3"] = ({k: A[k]["G_L3"] for k in ("H", "A", "F") if k in A and "G_L3" in A[k]}
                 if not SMOKE else "NOT-APPLICABLE (smoke denominator differs from the bands)")
    g["G_L4_ladder"] = [{"arm": k, "bpb": BPB[k], "vs_chance": BPB[k] - CHANCE,
                         "matched": A[k]["matched"], "of": A[k]["counted"]}
                        for k in ARMS if k in A and "matched" in A[k]]
    out["gates"] = g
    out["void"] = [k for k in ("G_L0", "G_L1") if g.get(k) == "VOID"]
    out["total_seconds"] = time.time() - t_start

    print("\n  bands: floor %d (part A), margin %d (E17), AT-FLOOR <= %d, RANKS >= %d, ceiling %d"
          % (FLOOR, MARGIN, AT_FLOOR_MAX, RANKS_MIN, len(prompts) * n_new), flush=True)
    for k in ("G_L0", "G_L1", "G_L2", "G_L3"):
        print("  %-5s %s" % (k, json.dumps(g.get(k))), flush=True)
    print("\n  ladder (BPB vs chance -> agreement):", flush=True)
    for r in g["G_L4_ladder"]:
        print("    %-4s %+9.6f  ->  %3d/%d" % (r["arm"], r["vs_chance"], r["matched"], r["of"]),
              flush=True)
    print("\n  VOID: %s" % (out["void"] or "none"), flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s  [%.0fs total]" % (OUT, out["total_seconds"]))


if __name__ == "__main__":
    main()
