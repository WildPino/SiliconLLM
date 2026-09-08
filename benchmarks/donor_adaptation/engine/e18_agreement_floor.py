#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E18 part A -- the frequency-coincidence FLOOR for greedy agreement.

E17 s7 recorded the hole this fills, and named it the cheapest open item in the programme:

    There is no measured floor for agreement by frequency coincidence.  These arms emit " the"
    and "\\n" repeatedly, and PyTorch also emits those tokens sometimes.  12/160 therefore CANNOT
    be claimed to be above zero information.

Every agreement number in E6, E14, E16 and E17 is quoted against an implicit floor of zero, which
is the floor for a UNIFORM guesser (160/151936 ~ 0.001 expected matches) and NOT the floor for a
degenerate model that emits high-frequency tokens.  The arms in question emit exactly those.

This computes the floor from the reference continuations alone.  No engine, no weights, no
tokenizer beyond decoding for the report -- so it cannot be contaminated by the thing it is
measuring.

  FLOOR_CONST   the best CONSTANT predictor: max over tokens t of #{positions where ref == t}.
                This is the score of the strongest possible zero-information model, and it is an
                UPPER bound on the floor: no real degenerate arm can do better without knowing
                something.
  FLOOR_ARM     what each measured arm's OWN most-frequent emitted token would have scored as a
                constant predictor -- i.e. how much of that arm's agreement is explained by it
                having landed on a common token.

Reference: results/e6/ref.json (PyTorch fp32 greedy), the same file every agreement in the
programme is scored against.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW   # noqa: E402

E6RES = os.path.join(HERE, "results", "e6")
OUT = os.path.join(HERE, "results", "e18_agreement_floor.json")

# the measured arms, and where their emitted ids live
#   tag -> (hf model whose reference it is scored against, source, key, matched/160)
ARMS = [
    ("A1  0.5B fp32",        "Qwen/Qwen2.5-0.5B", "e6", "A1", 160),
    ("A2  0.5B TQH",         "Qwen/Qwen2.5-0.5B", "e6", "A2", 3),
    ("A3  1.5B TQH",         "Qwen/Qwen2.5-1.5B", "e6", "A3", 10),
]


def main():
    from transformers import AutoTokenizer
    ref = json.load(open(os.path.join(E6RES, "ref.json"), encoding="utf-8"))
    e6 = json.load(open(os.path.join(E6RES, "engine.json"), encoding="utf-8"))

    out = {"source": "results/e6/ref.json",
           "note": "no engine run; computed from the reference continuations alone",
           "n_positions": len(PROMPTS) * N_NEW, "models": {}, "arms": {}}

    tks = {}
    for hf, runs in sorted(ref.items()):
        tk = tks.setdefault(hf, AutoTokenizer.from_pretrained(hf))
        # the 160 tokens the reference actually emits
        pos = []
        for r in runs:
            pos.extend(r["ids"][-N_NEW:])
        assert len(pos) == len(PROMPTS) * N_NEW, len(pos)
        c = collections.Counter(pos)
        top = c.most_common(5)
        best_tok, best_n = top[0]
        out["models"][hf] = {
            "n_positions": len(pos),
            "distinct_tokens": len(c),
            "floor_const_matched": best_n,
            "floor_const_frac": best_n / float(len(pos)),
            "floor_const_token_id": best_tok,
            "floor_const_token": tk.decode([best_tok]),
            "top5": [{"id": t, "tok": tk.decode([t]), "n": n} for t, n in top],
            "uniform_expected_matches": len(pos) / 151936.0,
        }
        print("%-22s %d positions, %d distinct.  FLOOR_CONST = %d/%d = %.2f%%  on %r"
              % (hf, len(pos), len(c), best_n, len(pos),
                 100.0 * best_n / len(pos), tk.decode([best_tok])), flush=True)
        for t, n in top:
            print("      %8d  %-12r %3d" % (t, tk.decode([t]), n), flush=True)

    # ---- what each arm's own dominant token would have scored
    for label, hf, src, key, matched in ARMS:
        assert src == "e6"
        runs = e6["arms"][key]["runs"]
        emitted = []
        for r in runs:
            emitted.extend(r["ids"][-N_NEW:])
        refpos = []
        for r in ref[hf]:
            refpos.extend(r["ids"][-N_NEW:])
        ce = collections.Counter(emitted)
        dom, dom_n = ce.most_common(1)[0]
        # how a constant predictor of THAT token would have scored
        as_const = sum(1 for t in refpos if t == dom)
        tk = tks[hf]
        out["arms"][key] = {
            "label": label, "hf": hf, "matched": matched,
            "dominant_token_id": dom, "dominant_token": tk.decode([dom]),
            "dominant_emitted_n": dom_n,
            "dominant_as_constant_predictor": as_const,
            "excess_over_own_dominant_const": matched - as_const,
            "floor_const_matched": out["models"][hf]["floor_const_matched"],
            "excess_over_floor_const": matched - out["models"][hf]["floor_const_matched"],
        }
        print("  %-22s emits %r %d/160 times; that token as a CONSTANT predictor scores %d/160; "
              "arm scored %d/160  (excess over best-constant floor %+d)"
              % (label, tk.decode([dom]), dom_n, as_const, matched,
                 matched - out["models"][hf]["floor_const_matched"]), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
