#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E16 -- the RANKING metric that the brief did not register, run as a CHARACTERISATION.

E14's law, established the day before E16 was written:

    a gate written as a one-sided band inherits the assumption that the treatment hurts;
    compare DISTANCES from the reference, and PAIR EVERY SCORING METRIC WITH A RANKING ONE.

BRIEF_E16_R3_AT_7B.md registers only scoring metrics -- G-R1 through G-R4 are all BPB.  That is
the same omission E14 was written to name, so it is repaired here in the only way that is honest
after the fact: this script REPORTS and decides NOTHING.  No band for it was justified from
anything measured before the run, so promoting it to a gate now would be exactly what E14 s6
forbade and E14 obeyed.  The gap is on the record either way.

Why it matters at this particular cell.  B2 reads 4.017233 BPB, which is BELOW the 4.070106
chance line -- so G-R1's registered label is RULE-FIXES-IT -- but only by 0.052874, while the fp32
arm sits 3.396080 below it.  BPB scores; greedy ranks.  A model that is 0.05 BPB better than
uniform guessing may still choose a different top-1 token essentially everywhere, and the
registered instrument set cannot tell those apart.

The reference is E7's OWN fp32 continuation, already on disk (f32_p*.ids.bin, 2026-09-07), which
reproduced PyTorch 160/160 (E7 s2).  So the reference costs nothing to re-derive and is the same
one E7's planted control was scored against.

  reference   qwen25-coder7b_f32.bin        E7's stored greedy ids
  B1          qwen25-coder7b_p.bin          R0, fold layers   -- E7 scored it 0/160
  B2          qwen25-coder7b_p_r3.bin       R3, fold layers
  B3          qwen25-coder7b_p_r3_nofold.bin R3, fold none

PROMPTS and N_NEW are imported from e6_generate -- one definition, never re-derived.

Env: D_THREADS (default 6)
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW, run, parse_gen   # noqa: E402

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"
E7 = r"D:\_ktmp\e7"
TMP = r"D:\_ktmp\e16g"
OUT = os.path.join(HERE, "results", "e16_greedy_rank.json")
MODEL = "Qwen/Qwen2.5-Coder-7B"

ARMS = [
    ("B1", os.path.join(E7, "qwen25-coder7b_p.bin"),            "R0, fold layers -- E7's planted control, 0/160"),
    ("B2", os.path.join(E7, "qwen25-coder7b_p_r3.bin"),         "R3, fold layers -- BPB 4.017233, -0.052874 vs chance"),
    ("B3", os.path.join(E7, "qwen25-coder7b_p_r3_nofold.bin"),  "R3, fold none"),
]
THREADS = os.environ.get("D_THREADS", "6")


def main():
    from transformers import AutoTokenizer
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tk = AutoTokenizer.from_pretrained(MODEL)

    # E7's stored fp32 continuations are the reference.  Re-derive the prompt ids here so the
    # prefix length is known, and check the stored file starts with them -- otherwise the file
    # is not the continuation of the prompt we think it is.
    ref, ids_paths = {}, []
    for i, p in enumerate(PROMPTS):
        pids = tk(p)["input_ids"]
        path = os.path.join(TMP, "p%d.bin" % i)
        with open(path, "wb") as f:
            f.write(struct.pack("<%di" % len(pids), *pids))
        ids_paths.append(path)

        raw = open(os.path.join(E7, "f32_p%d.ids.bin" % i), "rb").read()
        got = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        assert got[:len(pids)] == pids, \
            "prompt %d: E7's stored ids do not start with this prompt's tokens" % i
        assert len(got) == len(pids) + N_NEW, (len(got), len(pids), N_NEW)
        ref[i] = got[-N_NEW:]

    print("E16 greedy -- CHARACTERISATION, not a gate (see this file's docstring)", flush=True)
    print("reference: E7's fp32 continuation, %d prompts x %d new tokens = %d positions"
          % (len(PROMPTS), N_NEW, len(PROMPTS) * N_NEW), flush=True)

    out = {"brief": "briefs/BRIEF_E16_R3_AT_7B.md -- NOT a registered gate",
           "not_a_gate": "E14 s6's rule: no band for this was justified before the run, so no "
                         "verdict is read off it",
           "model": MODEL, "n_new": N_NEW, "prompts": PROMPTS, "threads": int(THREADS),
           "reference": "E7 f32_p*.ids.bin (qwen25-coder7b_f32.bin, 160/160 vs PyTorch)",
           "arms": {}}

    for tag, w, note in ARMS:
        if not os.path.exists(w):
            print("  %-3s MISSING %s -- skipped" % (tag, w), flush=True)
            continue
        tot = match = 0
        first_div = None
        per_prompt = []
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (tag, i))
            r = parse_gen(run([ENGINE, "--weights", w, "--threads", THREADS,
                               "--generate", ids_paths[i], str(N_NEW), pfx]))
            ours = r["ids"][-N_NEW:]
            m = sum(1 for a, b in zip(ours, ref[i]) if a == b)
            for k, (a, b) in enumerate(zip(ours, ref[i])):
                if a != b:
                    if first_div is None:
                        first_div = [i, k]
                    break
            per_prompt.append({"prompt": i, "matched": m, "of": N_NEW,
                               "diverges_at": next((k for k, (a, b) in
                                                    enumerate(zip(ours, ref[i])) if a != b), None)})
            match += m
            tot += N_NEW
            for ext in (".prefill.bin", ".ids.bin"):
                if os.path.exists(pfx + ext):
                    os.remove(pfx + ext)
        out["arms"][tag] = {"weights": w, "note": note, "agree": match / float(tot),
                            "matched": match, "counted": tot, "first_div": first_div,
                            "per_prompt": per_prompt}
        print("  %-3s %5.1f%%  (%d/%d)  first divergence %s   %s"
              % (tag, 100.0 * match / tot, match, tot, first_div, note), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    print("\n  This is a characterisation. No verdict is read off it, and G-R1's registered "
          "label stands as registered.", flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
