#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E14 G-N3 -- the greedy characterisation, which the BPB runner never implemented.

BRIEF_E14_ACTIVATION_INT8_COST.md section 4 registers G-N3 as "--generate, 160 tokens, the five
frozen prompts, A1/A2 vs A0", and section 6 fixes what it may be used for:

    "G-N3 therefore compares A1/A2 against A0 -- the same ternary weights with fp32 activations --
     and reports the agreement as a characterisation.  It is NOT a gate, and no verdict is read
     off it, because there is no pre-registered band that could be justified from anything
     measured."

That constraint is respected here.  This script reports; it decides nothing.  It exists because
e14_activation_cost.py runs --bpb only, so G-N3 was pre-registered and then never run.

Why it is worth running anyway.  At the 1.5B verdict cell the model sits 0.625503 BELOW the chance
line -- BPB means what it usually means there -- and dBPB(A1) came out NEGATIVE: quantizing the
activations to int8 IMPROVED bits-per-byte.  A negative delta is outside the regime section 4's
one-sided bands were drawn for, and BPB alone cannot separate "costs nothing" from "softened the
logits and moved the metric".  Greedy is rank-based: temperature-like softening leaves the argmax
alone, so agreement near 100% and a BPB gain together point at calibration, while a BPB gain with
degraded agreement points at the metric moving on its own.  Neither reading is a verdict.  Both
are worth having on the record.

The prompts, N_NEW and the parsing all come from e6_generate -- one definition, imported, not
re-derived.

Env: D_THREADS (default 6), E14_G3_CELL (default 1.5B; "0.5B" also available)
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW, run, parse_gen   # noqa: E402

ENGINE = r"D:\_ktmp\e13\donor_engine.exe"      # the E13 build, same binary the BPB arms used
TMP = r"D:\_ktmp\e14g3"
OUT = os.path.join(HERE, "results", "e14_gn3_greedy.json")

CELLS = {
    "0.5B": (r"D:\_ktmp\e1\qwen25-05b_tqh.bin", "Qwen/Qwen2.5-0.5B"),
    "1.5B": (r"D:\_ktmp\e1\qwen25-15b_tqh.bin", "Qwen/Qwen2.5-1.5B"),
}
ARMS = [
    ("A0", [],                             "fp32 activations -- the reference for this gate"),
    ("A1", ["--lut"],                      "int8 activations, ONE scale per vector"),
    ("A2", ["--lut", "--lut-group", "32"], "int8 activations, one scale per 32 channels"),
    ("A3", ["--lutblk"],                   "E13 blocked layout -- must equal A1 exactly"),
]

THREADS = os.environ.get("D_THREADS", "6")
CELL = os.environ.get("E14_G3_CELL", "1.5B")


def main():
    from transformers import AutoTokenizer
    weights, hf = CELLS[CELL]
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    tk = AutoTokenizer.from_pretrained(hf)
    ids_paths = []
    for i, p in enumerate(PROMPTS):
        ids = tk(p)["input_ids"]
        pth = os.path.join(TMP, "p%d.bin" % i)
        with open(pth, "wb") as f:
            f.write(struct.pack("<%di" % len(ids), *ids))
        ids_paths.append(pth)

    print("E14 G-N3  cell %s  %s  %d prompts x %d new tokens"
          % (CELL, os.path.basename(weights), len(PROMPTS), N_NEW), flush=True)

    out = {"brief": "briefs/BRIEF_E14_ACTIVATION_INT8_COST.md section 4 (G-N3), section 6",
           "cell": CELL, "weights": weights, "model": hf, "n_new": N_NEW,
           "prompts": PROMPTS, "threads": int(THREADS),
           "not_a_gate": "section 6: no verdict is read off G-N3", "arms": {}}

    for tag, flags, note in ARMS:
        runs = []
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (tag, i))
            r = parse_gen(run([ENGINE, "--weights", weights, "--threads", THREADS,
                               "--generate", ids_paths[i], str(N_NEW), pfx] + flags))
            for ext in (".prefill.bin", ".ids.bin"):
                if os.path.exists(pfx + ext):
                    os.remove(pfx + ext)
            runs.append({"prompt": i, "ids": r["ids"]})
        out["arms"][tag] = runs
        print("  %-3s done  %s" % (tag, note), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    ref = {x["prompt"]: x["ids"][-N_NEW:] for x in out["arms"]["A0"]}
    print("\n---- agreement against A0 (fp32 activations, identical ternary weights) ----",
          flush=True)
    out["agreement_vs_A0"] = {}
    for tag in ("A1", "A2", "A3"):
        tot = match = 0
        first_div = None
        per_prompt = []
        for x in out["arms"][tag]:
            ours, theirs = x["ids"][-N_NEW:], ref[x["prompt"]]
            m = sum(1 for a, b in zip(ours, theirs) if a == b)
            for k, (a, b) in enumerate(zip(ours, theirs)):
                if a != b and first_div is None:
                    first_div = [x["prompt"], k]
                    break
            per_prompt.append({"prompt": x["prompt"], "matched": m, "of": N_NEW})
            match += m
            tot += N_NEW
        out["agreement_vs_A0"][tag] = {"agree": match / float(tot), "matched": match,
                                       "counted": tot, "first_div": first_div,
                                       "per_prompt": per_prompt}
        print("  %-3s %5.1f%%  (%d/%d)  first divergence %s"
              % (tag, 100.0 * match / tot, match, tot, first_div), flush=True)

    a3 = out["agreement_vs_A0"]["A3"]["matched"]
    a1 = out["agreement_vs_A0"]["A1"]["matched"]
    ident = all(x["ids"] == y["ids"] for x, y in zip(out["arms"]["A1"], out["arms"]["A3"]))
    out["A3_equals_A1_token_for_token"] = ident
    print("\n  A3 vs A1 token-for-token identical: %s  (%d vs %d matched against A0)"
          % (ident, a1, a3), flush=True)
    print("\n  Section 6 stands: this is a characterisation, not a gate. No verdict is read "
          "off it.", flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
