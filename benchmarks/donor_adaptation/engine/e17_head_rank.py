#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E17 -- does the ternary HEAD rank?

Pre-registration: docs/research/donor_adaptation/briefs/BRIEF_E17_DOES_THE_HEAD_RANK.md,
pushed before any arm ran (commit cf33251).

Every E16 arm was exported --head-ternary (e16_r3_at_7b.py:167).  On Qwen2.5-Coder-7B the
embeddings are UNTIED, so that flag ternarizes a standalone 152064 x 3584 = 545M-parameter output
projection -- the one tensor whose entire job is ranking.  E1 s4.3 says that tensor is nearly FREE
in BPB (-0.008546 at 1.5B, +0.022070 at 0.5B).  No probe has ever run a head-fp32 arm through
--generate at any scale, so the pairing E14's law demands has never been available for it.

Nothing is exported here.  Every artifact is on disk from E1; every reference is on disk from E6.

  H0a  qwen25-05b_f32.bin    known-positive, MUST fire 160/160 or E17 is VOID   (E6 A1)
  H0b  qwen25-15b_tqh.bin    known-negative, MUST replicate 10/160 or VOID      (E6 A3 == E16 C0)
  H0c  qwen25-15b_f32.bin    known-positive at the second scale -- a new cell
  H1   qwen25-15b_tq.bin     the head axis at 1.5B, vs H0b
  H2   qwen25-05b_tq.bin     the head axis at 0.5B, vs E6 A2

G-H0d additionally re-runs H0b under the E13 binary that E16 used, because E17 compares numbers
across the two builds and they are not the same file.

PROMPTS and N_NEW are imported from e6_generate -- one definition, never re-derived.

Env: D_THREADS (default 6)
"""
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e6_generate import PROMPTS, N_NEW, run, parse_gen   # noqa: E402

ENGINE = os.path.join(HERE, "donor_engine.exe")        # the build E6 used
ENGINE_E13 = r"D:\_ktmp\e13\donor_engine.exe"          # the build E16 used
E1 = r"D:\_ktmp\e1"
TMP = r"D:\_ktmp\e17"
E6RES = os.path.join(HERE, "results", "e6")
OUT = os.path.join(HERE, "results", "e17_head_rank.json")
THREADS = os.environ.get("D_THREADS", "6")

Q05 = "Qwen/Qwen2.5-0.5B"
Q15 = "Qwen/Qwen2.5-1.5B"

# tag, weights, hf model, role, the E6 number it must replicate (matched/160) or None
ARMS = [
    ("H0a", os.path.join(E1, "qwen25-05b_f32.bin"),  Q05, "known-positive (fp32 0.5B)", 160),
    ("H0b", os.path.join(E1, "qwen25-15b_tqh.bin"),  Q15, "known-negative (E6 A3 == E16 C0)", 10),
    ("H0c", os.path.join(E1, "qwen25-15b_f32.bin"),  Q15, "known-positive at 1.5B -- NEW cell", None),
    ("H1",  os.path.join(E1, "qwen25-15b_tq.bin"),   Q15, "head fp32, ternary body -- 1.5B", None),
    ("H2",  os.path.join(E1, "qwen25-05b_tq.bin"),   Q05, "head fp32, ternary body -- 0.5B", None),
]

# BPB, E1 engine column, s2.1 / s2.2 -- quoted, never re-measured here
E1_BPB = {"H0b": 3.475706691632780, "H1": 3.484253077409102,
          "A2": 4.531233733626962, "H2": 4.509163909048454}

# brief s5: derived from the measured known-negative ceiling and the known-positive band
NEG_CEILING = 10 / 160.0                      # E6 A3, the worst of five ternary readings
POS_BAND = 1.0                                # E6 A1 and E7 fp32, both exactly 160/160
MECHANISM_BAR = NEG_CEILING + 0.5 * (POS_BAND - NEG_CEILING)     # 0.53125 == 85/160
NOT_MECHANISM_BAR = 2.0 * NEG_CEILING                            # 0.125   == 20/160

# brief s3: a head-axis pair may differ on these and on NOTHING else
AXIS_FREE = ("head_ternary", "rule_applied_to", "tied",
             "bytes", "sha256", "mean_ternary_zero_fraction")


def sidecar(path):
    return json.load(open(path + ".json", encoding="utf-8"))


def check_axis(tq_path, tqh_path):
    """The registered contrast is head_ternary alone.  Returns (ok, bad) where bad names every
    field that moved and was not admitted by the brief."""
    a, b = sidecar(tq_path), sidecar(tqh_path)
    bad = []
    for k in sorted(set(a) | set(b)):
        if k in AXIS_FREE:
            continue
        if a.get(k) != b.get(k):
            bad.append((k, a.get(k), b.get(k)))
    # and the admitted fields must have moved in the registered DIRECTION, not just moved
    if not (a.get("head_ternary") is False and b.get("head_ternary") is True):
        bad.append(("head_ternary", a.get("head_ternary"), b.get("head_ternary")))
    ra, rb = a.get("rule_applied_to") or [], b.get("rule_applied_to") or []
    if sorted(set(rb) - set(ra)) != ["lm_head"] or set(ra) - set(rb):
        bad.append(("rule_applied_to", ra, rb))
    return (not bad), bad


def label(agree):
    if agree >= MECHANISM_BAR:
        return "HEAD-IS-THE-MECHANISM"
    if agree <= NOT_MECHANISM_BAR:
        return "HEAD-IS-NOT-THE-MECHANISM"
    return "INDETERMINATE"


def main():
    from transformers import AutoTokenizer
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    ref = json.load(open(os.path.join(E6RES, "ref.json"), encoding="utf-8"))
    e6eng = json.load(open(os.path.join(E6RES, "engine.json"), encoding="utf-8"))
    assert e6eng["n_new"] == N_NEW, (e6eng["n_new"], N_NEW)
    assert e6eng["prompts"] == PROMPTS, "E6's stored prompts are not the imported PROMPTS"

    out = {"brief": "briefs/BRIEF_E17_DOES_THE_HEAD_RANK.md (cf33251)",
           "engine": ENGINE, "engine_e16": ENGINE_E13, "threads": int(THREADS),
           "n_new": N_NEW, "prompts": PROMPTS,
           "reference": "results/e6/ref.json -- PyTorch fp32 greedy, eager, threads 6",
           "bands": {"neg_ceiling": NEG_CEILING, "pos_band": POS_BAND,
                     "mechanism_bar": MECHANISM_BAR,
                     "not_mechanism_bar": NOT_MECHANISM_BAR},
           "arms": {}}

    # ---- brief s9: re-derive the prompt ids under EACH arm's own tokenizer and check they are
    # the ids E6 stored.  E6 tokenized once, with the 0.5B tokenizer, and used those ids for the
    # 1.5B arm too; both are Qwen2.5 at V=151936 but that is checked here, not inherited.
    ids_paths = {}
    for hf in (Q05, Q15):
        tk = AutoTokenizer.from_pretrained(hf)
        paths = []
        for i, p in enumerate(PROMPTS):
            pids = tk(p)["input_ids"]
            assert pids == e6eng["prompt_ids"][i], \
                "prompt %d tokenizes differently under %s than E6 stored" % (i, hf)
            path = os.path.join(TMP, "p%d.bin" % i)
            with open(path, "wb") as f:
                f.write(struct.pack("<%di" % len(pids), *pids))
            paths.append(path)
        ids_paths[hf] = paths
    out["prompt_ids_cross_tokenizer"] = "IDENTICAL under 0.5B and 1.5B tokenizers, and equal to E6"
    print("prompt ids: identical under both tokenizers and equal to E6's stored ids", flush=True)

    # ---- the single-axis gate, before any agreement is read
    axis = {}
    for tag, tq, tqh in (("1.5B", os.path.join(E1, "qwen25-15b_tq.bin"),
                                  os.path.join(E1, "qwen25-15b_tqh.bin")),
                         ("0.5B", os.path.join(E1, "qwen25-05b_tq.bin"),
                                  os.path.join(E1, "qwen25-05b_tqh.bin"))):
        ok, bad = check_axis(tq, tqh)
        axis[tag] = {"single_axis": ok, "unadmitted_moves": bad}
        print("  axis gate %-5s %s%s" % (tag, "PASS" if ok else "FAIL",
                                         "" if ok else "  moved: %s" % bad), flush=True)
    out["axis_gate"] = axis
    if not all(v["single_axis"] for v in axis.values()):
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("E17 ABORTED: a head-axis pair differs on more than head_ternary")

    # ---- the arms
    def score(tag, weights, hf, engine):
        tot = match = 0
        first_div = None
        per_prompt = []
        rr = {x["prompt"]: x for x in ref[hf]}
        allids = []
        for i in range(len(PROMPTS)):
            pfx = os.path.join(TMP, "%s_p%d" % (tag, i))
            r = parse_gen(run([engine, "--weights", weights, "--threads", THREADS,
                               "--generate", ids_paths[hf][i], str(N_NEW), pfx]))
            ours = r["ids"][-N_NEW:]
            theirs = rr[i]["ids"][-N_NEW:]
            allids.append(ours)
            m = sum(1 for a, b in zip(ours, theirs) if a == b)
            d = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
            if d is not None and first_div is None:
                first_div = [i, d]
            per_prompt.append({"prompt": i, "matched": m, "of": N_NEW, "diverges_at": d,
                               "decode_toks": r.get("decode_toks")})
            match += m
            tot += N_NEW
            for ext in (".prefill.bin", ".ids.bin"):
                if os.path.exists(pfx + ext):
                    os.remove(pfx + ext)
        return {"weights": weights, "hf": hf, "agree": match / float(tot), "matched": match,
                "counted": tot, "first_div": first_div, "per_prompt": per_prompt,
                "ids_sha": hashlib.sha256(
                    json.dumps(allids).encode()).hexdigest()[:16]}

    for tag, w, hf, role, expect in ARMS:
        if not os.path.exists(w):
            print("  %-4s MISSING %s -- skipped" % (tag, w), flush=True)
            continue
        try:
            rec = score(tag, w, hf, ENGINE)
        except SystemExit as e:
            # brief s5: a load failure on H0c is a FINDING about the fp32 path at 1.5B (E1 s4.4
            # records the engine once refusing >2 GB), not a reason to discard the ternary arms.
            out["arms"][tag] = {"weights": w, "hf": hf, "role": role, "failed": str(e)}
            print("  %-4s FAILED: %s" % (tag, e), flush=True)
            json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
            continue
        rec["role"] = role
        rec["e6_expected_matched"] = expect
        if expect is not None:
            rec["replicates_e6"] = (rec["matched"] == expect)
        out["arms"][tag] = rec
        print("  %-4s %6.2f%%  (%3d/%d)  first-div %-8s  %s%s"
              % (tag, 100.0 * rec["agree"], rec["matched"], rec["counted"],
                 rec["first_div"], role,
                 "" if expect is None else
                 ("   [E6 %d/160 %s]" % (expect, "REPLICATED" if rec["replicates_e6"] else "MISMATCH"))),
              flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---- G-H0d: are E6's binary and E16's binary the same instrument?
    if os.path.exists(ENGINE_E13) and "ids_sha" in out["arms"].get("H0b", {}):
        alt = score("H0b_e13", os.path.join(E1, "qwen25-15b_tqh.bin"), Q15, ENGINE_E13)
        same = (alt["ids_sha"] == out["arms"]["H0b"]["ids_sha"])
        out["G_H0d"] = {"binary": ENGINE_E13, "token_identical": bool(same),
                        "agree": alt["agree"], "matched": alt["matched"],
                        "verdict": "COMPARABLE" if same else "NOT-COMPARABLE"}
        print("  G-H0d  E13 build on H0b: %d/160, token-identical to E6 build: %s"
              % (alt["matched"], same), flush=True)

    # ---- verdicts
    A = out["arms"]
    gates = {}
    gates["G_H0a"] = {"agree": A.get("H0a", {}).get("agree"), "required": 1.0,
                      "verdict": "FIRES" if A.get("H0a", {}).get("matched") == 160 else "VOID"}
    gates["G_H0b"] = {"matched": A.get("H0b", {}).get("matched"), "required": 10,
                      "first_div": A.get("H0b", {}).get("first_div"),
                      "verdict": "REPLICATED" if (A.get("H0b", {}).get("matched") == 10 and
                                                  A.get("H0b", {}).get("first_div") == [0, 0]) else "VOID"}
    if "matched" in A.get("H0c", {}):
        gates["G_H0c"] = {"agree": A["H0c"]["agree"], "matched": A["H0c"]["matched"],
                          "verdict": "FIRES" if A["H0c"]["matched"] == 160 else "DOES-NOT-FIRE"}
    if "matched" in A.get("H1", {}):
        gates["G_H1"] = {"agree": A["H1"]["agree"], "verdict": label(A["H1"]["agree"]),
                         "vs_H0b_points": 100.0 * (A["H1"]["agree"] - A["H0b"]["agree"])}
    if "matched" in A.get("H2", {}):
        a2 = 3 / 160.0                                   # E6 A2, quoted
        gates["G_H2"] = {"agree": A["H2"]["agree"], "verdict": label(A["H2"]["agree"]),
                         "vs_A2_points": 100.0 * (A["H2"]["agree"] - a2)}
    gates["G_H3"] = {
        "1.5B": {"dBPB_head": E1_BPB["H0b"] - E1_BPB["H1"],
                 "dAgree_points_head": (100.0 * (A["H0b"]["agree"] - A["H1"]["agree"])
                                        if "matched" in A.get("H1", {}) else None)},
        "0.5B": {"dBPB_head": E1_BPB["A2"] - E1_BPB["H2"],
                 "dAgree_points_head": (100.0 * (3 / 160.0 - A["H2"]["agree"])
                                        if "matched" in A.get("H2", {}) else None)},
        "note": "dBPB and dAgree are both (head-ternary MINUS head-fp32); BPB is E1's engine "
                "column, never re-measured here"}
    out["gates"] = gates

    void = [k for k in ("G_H0a", "G_H0b") if gates[k]["verdict"] == "VOID"]
    out["verdict"] = "VOID (%s)" % ",".join(void) if void else gates.get("G_H1", {}).get("verdict")
    out["stage2_runs"] = (not void) and gates.get("G_H1", {}).get("verdict") == "HEAD-IS-THE-MECHANISM"

    print("\n  bands: neg ceiling %.4f, pos band %.1f -> mechanism >= %.5f (85/160), "
          "not-mechanism <= %.4f (20/160)"
          % (NEG_CEILING, POS_BAND, MECHANISM_BAR, NOT_MECHANISM_BAR), flush=True)
    for k in sorted(gates):
        print("  %-6s %s" % (k, json.dumps(gates[k])), flush=True)
    print("\n  VERDICT: %s" % out["verdict"], flush=True)
    print("  stage 2 (7B head-fp32 export) runs: %s" % out["stage2_runs"], flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
