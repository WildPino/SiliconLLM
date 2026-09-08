#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E18 part C -- the SPEED axis of the organ ladder, derived from measured quantities only.

Part B asks how much ternarization a model that still ranks can survive.  That question only
matters if some rung of the ladder is also fast enough.  This computes what each rung would cost,
so quality and speed are read off the same ladder.

EVERY input is a measured number with a citation.  Nothing here is a new measurement, and no
timing is taken -- this is arithmetic over the ledger, and it is labelled as such.

  packed kernel ceiling   ~53 G-weights/s, engine at 51.7-52.2   ledger s23.2 / s23.3 (E10)
  fp32 stream             34.75 GB/s                             ledger s26 (E13, same sweep)
  bytes per ternary weight  VERIFIED from an artifact below, not assumed

BYTE CONVENTION (the standing law: charged bytes are not moved bytes, and a ceiling is a
denominator).  Everything below is in MOVED bytes: 0.5 B/weight packed, 4 B/weight fp32, against
25.5 GB/s and 34.75 GB/s which are both moved-byte rates from the same sweeps.

The embedding table is NOT counted as streamed: it is a row lookup, one row per token.  The
output head IS counted -- it is a full matvec over the vocabulary every step.
"""
import json
import os

E1 = r"D:\_ktmp\e1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "e18_ladder_bandwidth.json")

# ---- measured, cited ------------------------------------------------------------------
PACKED_GW_S = 53.0e9      # ledger s23.3: the packed kernel's own ceiling, engine at 51.7-52.2
PACKED_GB_S = 25.5e9      # ledger s23.3, same row
FP32_GB_S = 34.75e9       # ledger s26 (E13): the fp32 arm in the same sweep lutblk was read in
MEASURED_7B_TOKS = 6.79   # ledger s19/s22, the standing rate on the real donor
TARGET_GOOD, TARGET_GREAT = 50.0, 100.0


def qwen_shapes(D, F, L, n_heads, n_kv, head_dim, V, tied):
    q_out, kv_out = n_heads * head_dim, n_kv * head_dim
    per_layer = {"q_proj": q_out * D, "k_proj": kv_out * D, "v_proj": kv_out * D,
                 "o_proj": D * q_out, "gate_proj": F * D, "up_proj": F * D, "down_proj": D * F}
    attn = sum(per_layer[k] for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
    ffn = sum(per_layer[k] for k in ("gate_proj", "up_proj", "down_proj"))
    rows = (q_out + kv_out + kv_out + D + F + F + D)
    return {"per_layer": per_layer, "attn": attn * L, "ffn": ffn * L,
            "head": V * D, "rows_body": rows * L, "rows_head": V,
            "embed": V * D, "tied": tied, "L": L, "D": D, "F": F, "V": V}


def verify_bytes_per_weight():
    """The packed format's cost per weight, read off a real artifact rather than assumed."""
    path = os.path.join(E1, "qwen25-15b_tqh.bin")
    if not os.path.exists(path):
        return None
    s = qwen_shapes(1536, 8960, 28, 12, 2, 128, 151936, tied=False)
    tern = s["attn"] + s["ffn"] + s["head"]          # TQH converts body + head
    rows = s["rows_body"] + s["rows_head"]
    size = os.path.getsize(path)
    embed_b = s["embed"] * 4
    scales_b = rows * 4
    norms_b = (2 * s["L"] * s["D"] + s["D"]) * 4
    biases_b = s["L"] * (12 * 128 + 2 * 128 + 2 * 128) * 4
    header_b = 8 + 9 * 4 + 2 * 4
    codes_b = size - embed_b - scales_b - norms_b - biases_b - header_b
    return {"artifact": path, "file_bytes": size, "ternary_weights": tern,
            "embed_bytes": embed_b, "scale_bytes": scales_b, "norm_bytes": norms_b,
            "bias_bytes": biases_b, "header_bytes": header_b, "code_bytes": codes_b,
            "bytes_per_weight": codes_b / float(tern),
            "bits_per_weight": 8.0 * codes_b / float(tern)}


def main():
    out = {"note": "DERIVATION over measured quantities; no timing taken, none quotable",
           "convention": "moved bytes throughout: 0.5 B/weight packed, 4 B/weight fp32, "
                         "against 25.5 and 34.75 GB/s from the same sweeps",
           "inputs": {"packed_G_weights_per_s": PACKED_GW_S, "packed_GB_s": PACKED_GB_S,
                      "fp32_GB_s": FP32_GB_S, "measured_7B_toks": MEASURED_7B_TOKS,
                      "citations": {"packed ceiling": "ledger s23.2/s23.3 (E10)",
                                    "fp32 stream": "ledger s26 (E13), same sweep",
                                    "6.79 tok/s": "ledger s19/s22"}}}

    v = verify_bytes_per_weight()
    out["bytes_per_weight_verified"] = v
    if v:
        print("bytes/weight verified from %s:" % os.path.basename(v["artifact"]))
        print("  file %d B, %d ternary weights -> codes %d B = %.6f B/weight = %.4f bits"
              % (v["file_bytes"], v["ternary_weights"], v["code_bytes"],
                 v["bytes_per_weight"], v["bits_per_weight"]))
    BW_TERN = v["bytes_per_weight"] if v else 0.5
    BW_FP32 = 4.0
    tern_w_s = PACKED_GB_S / BW_TERN
    fp32_w_s = FP32_GB_S / BW_FP32
    out["derived_rates"] = {"ternary_weights_per_s": tern_w_s, "fp32_weights_per_s": fp32_w_s,
                            "fp32_weight_costs_x_ternary": tern_w_s / fp32_w_s}
    print("\n  ternary %.2f G-w/s   fp32 %.2f G-w/s   -> an fp32 weight costs %.2fx a ternary one"
          % (tern_w_s / 1e9, fp32_w_s / 1e9, tern_w_s / fp32_w_s))

    # ---- the active-weight budget the goal implies
    # The ledger pairs "~25.5 GB/s" with "~53 G-w/s" in ONE row (s23.3), but 25.5 / 0.5 = 51.0:
    # its own two companion numbers disagree by 4%.  Rather than silently pick one, the budget is
    # reported across the whole measured span -- the kernel bench range from s23.2, the engine's
    # own two organs, and the 51.0 derived here from the GB/s figure and the verified 0.5 B/weight.
    RATE_SPAN = {"kernel bench low (s23.2)": 49.11e9,
                 "derived: 25.5 GB/s / 0.5 B-per-weight": tern_w_s,
                 "engine gate+up (s23.2)": 51.66e9,
                 "engine down (s23.2)": 52.22e9,
                 "kernel bench high (s23.2)": 52.95e9,
                 "ledger rounded companion (s23.3)": PACKED_GW_S}
    out["rate_span_note"] = ("ledger s23.3 pairs ~25.5 GB/s with ~53 G-w/s, but 25.5/0.5 = 51.0; "
                             "the two companion numbers disagree by 4%, so the budget is given "
                             "across the measured span rather than at one rounded point")
    out["inputs"]["rate_span_G_w_s"] = {k: v / 1e9 for k, v in RATE_SPAN.items()}
    out["budget"] = {}
    lo, hi = min(RATE_SPAN.values()), max(RATE_SPAN.values())
    print("\n  NOTE: ledger s23.3 pairs ~25.5 GB/s with ~53 G-w/s, but 25.5/0.5 = 51.0 --")
    print("        its own companion numbers disagree by 4%. Budget across the measured span:")
    for name, t in (("good_50", TARGET_GOOD), ("great_100", TARGET_GREAT)):
        out["budget"][name] = {
            "tok_s": t,
            "active_ternary_weights_low": lo / t, "active_ternary_weights_high": hi / t,
            "share_of_a_10B_model_low": lo / t / 10e9,
            "share_of_a_10B_model_high": hi / t / 10e9,
            "per_rate": {k: {"weights": r / t, "share_of_10B": r / t / 10e9}
                         for k, r in sorted(RATE_SPAN.items())}}
        print("    %5.0f tok/s -> %.3f - %.3f G active ternary weights/token"
              "  = %.1f%% - %.1f%% of a 10 B model"
              % (t, lo / t / 1e9, hi / t / 1e9, 100.0 * lo / t / 10e9, 100.0 * hi / t / 10e9))

    # ---- the ladder, on the 7B the programme actually quotes
    s7 = qwen_shapes(3584, 18944, 28, 28, 4, 128, 152064, tied=False)
    active = s7["attn"] + s7["ffn"] + s7["head"]
    out["coder7b"] = {"attn": s7["attn"], "ffn": s7["ffn"], "head": s7["head"],
                      "active_per_token": active,
                      "shares": {"ffn": s7["ffn"] / active, "attn": s7["attn"] / active,
                                 "head": s7["head"] / active}}
    print("\n  Qwen2.5-Coder-7B active/token = %.3f G weights"
          "   (FFN %.1f%%, attn %.1f%%, head %.1f%%)"
          % (active / 1e9, 100.0 * s7["ffn"] / active, 100.0 * s7["attn"] / active,
             100.0 * s7["head"] / active))

    LADDER = [("base", set()), ("H", {"head"}), ("A", {"attn"}), ("F", {"ffn"}),
              ("FA", {"ffn", "attn"}), ("FAH", {"ffn", "attn", "head"})]
    parts = {"ffn": s7["ffn"], "attn": s7["attn"], "head": s7["head"]}
    out["ladder"] = []
    print("\n  %-5s %-22s %12s %12s %9s %9s" % ("arm", "ternary organs", "tern G-w",
                                                "fp32 G-w", "s/token", "tok/s"))
    for tag, tern_organs in LADDER:
        nt = sum(parts[k] for k in tern_organs)
        nf = sum(parts[k] for k in parts if k not in tern_organs)
        sec = nt / tern_w_s + nf / fp32_w_s
        row = {"arm": tag, "ternary_organs": sorted(tern_organs),
               "ternary_weights": nt, "fp32_weights": nf,
               "seconds_per_token_weight_path": sec, "tok_s_weight_path": 1.0 / sec,
               "short_of_50": TARGET_GOOD * sec}
        out["ladder"].append(row)
        print("  %-5s %-22s %12.3f %12.3f %9.5f %9.2f"
              % (tag, ",".join(sorted(tern_organs)) or "-", nt / 1e9, nf / 1e9, sec, 1.0 / sec))

    fah = [r for r in out["ladder"] if r["arm"] == "FAH"][0]
    out["sanity"] = {"FAH_weight_path_tok_s": fah["tok_s_weight_path"],
                     "measured_engine_tok_s": MEASURED_7B_TOKS,
                     "ratio": fah["tok_s_weight_path"] / MEASURED_7B_TOKS,
                     "note": "the weight path alone is an UPPER bound; the engine also does "
                             "attention math, norms, softmax and the glue, so measured < derived "
                             "is the expected direction"}
    print("\n  sanity: FAH weight path %.2f tok/s vs the engine's measured %.2f (ratio %.2f) -- "
          "the weight path is an upper bound, so measured below derived is the right direction"
          % (fah["tok_s_weight_path"], MEASURED_7B_TOKS,
             fah["tok_s_weight_path"] / MEASURED_7B_TOKS))
    print("\n  THE POINT: the FASTEST rung is the one that converts everything, and it is %.1fx "
          "short of 50 tok/s. Every rung that improves quality moves AWAY from the target."
          % (TARGET_GOOD / fah["tok_s_weight_path"]))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
