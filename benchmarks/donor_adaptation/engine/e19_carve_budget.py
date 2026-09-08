#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E19 part A -- what carve DEPTH does E18's budget actually require?

E18 part C derived the budget: 50 tok/s permits ~0.98-1.06 G active ternary weights per token.
The only structural lever that can reach it is conditional activation -- carving.  Every carve this
programme has ever built (D0, D0c) carves the FFN and ONLY the FFN.

This asks the question that must be answered before a carve is worth measuring: with the FFN carved
to an arbitrary fraction -- INCLUDING ZERO -- what does the rest of the model already cost?

Same derivation discipline as `e18_ladder_bandwidth.py`, which it imports rather than restates:
moved-byte convention, measured rates with citations, NO timing taken and none quotable.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e18_ladder_bandwidth import qwen_shapes, verify_bytes_per_weight   # noqa: E402
from e18_ladder_bandwidth import PACKED_GB_S, TARGET_GOOD, TARGET_GREAT  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "e19_carve_budget.json")

# The measured span, exactly as E18 part C reported it -- ledger s23.3 pairs "~25.5 GB/s" with
# "~53 G-w/s" and 25.5/0.5 = 51.0, so the two companion numbers disagree by 4%.  Span, not point.
RATE_SPAN = {"kernel bench low (s23.2)": 49.11e9,
             "engine gate+up (s23.2)": 51.66e9,
             "engine down (s23.2)": 52.22e9,
             "kernel bench high (s23.2)": 52.95e9,
             "ledger rounded companion (s23.3)": 53.0e9}

MODELS = {
    # tag: (D, F, L, n_heads, n_kv, head_dim, V)
    "Qwen2.5-Coder-7B": (3584, 18944, 28, 28, 4, 128, 152064),
    "Qwen2.5-1.5B (D0/D0c donor)": (1536, 8960, 28, 12, 2, 128, 151936),
}


def main():
    v = verify_bytes_per_weight()
    bw = v["bytes_per_weight"] if v else 0.5
    tern_w_s = PACKED_GB_S / bw
    span = dict(RATE_SPAN)
    span["derived: 25.5 GB/s / %.6f B-per-weight" % bw] = tern_w_s
    lo, hi = min(span.values()), max(span.values())

    out = {"note": "DERIVATION over measured quantities; no timing taken, none quotable",
           "imports": "shapes and bytes/weight verification from e18_ladder_bandwidth.py",
           "bytes_per_weight_verified": v,
           "rate_span_G_w_s": {k: r / 1e9 for k, r in span.items()},
           "convention": "moved bytes; every weight charged as ternary (0.5 B) -- the FRIENDLIEST "
                         "possible assumption, since it grants full ternarization for free",
           "models": {}}

    print("bytes/weight verified = %.6f  (%.4f bits)  -> ternary %.2f-%.2f G-w/s across the span"
          % (bw, 8.0 * bw, lo / 1e9, hi / 1e9))

    for tag, (D, F, L, nh, nkv, hd, V) in MODELS.items():
        s = qwen_shapes(D, F, L, nh, nkv, hd, V, tied=False)
        attn, ffn, head = s["attn"], s["ffn"], s["head"]
        active = attn + ffn + head
        floor = attn + head                      # what remains with the FFN carved to ZERO

        row = {"D": D, "F": F, "L": L, "V": V,
               "attn": attn, "ffn": ffn, "head": head, "active_per_token": active,
               "uncarvable_floor_attn_plus_head": floor,
               "floor_share_of_active": floor / float(active),
               "budget_50": {"low": lo / TARGET_GOOD, "high": hi / TARGET_GOOD},
               "budget_100": {"low": lo / TARGET_GREAT, "high": hi / TARGET_GREAT},
               "tok_s_with_ffn_at_zero": {"low": lo / floor, "high": hi / floor},
               "floor_exceeds_50_budget": bool(floor > hi / TARGET_GOOD)}

        # what FFN activation fraction would 50 tok/s need, if it is reachable at all?
        need = {}
        for t, name in ((TARGET_GOOD, "50"), (TARGET_GREAT, "100")):
            b_lo, b_hi = lo / t, hi / t
            # solve attn + head + f*ffn = budget  ->  f = (budget - floor) / ffn
            f_lo, f_hi = (b_lo - floor) / ffn, (b_hi - floor) / ffn
            need[name] = {"required_ffn_active_fraction_low": f_lo,
                          "required_ffn_active_fraction_high": f_hi,
                          "reachable_by_carving_ffn_alone": bool(f_hi > 0.0)}
        row["required_ffn_carve"] = need

        # Proportional projection to the goal's size.  NO config is invented: this carries the
        # MEASURED attn+head share of a real donor across to a 10 B of the same proportions, and
        # is labelled a projection wherever it is quoted.
        proj_floor = (floor / float(active)) * 10e9
        row["projection_to_10B_same_proportions"] = {
            "assumption": "attn+head keep this donor's MEASURED share of active weights",
            "attn_plus_head_share": floor / float(active),
            "projected_attn_plus_head": proj_floor,
            "budget_50_high": hi / TARGET_GOOD,
            "times_over_the_50_budget": proj_floor / (hi / TARGET_GOOD),
            "tok_s_with_ffn_at_zero": {"low": lo / proj_floor, "high": hi / proj_floor}}
        out["models"][tag] = row

        print("\n%s" % tag)
        print("  active/token %.3f G   = FFN %.3f (%.1f%%) + attn %.3f + head %.3f"
              % (active / 1e9, ffn / 1e9, 100.0 * ffn / active, attn / 1e9, head / 1e9))
        print("  50 tok/s budget      : %.3f - %.3f G active weights"
              % (lo / TARGET_GOOD / 1e9, hi / TARGET_GOOD / 1e9))
        print("  UNCARVABLE floor     : attn + head = %.3f G  (%.1f%% of active)"
              % (floor / 1e9, 100.0 * floor / active))
        print("  with the FFN at ZERO : %.1f - %.1f tok/s" % (lo / floor, hi / floor))
        f = need["50"]
        if f["required_ffn_active_fraction_high"] <= 0.0:
            print("  => 50 tok/s is UNREACHABLE by carving the FFN alone, at ANY depth,")
            print("     because attn+head ALONE already exceed the whole budget.")
        else:
            print("  => 50 tok/s needs FFN active fraction %.4f - %.4f"
                  % (f["required_ffn_active_fraction_low"],
                     f["required_ffn_active_fraction_high"]))
        pj = row["projection_to_10B_same_proportions"]
        print("  projection: a 10 B keeping THIS donor's attn+head share (%.1f%%) has a"
              % (100.0 * floor / active))
        print("              %.3f G uncarvable floor = %.2fx the ENTIRE 50 tok/s budget,"
              % (pj["projected_attn_plus_head"] / 1e9, pj["times_over_the_50_budget"]))
        print("              i.e. %.1f-%.1f tok/s with its FFN carved to ZERO"
              % (pj["tok_s_with_ffn_at_zero"]["low"], pj["tok_s_with_ffn_at_zero"]["high"]))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
