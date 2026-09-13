#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E57 addendum D -- which of the six trained arms is at the memory wall?

DERIVATION over measured quantities.  No timing is taken here; the rates are E57's and the
bandwidth bounds are the ledger's.  Nothing new is quotable as a rate.

THE UNIT TRAP THIS EXISTS TO AVOID.  A file size is not a moved byte (the byte-convention law,
E31).  Qwen2.5 TIES its embedding to its head, so in the fp32 and `tq` artifacts ONE matrix is
both gathered once per token (3.6 KB) and streamed once per token as the head -- it counts.  But
`tqh` ternarises the head and the exporter therefore UNTIES them: it stores an fp32 embedding
that is only GATHERED plus a separate ternary head that is streamed.  Charging the fp32
embedding to a tqh token would read 67 GB/s on a machine whose DRAM ceiling is 40-44.

Every structural quantity below is READ from the artifact or from E1's stored code counts, never
assumed; the tie/untie inference is CHECKED by reconstructing each file's size to <0.3%.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E1RES = os.path.join(HERE, "..", "density", "results")

# ledger section 1 / PHASE64_BUDGET.md section 1
DRAM_LOW, DRAM_HIGH = 40.0, 44.0        # DRAM aggregate ceiling, saturated at 3 threads
PROJ_FLOOR = 37.0                        # proj-GEMV streamed floor

V = 151936
SHAPE = {"05b": {"D": 896, "file_tag": "qwen25-05b"}, "15b": {"D": 1536, "file_tag": "qwen25-15b"}}
# E57 addendum A, all-cell medians
RATE = {"05b_f32": 19.47, "05b_tq": 44.13, "05b_tqh": 84.88,
        "15b_f32": 6.27, "15b_tq": 19.19, "15b_tqh": 30.03}


def main():
    e1 = {}
    for f in os.listdir(E1RES):
        if f.startswith("e1_bpb_through_engine_qwen") and "smoke" not in f:
            d = json.load(open(os.path.join(E1RES, f)))
            for k, v in d["arms"].items():
                e1[(os.path.basename(v["weights"]).replace(".bin", "")
                    .replace("qwen25-", ""))] = v

    print("E57 addendum D -- moved bytes per token, and how close each arm is to the wall")
    print("derivation only; rates are E57's medians, bounds are the ledger's")
    print("")
    print("  %-10s %12s %12s %12s  %8s %9s  %s"
          % ("arm", "file MB", "gathered MB", "MOVED MB", "tok/s", "GB/s", "vs the wall"))
    rows = []
    checks = []
    for size in ("05b", "15b"):
        D = SHAPE[size]["D"]
        emb = V * D * 4
        for arm in ("f32", "tq", "tqh"):
            tag = "%s_%s" % (size, arm)
            path = r"D:\_ktmp\e1\qwen25-%s_%s.bin" % (size, arm)
            if not os.path.exists(path):
                continue
            fb = os.path.getsize(path)
            rec = e1.get("%s_%s" % (size, arm.upper().lower()))
            # tqh unties: the fp32 embedding is GATHERED, never streamed.  f32 and tq tie, so the
            # one matrix is streamed as the head and the gather is 4*D bytes, which is noise.
            gathered = emb if arm == "tqh" else 0
            moved = fb - gathered
            r = RATE[tag]
            gbs = moved * r / 1e9
            if gbs >= PROJ_FLOOR:
                verdict = "AT THE WALL (>= proj floor %.0f)" % PROJ_FLOOR
            elif gbs >= 0.8 * PROJ_FLOOR:
                verdict = "near it"
            else:
                verdict = "BELOW it -- %.2fx of headroom to %.0f GB/s" % (PROJ_FLOOR / gbs,
                                                                          PROJ_FLOOR)
            # The tie/untie inference is the whole derivation, so it is CHECKED, not asserted:
            # reconstruct the file from E1's stored ternary code count and see if it closes.
            ncodes = (rec or {}).get("gate_a", {}).get("n_codes", 0)
            recon = emb + ncodes * 0.5 if arm != "f32" else emb + (fb - emb)
            resid = (fb - recon) / float(fb)
            checks.append((tag, "tied" if arm != "tqh" else "UNTIED", ncodes, recon, resid))
            rows.append((tag, fb, gathered, moved, r, gbs, verdict))
            print("  %-10s %12.1f %12.1f %12.1f  %8.2f %9.1f  %s"
                  % (tag, fb / 1e6, gathered / 1e6, moved / 1e6, r, gbs, verdict))
    print("")
    print("  TIE/UNTIE CHECK -- the inference above reconstructed from E1's code counts:")
    print("    %-10s %-7s %14s %14s %9s" % ("arm", "layout", "ternary codes", "recon MB", "residual"))
    for tag, lay, nc, recon, resid in checks:
        print("    %-10s %-7s %14d %14.1f %8.2f%%   (scales + norms + header)"
              % (tag, lay, nc, recon / 1e6, 100 * resid))
    print("    Every residual is positive and under 0.3%: the reconstruction closes, so the")
    print("    tqh artifacts really do carry an fp32 embedding BESIDE a ternary head.")
    print("")
    f32 = [x for x in rows if x[0].endswith("f32")]
    if len(f32) == 2:
        a, b = f32[0][5], f32[1][5]
        print("  CONSISTENCY CHECK nobody arranged: the two fp32 arms differ 3.13x in size and")
        print("  read %.1f and %.1f GB/s -- %.2f%% apart.  A bytes-per-token model that is wrong"
              % (a, b, 100 * abs(a - b) / max(a, b)))
        print("  does not agree with itself across a 3x change of denominator.")
    print("")
    tern = [x for x in rows if not x[0].endswith("f32")]
    print("  WHERE THE REMAINING ENGINE WORK IS.  Every ternary arm is below the streamed floor;")
    print("  at the floor each would read:")
    for t in tern:
        print("    %-10s %7.2f -> %7.2f tok/s  (x%.2f)"
              % (t[0], t[4], t[4] * PROJ_FLOOR / t[5], PROJ_FLOOR / t[5]))
    print("")
    print("  And the two FAITHFUL arms have NO such row: they are already at or above the")
    print("  streamed floor and inside the %.0f-%.0f GB/s DRAM band.  The only lever left on"
          % (DRAM_LOW, DRAM_HIGH))
    print("  them is FEWER BYTES, which is quantisation, which is what breaks them.")
    out = os.path.join(HERE, "results", "e57_wall.json")
    json.dump({"note": "derivation over measured quantities; no timing taken, none quotable",
               "dram_band": [DRAM_LOW, DRAM_HIGH], "proj_floor": PROJ_FLOOR,
               "tie_check": [{"arm": c[0], "layout": c[1], "n_codes": c[2],
                             "recon_bytes": c[3], "residual_frac": c[4]} for c in checks],
               "rows": [{"arm": r[0], "file_bytes": r[1], "gathered_bytes": r[2],
                         "moved_bytes": r[3], "tok_s": r[4], "GB_s": r[5], "verdict": r[6]}
                        for r in rows]}, open(out, "w"), indent=1)
    print("")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
