# -*- coding: utf-8 -*-
"""Re-score E51 run 1's OWN speed cells.  No cell is re-measured.

`phase_speed` crashed in run 1: `fit()` raised on a single judged window, so G-E51c printed
its per-window table and then no verdict at all.  The cells are in `e51_run1.log`; this reads
them back and runs the FIXED decision functions over them, exactly as E50 B.8 rebuilt its
results from run 1's own artifacts rather than running anything again.

    python e51c_rescore.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e51_math_errno as E51                                          # noqa: E402

LOG = os.path.join(HERE, "e51_run1.log")
OUT = os.path.join(HERE, "results", "e51_math_errno.json")
CELL = re.compile(r"^\s+(base|e51)\s+n=(\d+)\s+rep (\d+)/(\d+)\s+([\d.]+) tok/s\s+occ ([\d.]+)%")

cells, occ = {}, {}
for line in open(LOG, encoding="utf-8", errors="replace"):
    m = CELL.match(line)
    if not m:
        continue
    tag, n, _, _, r, o = m.group(1), int(m.group(2)), 0, 0, float(m.group(5)), float(m.group(6))
    cells.setdefault(n, {"base": [], "e51": []})[tag].append(r)
    occ.setdefault(n, []).append(o)

assert cells, "no cells parsed out of %s" % LOG
for n in cells:
    assert len(cells[n]["base"]) == len(cells[n]["e51"]) == 5, (n, cells[n])
print("parsed %d windows x 5 reps x 2 builds = %d cells from run 1's log"
      % (len(cells), sum(len(v["base"]) + len(v["e51"]) for v in cells.values())))

rows = E51.g_e51c(cells)
summary = E51.g_e51c_summary(rows)
print("\n  %-6s %9s %9s %8s %8s %8s  %s"
      % ("n", "base", "e51", "delta", "sp_base", "sp_e51", "verdict"))
for r in rows:
    print("  %-6d %8.2f  %8.2f  %+7.1f%% %7.1f%% %7.1f%%  %s%s"
          % (r["n"], r["base"], r["e51"], 100 * r["delta"], 100 * r["spread_base"],
             100 * r["spread_e51"], r["verdict"],
             "" if r["verdict"] == "UNRESOLVABLE"
             else ("  separable" if r["separable"] else "  NOT separable")))
print("\n  G-E51c : %s\n  %s" % (summary["verdict"], summary["why"]))

judged = [r for r in rows if r["verdict"] == "JUDGED"]
if len({r["n"] for r in judged}) < 2:
    print("  C50 NOT COMPUTABLE -- %d window(s) survived the dispersion rule and a straight "
          "line needs two." % len(judged))

# --- the paired view.  DIAGNOSTIC, not a gate: it is not what section 4 registered. --------
print("\n  paired per-rep ratio e51/base (DIAGNOSTIC -- G-E51c is registered on the medians,"
      "\n  and a metric invented after seeing the data may not promote a result):")
for n in sorted(cells):
    rt = [e / b for b, e in zip(cells[n]["base"], cells[n]["e51"])]
    print("    n=%-5d %s   median %+.1f%%"
          % (n, " ".join("%+.1f%%" % (100 * (x - 1)) for x in rt),
             100 * (sorted(rt)[len(rt) // 2] - 1)))
print("\n  and the LEVEL of each arm, rep by rep, which is what the spread test actually saw:")
for n in sorted(cells):
    print("    n=%-5d base %s" % (n, " ".join("%7.2f" % v for v in cells[n]["base"])))
    print("           e51  %s" % " ".join("%7.2f" % v for v in cells[n]["e51"]))

out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
out["G_E51c"] = {"rows": rows, "fits": {}, "cells": cells, "summary": summary,
                 "tolerance": E51.G51C_SPREAD_TOL, "reps": 5,
                 "occupancy": occ,
                 "provenance": "re-scored from e51_run1.log; no cell re-measured"}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
print("\nwrote %s" % OUT)
