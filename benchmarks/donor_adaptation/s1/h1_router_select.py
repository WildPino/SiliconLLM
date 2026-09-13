"""H1 -- apply the REGISTERED selection rule of BRIEF_H1 addendum D.3 to the
eight-layer router smoke.

The rule was pushed (5604b0d) BEFORE layers 6-24 ran.  This script does nothing
but execute it verbatim on the two result files; it makes no choices of its own.

  1. Eligible = mean occ_max over the measured layers <= OCC_BAR (0.5).
  2. Among eligible, the setting beating STATIC on the most layers.
  3. Ties by best mean ratio.
  4. If no eligible setting beats STATIC on a MAJORITY of layers, record that and
     ship the best eligible one WITH THE SHORTFALL STATED -- not the best
     collapsed one.
  5. If nothing of either kind beats STATIC on any layer, addendum A cons. 3
     fires: H1 does not launch.

Two things the registered rule did NOT ask for are computed and reported anyway,
because they are the rule's blind spots and must be on the record next to it:

  * TOTAL ERROR MASS, sum over layers.  layers-won is a RANK metric; E14 s3 says
    a rank metric needs its score partner shown beside it.  They disagree here.
  * PER-LAYER eligibility, as a sensitivity on rule 1's averaging.  A setting can
    average below the bar while collapsing on one layer.

Neither is allowed to change the shipped setting.  E14 s6 forbids promoting a
post-hoc metric to a gate.
"""
import json, os, sys

OCC_BAR = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [os.path.join(HERE, "results/h1/h1_router_smoke_L3.json"),
         os.path.join(HERE, "results/h1/h1_router_smoke_L6-24.json")]
OUT = os.path.join(HERE, "results/h1/h1_router_select.json")


def key(g):
    return "lr=%g,aux=%g" % (g["lr"], g["aux"])


def main():
    grid, static = [], {}
    for f in FILES:
        d = json.load(open(f))
        grid += d["grid"]
        for li, rec in d["per_layer"].items():
            static[int(li)] = rec
    layers = sorted(static)
    # a setting is comparable only where it was measured on EVERY layer
    by = {}
    for g in grid:
        by.setdefault(key(g), {})[g["layer"]] = g
    full = {k: v for k, v in by.items() if set(v) == set(layers)}
    part = sorted(set(by) - set(full))

    print("== H1 router selection -- addendum D.3, applied verbatim ==")
    print("   layers %s   settings measured on all %d: %d   partial (dropped): %s"
          % (layers, len(layers), len(full), part or "none"))
    print()
    print("   layer   power     STATIC    STATIC/power")
    tot_static = 0.0
    for li in layers:
        r = static[li]
        tot_static += r["static"]
        print("   %5d   %.5f   %.6f   %.4f" % (li, r["power"], r["static"],
                                               r["static"] / r["power"]))
    print("   TOTAL STATIC ERROR MASS %.6f   (layer %d alone = %.1f%%)"
          % (tot_static, layers[-1],
             100.0 * static[layers[-1]]["static"] / tot_static))
    print()

    rows = []
    for k, cells in sorted(full.items()):
        won = [li for li in layers if cells[li]["beats"]]
        occ = [cells[li]["occ_max"] for li in layers]
        mocc = sum(occ) / len(occ)
        mrat = sum(cells[li]["ratio"] for li in layers) / len(layers)
        mass = sum(cells[li]["heldout"] for li in layers)
        # sensitivity only: a win on a layer whose own occ_max is over the bar
        clean = [li for li in won if cells[li]["occ_max"] <= OCC_BAR]
        rows.append(dict(k=k, won=won, mocc=mocc, mrat=mrat, mass=mass,
                         clean=clean, elig=mocc <= OCC_BAR, cells=cells))

    print("   setting             layers won      mean occ_max  elig  mean ratio  error mass   vs STATIC")
    for r in rows:
        print("   %-18s  %d of %d %-10s %.4f        %-4s  %.4f      %.6f   %+.4f%%"
              % (r["k"], len(r["won"]), len(layers), str(r["won"]), r["mocc"],
                 "YES" if r["elig"] else "no", r["mrat"], r["mass"],
                 100.0 * (r["mass"] / tot_static - 1.0)))
    print()

    if not any(r["won"] for r in rows):
        print("   RULE 5 FIRES: nothing beats STATIC on any layer.")
        print("   Addendum A consequence 3: H1 DOES NOT LAUNCH.")
        json.dump({"rule": "addendum D.3", "launch": False, "rule5_fires": True,
                   "layers": layers}, open(OUT, "w", encoding="utf-8"), indent=1)
        return 2

    elig = [r for r in rows if r["elig"]]
    if not elig:
        print("   NO ELIGIBLE SETTING (every setting collapses on average).")
        json.dump({"rule": "addendum D.3", "launch": False, "no_eligible": True,
                   "layers": layers}, open(OUT, "w", encoding="utf-8"), indent=1)
        return 2
    elig.sort(key=lambda r: (-len(r["won"]), r["mrat"]))
    win = elig[0]
    maj = len(layers) // 2 + 1
    short = len(win["won"]) < maj

    print("   rule 1 eligible: %s" % ", ".join(r["k"] for r in elig))
    print("   rule 2 most layers won among eligible: %s (%d)"
          % (win["k"], len(win["won"])))
    tied = [r for r in elig if len(r["won"]) == len(win["won"])]
    print("   rule 3 tie-break by mean ratio: %s"
          % ("not needed" if len(tied) == 1 else
             " < ".join("%s %.4f" % (r["k"], r["mrat"]) for r in tied)))
    print()
    print("   SHIPPED: %s" % win["k"])
    if short:
        print("   RULE 4 FIRES -- SHORTFALL: it beats STATIC on %d of %d layers,"
              % (len(win["won"]), len(layers)))
        print("   below the majority of %d.  Recorded as measured." % maj)
    print()
    print("   -- NOT part of the rule, reported beside it --")
    print("   score partner (E14 s3): total error mass %+.4f%% vs STATIC -- %s"
          % (100.0 * (win["mass"] / tot_static - 1.0),
             "WORSE in aggregate" if win["mass"] > tot_static else "better in aggregate"))
    best_mass = min(rows, key=lambda r: r["mass"])
    print("   best aggregate error mass of ANY setting: %s %+.4f%% (eligible: %s)"
          % (best_mass["k"], 100.0 * (best_mass["mass"] / tot_static - 1.0),
             "yes" if best_mass["elig"] else "NO -- collapsed"))
    print("   rule-1 averaging sensitivity: wins on layers whose OWN occ_max <= %.1f" % OCC_BAR)
    for r in rows:
        print("      %-18s  won %-10s  clean %-10s  occ at wins %s"
              % (r["k"], str(r["won"]), str(r["clean"]),
                 ["%.3f" % r["cells"][li]["occ_max"] for li in r["won"]]))
    alt = sorted([r for r in rows if r["clean"]],
                 key=lambda r: (-len(r["clean"]), r["mrat"]))
    print("   under per-layer eligibility the same rule would ship: %s"
          % (alt[0]["k"] if alt else "NOTHING"))
    print("   -> %s" % ("SAME setting; the defect does not move the answer."
                        if alt and alt[0]["k"] == win["k"] else
                        "DIFFERENT -- the defect is load-bearing, see the addendum."))

    json.dump({"rule": "briefs/BRIEF_H1_THE_CARVE_TRAINED_NOT_APPLIED.md addendum D.3, "
                       "registered BEFORE layers 6-24 ran (5604b0d)",
               "launch": True, "occ_bar": OCC_BAR, "layers": layers,
               "router_lr": win["cells"][layers[0]]["lr"],
               "aux": win["cells"][layers[0]]["aux"],
               "setting": win["k"], "layers_won": win["won"],
               "majority_needed": maj, "rule4_shortfall": bool(short),
               "mean_occ_max": win["mocc"], "mean_ratio": win["mrat"],
               "total_static_error": tot_static,
               "table": [{"setting": r["k"], "layers_won": r["won"],
                          "mean_occ_max": r["mocc"], "eligible": r["elig"],
                          "mean_ratio": r["mrat"], "error_mass": r["mass"],
                          "vs_static_pct": 100.0 * (r["mass"] / tot_static - 1.0),
                          "clean_wins": r["clean"]} for r in rows],
               "DIAGNOSTIC_not_a_gate": {
                   "shipped_error_mass_vs_static_pct":
                       100.0 * (win["mass"] / tot_static - 1.0),
                   "best_error_mass_setting": best_mass["k"],
                   "best_error_mass_eligible": bool(best_mass["elig"]),
                   "per_layer_eligibility_would_ship": alt[0]["k"] if alt else None}},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("   wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
