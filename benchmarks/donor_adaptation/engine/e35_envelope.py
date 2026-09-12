#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E35 -- the 50 tok/s envelope: measure the shape that fits, do not derive it.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E35_THE_FIFTY_TOK_S_ENVELOPE.md, pushed at
17e7fc8 BEFORE this file existed.  Nothing here may contradict it.

E34 measured that T10's attention+head floor reads 20.03 tok/s -- 2.50x short WITH THE FFN GONE.
That closes "can the goal's literal shape be reached by shrinking the FFN".  It does not answer
what the goal asks: what shape DOES run at 50, and how big can it be.  E34 s4.1 gave a desk
answer (31 of 48 layers at the CEILING) and E34 s7 named this the main line, because a desk
table decides nothing here.

THE MODEL, registered in the brief s2 before any of this ran:
    charged(L) = 41,943,040*L + 134,217,728          (attention*L + head, FFN at carve k=1)
    tok/s(L)   = 46.74e9 / charged(L)                 -> the crossing predicted at L = 19

  python e35_envelope.py --reps 5
  python e35_envelope.py --build-only
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                        # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e35_envelope.json")

GOAL = 50.0
E_GROUPS, CARVE_K = 256, 1
NTOK = 40
ANCHOR_TOL = 0.10
LINEARITY_TOL = 0.05
BANDS = [(24.0, "DEPTH-IS-AFFORDABLE"), (16.0, "DEPTH-IS-HALVED"), (0.0, "DEPTH-COLLAPSES")]

#        tag          shape     L    the E34 artifact is reused for L=48 (same construction)
ARMS = [("T10-L48", "T10",    48, r"D:\_ktmp\e34\e34_t10_carve.bin"),
        ("T10-L32", "T10L32", 32, None),
        ("T10-L24", "T10L24", 24, None),
        ("T10-L16", "T10L16", 16, None),
        ("T10-L12", "T10L12", 12, None)]

# brief s2's predicted table, quoted so the measurement can contradict it
BRIEF_PRED = {48: 21.8, 32: 31.7, 24: 41.0, 16: 58.0, 12: 73.3}
BRIEF_CROSSING = 19


def log(*a):
    print(*a, flush=True)


def synth():
    spec = importlib.util.spec_from_file_location("se", os.path.join(HERE, "synth_export.py"))
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["synth_export.py"]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


def build(d, tag, shape, path):
    if path:
        if not os.path.exists(path):
            raise SystemExit("the reused control artifact is missing: " + path)
        log("  reuse %-9s %s" % (tag, os.path.basename(path)))
        return path
    out = os.path.join(d, "e35_%s.bin" % shape.lower())
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-9s %s" % (tag, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", str(CARVE_K)]
    log("  build %-9s %s" % (tag, shape))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed for " + tag)
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln or ln.startswith("wrote"):
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e35")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    d = winpath(a.dir)
    os.makedirs(d, exist_ok=True)
    engine = a.engine if os.path.isabs(a.engine) else os.path.join(HERE, a.engine)
    t0 = time.time()
    m = synth()

    e34 = json.load(open(os.path.join(RES, "e34_floor_under_floor.json")))
    ctrl = e34["arms"]["T10-FLOOR"]["mean_tok_s"]
    num = e34["arms"]["T10-DENSE"]["charged_G_w_per_s"] * 1e9
    log("== E35: the 50 tok/s envelope ==")
    log("  anchors READ FROM FILE: E34 T10-FLOOR %.3f tok/s, numerator %.2f G-w/s"
        % (ctrl, num / 1e9))
    log("  the model, registered in the brief before anything ran:")
    for _, shape, L, _p in ARMS:
        D, F, LL, NH, NKV, HD, V, tied, _ = m.SHAPES[shape]
        ch = (NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D) * L + V * D
        log("     L=%2d  charged %10d  predicted %6.2f tok/s   (brief s2 said %.1f)"
            % (L, ch, num / ch, BRIEF_PRED[L]))
    log("     predicted crossing at L = %d" % BRIEF_CROSSING)

    out = {"brief": "briefs/BRIEF_E35_THE_FIFTY_TOK_S_ENVELOPE.md (17e7fc8)",
           "goal_tok_s": GOAL, "E": E_GROUPS, "carve_k": CARVE_K, "ntok": NTOK,
           "threads": a.threads, "reps": a.reps, "bands": BANDS,
           "e34_control_tok_s": ctrl, "numerator_w_per_s": num,
           "brief_predicted": BRIEF_PRED, "brief_crossing_L": BRIEF_CROSSING, "arms": {}}

    log("")
    paths, metas, g_b, ok_b = {}, {}, {}, True
    for tag, shape, L, p in ARMS:
        paths[tag] = build(d, tag, shape, p)
        metas[tag] = json.load(open(paths[tag] + ".json"))
        D, F, LL, NH, NKV, HD, V, tied, _ = m.SHAPES[shape]
        assert LL == L, "shape table and arm disagree on L"
        runner_act = m.active_weights(D, F, L, NH, NKV, HD, V, E_GROUPS, CARVE_K)
        floor = (NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D) * L + V * D
        same = (metas[tag]["active_weights_per_token"] == runner_act)
        ok_b = ok_b and same
        g_b[tag] = {"exporter": metas[tag]["active_weights_per_token"], "runner": runner_act,
                    "floor": floor, "residue": runner_act - floor, "agrees": bool(same)}
    log("")
    for tag, v in g_b.items():
        log("  G-E35B  %-9s exporter %10d  runner %10d  floor %10d  residue %+d -> %s"
            % (tag, v["exporter"], v["runner"], v["floor"], v["residue"],
               "OK" if v["agrees"] else "MISMATCH"))
    out["G_E35B"] = {"per_arm": g_b, "fires": ok_b}
    if not ok_b:
        out["VOID"] = "G-E35B: exporter and runner disagree on charged weights"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E35B VOID -- see " + OUT)

    if a.build_only:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  --build-only: no timing taken")
        return 0

    busy, peak = cpu_busy(4)
    log("")
    log("  contention witness: %.1f%% mean / %.0f%% peak (bar %.0f%% on the MEAN)"
        % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)
    out["cpu_busy_mean_pct"], out["cpu_busy_peak_pct"] = [busy], [peak]

    rates = dict((t, []) for t, _, _, _ in ARMS)
    log("")
    for r in range(a.reps):                        # reps OUTERMOST
        for tag, shape, L, _p in ARMS:
            v = bench(engine, paths[tag], NTOK, a.threads, [])
            rates[tag].append(v)
            log("  rep %d  %-9s %7.2f tok/s" % (r + 1, tag, v))
        b2, p2 = cpu_busy(2)
        out["cpu_busy_mean_pct"].append(b2)
        out["cpu_busy_peak_pct"].append(p2)
        log("         box %.1f%% mean / %.0f%% peak%s"
            % (b2, p2, "   !! ABOVE THE BAR" if b2 > IDLE_BAR else ""))

    for tag, shape, L, _p in ARMS:
        v = sorted(rates[tag])
        mean = sum(v) / len(v)
        ch = g_b[tag]["runner"]
        out["arms"][tag] = {
            "L": L, "shape": shape, "rates": rates[tag], "mean_tok_s": mean,
            "median_tok_s": v[len(v) // 2], "spread": (v[-1] - v[0]) / mean,
            "charged": ch, "floor_charged": g_b[tag]["floor"],
            "predicted_tok_s": num / ch, "rel_dev_vs_model": mean / (num / ch) - 1.0,
            "charged_G_w_per_s": mean * ch / 1e9,
            "ffn_budget_left_at_50": num / GOAL - ch}

    # ---- G-E35A: the planted control
    got = out["arms"]["T10-L48"]["mean_tok_s"]
    out["G_E35A"] = {"measured": got, "e34": ctrl, "rel_dev": got / ctrl - 1.0,
                     "tol": ANCHOR_TOL, "fires": bool(abs(got / ctrl - 1.0) <= ANCHOR_TOL)}
    log("")
    log("  G-E35A  T10-L48 %.2f vs E34's T10-FLOOR %.2f (%+.1f%%) -> %s"
        % (got, ctrl, 100 * out["G_E35A"]["rel_dev"],
           "FIRES" if out["G_E35A"]["fires"] else "VOID"))
    if not out["G_E35A"]["fires"]:
        out["VOID"] = "G-E35A: not comparable to E34"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  VOID -- no verdict is quotable from this run.")
        return 2

    # ---- prediction 4: is tok/s linear in 1/charged across the arms?
    devs = dict((t, out["arms"][t]["rel_dev_vs_model"]) for t, _, _, _ in ARMS)
    worst = max(abs(x) for x in devs.values())
    out["linearity"] = {"rel_dev_vs_model": devs, "worst": worst, "tol": LINEARITY_TOL,
                        "holds": bool(worst <= LINEARITY_TOL)}
    log("")
    for t, _, L, _p in ARMS:
        A = out["arms"][t]
        log("  L=%2d  measured %6.2f   model %6.2f   %+6.1f%%   charged %.4f G   "
            "FFN budget left at 50: %+.4f G" % (L, A["mean_tok_s"], A["predicted_tok_s"],
                                                100 * A["rel_dev_vs_model"], A["charged"] / 1e9,
                                                A["ffn_budget_left_at_50"] / 1e9))
    log("  worst deviation from the registered model: %.1f%% (prediction 4 bar %.0f%%) -> %s"
        % (100 * worst, 100 * LINEARITY_TOL, "HOLDS" if out["linearity"]["holds"] else "BREAKS"))

    # ---- the verdict: interpolate L* in charged, where tok/s crosses 50
    pts = sorted(((out["arms"][t]["charged"], out["arms"][t]["mean_tok_s"], out["arms"][t]["L"])
                  for t, _, _, _ in ARMS))
    lo = hi = None
    for i in range(len(pts) - 1):
        if (pts[i][1] - GOAL) * (pts[i + 1][1] - GOAL) <= 0:
            lo, hi = pts[i], pts[i + 1]
    if lo is None:
        name, Lstar = "NO-CROSSING", None
        log("")
        log("  the curve never crosses %.0f tok/s inside the measured arms" % GOAL)
    else:
        # tok/s = N / charged  =>  charged* = N/GOAL with N interpolated between the two arms
        n_lo, n_hi = lo[1] * lo[0], hi[1] * hi[0]
        n_star = (n_lo + n_hi) / 2.0
        ch_star = n_star / GOAL
        Lstar = (ch_star - 134217728.0) / 41943040.0
        name = next(nm for bar, nm in BANDS if Lstar >= bar)
        log("")
        log("  crossing bracketed by L=%d (%.2f) and L=%d (%.2f)" % (lo[2], lo[1], hi[2], hi[1]))
    out["verdict"] = {"name": name, "L_star": Lstar, "brief_predicted_L": BRIEF_CROSSING,
                      "in_17_21": bool(Lstar is not None and 17 <= Lstar <= 21),
                      "below_19": bool(Lstar is not None and Lstar < 19),
                      "active_budget_at_50_G": num / GOAL / 1e9}
    log("  VERDICT CELL  L* = %s  ->  %s   (the brief predicted %d)"
        % ("%.2f" % Lstar if Lstar else "none", name, BRIEF_CROSSING))
    log("     prediction 2 (L* in [17,21]): %s ;  prediction 3 (L* < 19): %s"
        % (out["verdict"]["in_17_21"], out["verdict"]["below_19"]))
    log("")
    log("  THE ENVELOPE: at %.0f tok/s this box affords %.4f G ACTIVE charged weights/token."
        % (GOAL, num / GOAL / 1e9))
    log("  That is the number any 10 B must fit its ACTIVE half inside -- an MoE statement,")
    log("  not a depth statement.  Registered as prediction 5 before the run.")

    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
