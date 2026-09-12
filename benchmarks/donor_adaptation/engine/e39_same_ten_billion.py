#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E39 -- the same ten billion, put somewhere else.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E39_THE_SAME_TEN_BILLION_PUT_SOMEWHERE_ELSE.md,
pushed BEFORE this file existed and before the exporter could build the object.  Nothing here
may contradict it.

E34 and E36 both named the attention shape as the remaining lever and neither pulled it.  At
A10B the base term is 17.2 ms of a 10 ms budget and 82% of it is q/k/v/o.  A10B-R512 holds the
parameter count EXACTLY fixed -- 9,999,220,736, the integer E36 measured, and the same per-layer
total too -- and moves weight out of q/o into the FFN, where the carve makes it nearly free.

  python e39_same_ten_billion.py --build-only          # exports + the three gates, no timing
  python e39_same_ten_billion.py --reps 5              # the measurement (idle box, operator idle)
  python e39_same_ten_billion.py --reps 5 --order reversed
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                        # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402
import synth_export as SX                                            # noqa: E402
import e1_bpb_through_engine as E1                                   # noqa: E402

RES = os.path.join(HERE, "results")

GOAL, EXCELLENT = 50.0, 100.0
E_GROUPS = 256
KS = [1, 2, 3, 4, 6]
K_IN_FILE = 3
VERDICT_K = 3
NTOK = 40
TEN_B = 9999220736                                  # E36's integer, to the parameter

# Brief s6.  The cell is A10B-R512 at k=3, against A10B's measured 49.96.
BANDS = [(100.0, "TEN-B-AT-HUNDRED"), (80.0, "TEN-B-NEAR-HUNDRED"),
         (55.0, "RANK-BUYS-SPEED"), (0.0, "RANK-DOES-NOT-BUY-SPEED")]
E36_A10B_K3 = 49.96                                 # prediction 5 checks this session against it
E36_TOL = 0.05

# Brief s2, quoted so the measurement can contradict MY arithmetic and not the other way round.
def closed_form_r512(k):
    return 419430400 + 36962304 * k


# name -> (shape, rank, role)
ARMS = {
    "A10B":  ("A10B",      0,    "E36's object: the MATCHED-PARAMETER-COUNT control"),
    "R512":  ("A10B-R512", 512,  "the verdict object: A10B's parameters, put somewhere else"),
    "R0":    ("A10B-R512", 0,    "same FFN as R512, DENSE q/o: the rank axis alone (NOT 10 B)"),
    "R4096": ("A10B-R512", 4096, "G-E39C planted control: a rank that moves MORE weights"),
}


def log(*a):
    print(*a, flush=True)


def shape_of(name):
    s = SX.SHAPES[name]
    return dict(zip(("D", "F", "L", "NH", "NKV", "HD", "V", "tied"), s[:8]))


def charged(arm, k):
    sh = shape_of(ARMS[arm][0])
    return SX.active_weights(sh["D"], sh["F"], sh["L"], sh["NH"], sh["NKV"], sh["HD"], sh["V"],
                             E_GROUPS, k, ARMS[arm][1])


def total(arm):
    sh = shape_of(ARMS[arm][0])
    return SX.total_weights(sh["D"], sh["F"], sh["L"], sh["NH"], sh["NKV"], sh["HD"], sh["V"],
                            bool(sh["tied"]), ARMS[arm][1])


def build(arm, d, reuse):
    shape, rank, _ = ARMS[arm]
    if arm == "A10B" and reuse and os.path.exists(reuse):
        log("  have  A10B  <- %s  (E36's own artifact, not rebuilt)" % reuse)
        return reuse
    out = os.path.join(d, "e39_%s.bin" % arm.lower())
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-6s %s" % (arm, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", str(K_IN_FILE)]
    if rank:
        cmd += ["--rank", str(rank)]
    log("  build %-6s %s rank=%d  (~5.5 GB)" % (arm, shape, rank))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    txt = r.stdout.decode(errors="replace")
    if r.returncode != 0:
        log(txt[-3000:])
        log(r.stderr.decode(errors="replace")[-3000:])
        raise SystemExit("synth_export failed for " + arm)
    for ln in txt.splitlines():
        if "GATE V3" in ln or ln.startswith("wrote") or "--rank" in ln or "--carve W" in ln:
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e39")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ntok", type=int, default=NTOK)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--a10b", default="D:/_ktmp/e36/e36_a10b.bin")
    # E36's run-2 rule, now standing procedure: a fixed arm order under within-rep drift charges
    # the drift to the arms at one end, which is the slope the verdict stands on.  The reversed
    # run's only question is whether the k-slope depends on arm order; agreement leaves the
    # verdict unchanged NO MATTER WHAT NUMBER IT PRINTS.
    ap.add_argument("--order", default="forward", choices=("forward", "reversed"))
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    out_json = os.path.join(RES, "e39_same_ten_billion%s.json"
                            % ("_order_reversed" if a.order == "reversed" else ""))
    t0 = time.time()

    log("== E39: the same ten billion, put somewhere else ==")
    log("  SPEED ONLY and the weights are NOISE -- same limit as E36, stated in the brief s0.")

    files, meta = {}, {}
    for arm in ("A10B", "R512", "R0", "R4096"):
        files[arm] = build(arm, a.dir, a.a10b if arm == "A10B" else None)
        meta[arm] = {"file": files[arm], "shape": ARMS[arm][0], "rank": ARMS[arm][1],
                     "role": ARMS[arm][2], "bytes": os.path.getsize(files[arm]),
                     "total_weights": total(arm), "charged_k3": charged(arm, VERDICT_K)}

    # ---- G-E39A: it is still ten billion, and the FILE says so.
    ga = {}
    for arm in ("A10B", "R512"):
        h = E1.read_header(files[arm])
        recomputed = SX.total_weights(h["D"], h["F"], h["L"], h["NH"], h["NKV"], h["HD"],
                                      h["V"], bool(h["tied"]), ARMS[arm][1])
        ga[arm] = {"from_header": recomputed, "expected": TEN_B,
                   "agrees": bool(recomputed == TEN_B)}
    ga["fires"] = bool(all(v["agrees"] for v in ga.values() if isinstance(v, dict)))
    log("")
    log("  G-E39A  A10B %d == R512 %d == %d ?  -> %s"
        % (ga["A10B"]["from_header"], ga["R512"]["from_header"], TEN_B,
           "FIRES" if ga["fires"] else "VOID"))

    # ---- G-E39B: charged accounting against the brief's closed form, zero tolerance.
    gb = {"per_k": {}, "closed_form": "419430400 + 36962304*k"}
    for k in sorted(set(KS + [VERDICT_K])):
        h = E1.read_header(files["R512"])
        ex = SX.active_weights(h["D"], h["F"], h["L"], h["NH"], h["NKV"], h["HD"], h["V"],
                               E_GROUPS, k, 512)
        gb["per_k"][str(k)] = {"from_header": ex, "closed_form": closed_form_r512(k),
                               "agrees": bool(ex == closed_form_r512(k))}
    gb["fires"] = bool(all(v["agrees"] for v in gb["per_k"].values()))
    log("  G-E39B  charged == %s at every k, zero tolerance -> %s"
        % (gb["closed_form"], "FIRES" if gb["fires"] else "VOID"))

    out = {"brief": "BRIEF_E39 (pre-registered)", "arms": meta, "G_E39A": ga, "G_E39B": gb,
           "order": a.order, "reps": a.reps, "ntok": a.ntok, "ten_b_bar": TEN_B}
    if a.build_only:
        out["seconds"] = time.time() - t0
        json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
        log("  --build-only: wrote %s  [%.0fs]" % (out_json, out["seconds"]))
        return 0

    # ---- the timing.  Reps OUTERMOST so drift spreads across arms instead of pooling in one.
    plan = [("R512", k) for k in KS] + [("A10B", VERDICT_K), ("R0", VERDICT_K),
                                        ("R4096", VERDICT_K)]
    if a.order == "reversed":
        plan = plan[::-1]
    log("")
    log("  timing: %d arms x %d reps x %d tokens, reps OUTERMOST, order %s"
        % (len(plan), a.reps, a.ntok, a.order))
    rates, busy = {}, []
    for rep in range(1, a.reps + 1):
        b = cpu_busy()
        busy.append(b)
        for arm, k in plan:
            r = bench(a.engine, winpath(files[arm]), a.ntok, a.threads,
                      ["--carve-k", str(k)])
            rates.setdefault("%s_k%d" % (arm, k), []).append(r)
            log("    rep %d  %-6s k=%-2d %8.2f tok/s" % (rep, arm, k, r))
        log("         box %.1f%% busy" % b)

    arms_out = {}
    for key, rs in rates.items():
        arm, k = key.rsplit("_k", 1)
        k = int(k)
        m = sum(rs) / len(rs)
        c = charged(arm, k)
        arms_out[key] = {"arm": arm, "k": k, "rates": rs, "mean_tok_s": m,
                         "median_tok_s": sorted(rs)[len(rs) // 2],
                         "spread": (max(rs) - min(rs)) / m, "charged": c,
                         "charged_G_w_per_s": c * m / 1e9}
    out["speed_arms"] = arms_out
    out["cpu_busy_pct"] = busy

    # ---- G-E39C: a rank that moves MORE weights must COST more.
    r4 = arms_out.get("R4096_k%d" % VERDICT_K)
    r0 = arms_out.get("R0_k%d" % VERDICT_K)
    gc = None
    if r4 and r0:
        gc = {"rank4096_tok_s": r4["mean_tok_s"], "dense_tok_s": r0["mean_tok_s"],
              "rank4096_charged": r4["charged"], "dense_charged": r0["charged"],
              "fires": bool(r4["mean_tok_s"] < r0["mean_tok_s"])}
        log("")
        log("  G-E39C  rank 4096 %.2f tok/s < dense q/o %.2f tok/s ? -> %s"
            % (r4["mean_tok_s"], r0["mean_tok_s"], "FIRES" if gc["fires"] else "VOID"))
        if not gc["fires"]:
            log("          a rank that costs MORE did not cost more: the timing is not")
            log("          measuring the factored path.  No speed number below this counts.")
    out["G_E39C"] = gc

    # ---- prediction 5: is this session comparable to E36's?
    ctl = arms_out.get("A10B_k%d" % VERDICT_K)
    if ctl:
        rel = ctl["mean_tok_s"] / E36_A10B_K3 - 1.0
        out["session_vs_e36"] = {"a10b_k3_now": ctl["mean_tok_s"], "e36": E36_A10B_K3,
                                 "rel": rel, "within_5pct": bool(abs(rel) <= E36_TOL)}
        log("  session check  A10B k=3 now %.2f vs E36's %.2f  (%+.1f%%) -> %s"
            % (ctl["mean_tok_s"], E36_A10B_K3, 100 * rel,
               "comparable" if abs(rel) <= E36_TOL else "NOT comparable; ratios only"))

    # ---- the verdict, and the crossing, which E36 found was worth more than the cell.
    v = arms_out.get("R512_k%d" % VERDICT_K)
    if v:
        name = next(n for lo, n in BANDS if v["mean_tok_s"] >= lo)
        out["verdict"] = {"cell": "R512_k%d" % VERDICT_K, "tok_s": v["mean_tok_s"],
                          "name": name, "vs_a10b": v["mean_tok_s"] / E36_A10B_K3}
        log("")
        log("  VERDICT  R512 k=%d = %.2f tok/s  (%.2fx A10B's 49.96) -> %s"
            % (VERDICT_K, v["mean_tok_s"], v["mean_tok_s"] / E36_A10B_K3, name))

    ks = [k for k in KS if "R512_k%d" % k in arms_out]
    if len(ks) >= 2:
        import numpy as np
        t = np.array([1000.0 / arms_out["R512_k%d" % k]["mean_tok_s"] for k in ks])
        A = np.vstack([np.ones(len(ks)), np.array(ks, float)]).T
        (c0, c1), *_ = np.linalg.lstsq(A, t, rcond=None)
        grp = closed_form_r512(1) - closed_form_r512(0)
        fit = {"ms_base": c0, "ms_per_group": c1,
               "base_G_w_per_s": closed_form_r512(0) / c0 / 1e6,
               "group_G_w_per_s": grp / c1 / 1e6,
               "gather_penalty": (grp / c1) / (closed_form_r512(0) / c0)}
        for goal, nm in ((EXCELLENT, "k_star_100"), (GOAL, "k_star_50")):
            ks_ = (1000.0 / goal - c0) / c1
            fit[nm] = ks_
            fit[nm + "_neurons"] = ks_ * (shape_of("A10B-R512")["F"] // E_GROUPS)
        out["slope_fit"] = fit
        log("  slope    time = %.3f ms + %.4f ms/group   base %.1f G-w/s, group %.1f, "
            "penalty %.3f" % (c0, c1, fit["base_G_w_per_s"], fit["group_G_w_per_s"],
                              fit["gather_penalty"]))
        log("  CROSSING 100 tok/s at k* = %.3f  (%.0f of %d neurons a layer)"
            % (fit["k_star_100"], fit["k_star_100_neurons"], shape_of("A10B-R512")["F"]))

    out["seconds"] = time.time() - t0
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (out_json, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
