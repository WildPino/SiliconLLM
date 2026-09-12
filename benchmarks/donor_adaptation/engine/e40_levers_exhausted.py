#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E40 -- after every attention lever is pulled, how much FFN can 50 tok/s buy at ten billion?

Brief: docs/research/donor_adaptation/briefs/BRIEF_E40_HOW_MUCH_FFN_CAN_FIFTY_TOK_S_BUY.md,
pushed BEFORE this file existed and before the exporter could build the objects.  Nothing here
may contradict it.

E39 put the floor at 99.7 tok/s and measured 50 tok/s buying 4.25-4.78% FFN activation at a
genuine 9,999,220,736 parameters.  E38 measured the value of selection PEAKING at 25% activation
and reading 3.597 BPB at 6.25% against a dense 0.768.  This probe asks whether ANY attention
shape at ten billion on this box buys a fraction worth selecting at -- and if not, the T4 clause
in the standing goal has exactly one thing left to spend itself on.

Three arms, F solved so each hits E36's integer EXACTLY with F % 256 == 0:

  R512   F=48128  NKV=8  untied  rank 512    E39's own artifact, re-timed: the control
  NKV2   F=48640  NKV=2  untied  rank 512    one lever
  R128   F=49152  NKV=2  untied  rank 128    every lever that is REAL on this engine (addendum A)

  python e40_levers_exhausted.py --build-only     # exports + gates A and B, no timing
  python e40_levers_exhausted.py --reps 5         # the measurement (idle box)
  python e40_levers_exhausted.py --reps 5 --order reversed
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy                                  # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402
import synth_export as SX                                            # noqa: E402
import e1_bpb_through_engine as E1                                   # noqa: E402

RES = os.path.join(HERE, "results")

GOAL, EXCELLENT = 50.0, 100.0
E_GROUPS = 256
KS = [1, 2, 3, 4, 6]
K_IN_FILE = 3
TEN_B = 9999220736                                  # E36's integer, to the parameter
NTOK = 40

# Brief s3.  The cell is ALL's FFN activation fraction at 50 tok/s, from the two-term fit.
BANDS = [(25.0, "SPEED-BUYS-THE-PEAK"), (10.0, "HALFWAY-TO-THE-PEAK"),
         (6.0, "LEVERS-NEARLY-EXHAUSTED"), (0.0, "ATTENTION-LEVERS-EXHAUSTED")]
VERDICT_ARM, VERDICT_K = "R128", 3
E39_R512_K3 = 78.456                                # brief s4: session comparability, +-5%
E39_TOL = 0.05

# name -> (shape name in SHAPES, rank, role)
ARMS = {
    "R512": ("A10B-R512",   512, "E39's object, re-timed: the control and the session check"),
    "NKV2": ("A10B-NKV2",   512, "one lever: k/v heads 8 -> 2, nothing else moves"),
    # Addendum A: ALL (NKV 2 + tied + rank 256) was WITHDRAWN when G-E40A read 10,133,438,464 --
    # over by V*D exactly.  synth_export.py:207 forces tied=0 for the runnable ternary head, and a
    # tied head would have COST speed (fp32 embedding, 52% of the token at S05).  R128 pulls every
    # lever that is real on this engine and is a STRONGER arm than the one it replaces.
    "R128": ("A10B-R128",   128, "addendum A: every REAL lever -- NKV 2 + q/o rank 128"),
}

# Brief s5 G-E40B: written here from each shape's own dimensions, INDEPENDENTLY of the exporter,
# so a disagreement is a real disagreement and not the same arithmetic twice.
def closed_form(arm, k):
    sh = shape_of(ARMS[arm][0])
    D, F, L, NH, NKV, HD, V = (sh[x] for x in ("D", "F", "L", "NH", "NKV", "HD", "V"))
    rank = ARMS[arm][1]
    QO, KD = NH * HD, NKV * HD
    qo = 2 * (rank * (QO + D)) if rank else 2 * (QO * D)
    attn = qo + 2 * (KD * D)
    ffn = E_GROUPS * D + 3 * D * (F // E_GROUPS) * k          # router + k activated groups
    return (attn + ffn) * L + V * D


def log(*a):
    print(*a, flush=True)


def shape_of(name):
    s = SX.SHAPES[name]
    return dict(zip(("D", "F", "L", "NH", "NKV", "HD", "V", "tied"), s[:8]))


def charged(arm, k):
    sh = shape_of(ARMS[arm][0])
    return SX.active_weights(sh["D"], sh["F"], sh["L"], sh["NH"], sh["NKV"], sh["HD"], sh["V"],
                             E_GROUPS, k, ARMS[arm][1])


def total_from_header(path, arm):
    h = E1.read_header(path)
    return SX.total_weights(h["D"], h["F"], h["L"], h["NH"], h["NKV"], h["HD"], h["V"],
                            bool(h["tied"]), ARMS[arm][1])


def build(arm, d, reuse):
    shape, rank, _ = ARMS[arm]
    if arm == "R512" and reuse and os.path.exists(reuse):
        log("  have  R512  <- %s  (E39's own artifact, not rebuilt)" % reuse)
        return reuse
    out = os.path.join(d, "e40_%s.bin" % arm.lower())
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-5s %s" % (arm, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", shape,
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", str(K_IN_FILE), "--rank", str(rank)]
    log("  build %-5s %s rank=%d" % (arm, shape, rank))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    txt = r.stdout.decode(errors="replace")
    if r.returncode != 0:
        log(txt[-3000:])
        log(r.stderr.decode(errors="replace")[-3000:])
        raise SystemExit("synth_export failed for " + arm)
    for ln in txt.splitlines():
        if "GATE V3" in ln or ln.startswith("wrote") or "--rank" in ln:
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e40")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ntok", type=int, default=NTOK)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--r512", default="D:/_ktmp/e39/e39_r512.bin")
    ap.add_argument("--order", default="forward", choices=("forward", "reversed"))
    ap.add_argument("--refit", action="store_true",
                    help="recompute fits and the verdict from an existing results JSON without "
                         "re-timing.  Run 1's fit loop still named the arm addendum A withdrew, "
                         "so R128 was timed but not fitted; the RATES are untouched by this.")
    a = ap.parse_args()

    os.makedirs(a.dir, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    out_json = os.path.join(RES, "e40_levers_exhausted%s.json"
                            % ("_order_reversed" if a.order == "reversed" else ""))
    t0 = time.time()

    log("== E40: after every attention lever, how much FFN can 50 tok/s buy? ==")
    log("  SPEED ONLY and the weights are NOISE -- same limit as E36 and E39, brief s0.")

    files, meta = {}, {}
    for arm in ARMS:
        files[arm] = build(arm, a.dir, a.r512)
        h = E1.read_header(files[arm])
        meta[arm] = {"file": files[arm], "shape": ARMS[arm][0], "rank": ARMS[arm][1],
                     "role": ARMS[arm][2], "bytes": os.path.getsize(files[arm]),
                     "header": {k: h[k] for k in ("D", "F", "L", "NH", "NKV", "HD", "V", "tied")},
                     "total_from_header": total_from_header(files[arm], arm),
                     "charged_k3": charged(arm, VERDICT_K)}

    # ---- G-E40A: all three are still ten billion, read back from their own headers.
    ga = {"per_arm": {arm: meta[arm]["total_from_header"] for arm in ARMS}, "bar": TEN_B}
    ga["fires"] = bool(all(v == TEN_B for v in ga["per_arm"].values()))
    log("")
    log("  G-E40A  %s == %d ?  -> %s"
        % (" == ".join("%s %d" % (k, v) for k, v in sorted(ga["per_arm"].items())),
           TEN_B, "FIRES" if ga["fires"] else "VOID"))

    # ---- G-E40B: charged accounting against a closed form written here, zero tolerance.
    gb = {"per_arm": {}}
    for arm in ARMS:
        rows = {}
        for k in sorted(set(KS + [VERDICT_K])):
            ex, cf = charged(arm, k), closed_form(arm, k)
            rows[str(k)] = {"from_header": ex, "closed_form": cf, "agrees": bool(ex == cf)}
        gb["per_arm"][arm] = rows
    gb["fires"] = bool(all(r["agrees"] for arm in gb["per_arm"].values() for r in arm.values()))
    log("  G-E40B  charged == the runner's own closed form at every k, zero tolerance -> %s"
        % ("FIRES" if gb["fires"] else "VOID"))

    out = {"brief": "BRIEF_E40 (pre-registered)", "arms": meta, "G_E40A": ga, "G_E40B": gb,
           "order": a.order, "reps": a.reps, "ntok": a.ntok, "ten_b_bar": TEN_B}
    if a.build_only:
        out["seconds"] = time.time() - t0
        json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
        log("  --build-only: wrote %s  [%.0fs]" % (out_json, out["seconds"]))
        return 0

    if a.refit:
        prev = json.load(open(out_json, encoding="utf-8"))
        arms_out = prev["speed_arms"]
        out = prev
        log("")
        log("  --refit: %d timed cells reloaded from %s, nothing re-timed"
            % (len(arms_out), os.path.basename(out_json)))
        return finish(out, arms_out, out_json, t0)

    # ---- the timing.  Reps OUTERMOST so drift spreads across arms instead of pooling in one.
    plan = [(arm, k) for arm in ("R512", "NKV2", "R128") for k in KS]
    if a.order == "reversed":
        plan = plan[::-1]
    log("")
    log("  timing: %d arms x %d reps x %d tokens, reps OUTERMOST, order %s"
        % (len(plan), a.reps, a.ntok, a.order))
    rates, busy, peaks = {}, [], []
    for rep in range(1, a.reps + 1):
        b, pk = cpu_busy()
        busy.append(b)
        peaks.append(pk)
        for arm, k in plan:
            r = bench(a.engine, winpath(files[arm]), a.ntok, a.threads, ["--carve-k", str(k)])
            rates.setdefault("%s_k%d" % (arm, k), []).append(r)
            log("    rep %d  %-5s k=%-2d %8.2f tok/s" % (rep, arm, k, r))
        log("         box %.1f%% busy, peak core %.1f%%" % (b, pk))

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
    out["cpu_peak_pct"] = peaks

    return finish(out, arms_out, out_json, t0)


def finish(out, arms_out, out_json, t0):
    # ---- G-E40C: NKV2 charges strictly less than R512 at every k, so it must be FASTER at
    #      every k.  A shape that moves less weight and does not go faster means the stopwatch
    #      is not on the shape.
    gc = {"per_k": {}}
    for k in KS:
        r5, n2 = arms_out.get("R512_k%d" % k), arms_out.get("NKV2_k%d" % k)
        if not (r5 and n2):
            continue
        assert n2["charged"] < r5["charged"], "G-E40C is not even well posed at k=%d" % k
        gc["per_k"][str(k)] = {"r512": r5["mean_tok_s"], "nkv2": n2["mean_tok_s"],
                               "faster": bool(n2["mean_tok_s"] > r5["mean_tok_s"])}
    gc["fires"] = bool(gc["per_k"] and all(v["faster"] for v in gc["per_k"].values()))
    log("")
    log("  G-E40C  NKV2 faster than R512 at every k ? -> %s  (%s)"
        % ("FIRES" if gc["fires"] else "VOID",
           " ".join("k%s:%s" % (k, "+" if v["faster"] else "-")
                    for k, v in sorted(gc["per_k"].items(), key=lambda x: int(x[0])))))
    if not gc["fires"]:
        log("          a shape that moves LESS weight did not go faster: the stopwatch is not")
        log("          on the shape.  No speed number below this counts.")
    out["G_E40C"] = gc

    # ---- brief s4: is this session comparable to E39's?
    ctl = arms_out.get("R512_k%d" % VERDICT_K)
    if ctl:
        rel = ctl["mean_tok_s"] / E39_R512_K3 - 1.0
        out["session_vs_e39"] = {"r512_k3_now": ctl["mean_tok_s"], "e39": E39_R512_K3,
                                 "rel": rel, "within_5pct": bool(abs(rel) <= E39_TOL)}
        log("  session check  R512 k=3 now %.2f vs E39's %.2f  (%+.1f%%) -> %s"
            % (ctl["mean_tok_s"], E39_R512_K3, 100 * rel,
               "comparable" if abs(rel) <= E39_TOL else "NOT comparable; ratios only"))

    # ---- the two-term fit per arm, and the cell: ALL's activation fraction at 50 tok/s.
    import numpy as np
    fits = {}
    for arm in ("R512", "NKV2", "R128"):
        ks = [k for k in KS if "%s_k%d" % (arm, k) in arms_out]
        if len(ks) < 2:
            continue
        t = np.array([1000.0 / arms_out["%s_k%d" % (arm, k)]["mean_tok_s"] for k in ks])
        A = np.vstack([np.ones(len(ks)), np.array(ks, float)]).T
        (c0, c1), *_ = np.linalg.lstsq(A, t, rcond=None)
        F = shape_of(ARMS[arm][0])["F"]
        gsz = F // E_GROUPS
        grp = closed_form(arm, 1) - closed_form(arm, 0)
        f = {"ms_base": c0, "ms_per_group": c1, "F": F, "group_size": gsz,
             "floor_tok_s": 1000.0 / c0,
             "base_G_w_per_s": closed_form(arm, 0) / c0 / 1e6,
             "group_G_w_per_s": grp / c1 / 1e6,
             "gather_penalty": (grp / c1) / (closed_form(arm, 0) / c0)}
        for goal, nm in ((EXCELLENT, "k_star_100"), (GOAL, "k_star_50")):
            ks_ = (1000.0 / goal - c0) / c1
            f[nm] = ks_
            f[nm + "_neurons"] = ks_ * gsz
            f[nm + "_pct"] = 100.0 * ks_ * gsz / F
        fits[arm] = f
        log("  %-5s time = %.3f ms + %.4f ms/group   floor %6.1f tok/s   base %.1f G-w/s, "
            "group %.1f" % (arm, c0, c1, f["floor_tok_s"], f["base_G_w_per_s"],
                            f["group_G_w_per_s"]))
        log("        50 tok/s buys %6.2f%% of F   |   100 tok/s buys %6.2f%%"
            % (f["k_star_50_pct"], f["k_star_100_pct"]))
    out["fits"] = fits

    if VERDICT_ARM in fits:
        pct = fits[VERDICT_ARM]["k_star_50_pct"]
        name = next(n for lo, n in BANDS if pct >= lo)
        out["verdict"] = {"cell": "%s activation %% at 50 tok/s" % VERDICT_ARM,
                          "pct": pct, "name": name}
        log("")
        log("  VERDICT  %s buys %.2f%% FFN activation at 50 tok/s -> %s"
            % (VERDICT_ARM, pct, name))
        log("           (E38: the value of selection PEAKS at 25%% and reads 3.597 BPB at 6.25%%)")

    out["seconds"] = time.time() - t0
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (out_json, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
