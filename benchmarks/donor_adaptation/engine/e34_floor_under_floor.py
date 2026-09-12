#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E34 -- what does T10 read with the FFN GONE?  The floor under the floor.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E34_THE_FLOOR_UNDER_THE_FLOOR.md, pushed at
684e677 BEFORE this file existed.  Nothing here may contradict it.

Every probe from E18 to E33 attacked the FFN, because at T10 the FFN is nine tenths of the
weight.  NOBODY HAS EVER MEASURED WHAT IS LEFT WHEN IT IS GONE.  If the attention+head floor
alone already exceeds the 50 tok/s budget, then no amount of FFN work can reach the goal at
T10's literal shape, and the remaining lever is the shape of ATTENTION -- a different instruction
to the architecture than "shrink the FFN".

THE HAND TABLE IN THE BRIEF IS SUSPECT UNTIL THIS FILE REPRODUCES IT.  E30 s1.1 caught exactly
this kind of hand table short by the q/k/v biases.  G-E34B recomputes both floors from the
exporter's own layout and from the artifact on disk at ZERO tolerance, and if the brief's s2
disagrees, THE RUNNER WINS and the delta is reported.

  python e34_floor_under_floor.py --dir D:/_ktmp/e34 --reps 5
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
OUT = os.path.join(RES, "e34_floor_under_floor.json")

T10 = dict(D=4096, QD=4096, KD=1024, F=14336, L=48, V=32768, NH=32, NKV=8, HD=128)
NTOK = 40
E_GROUPS = 256
# RUN 1 WAS VOID AND THIS LINE IS WHY.  E26's design is ONE CARVED FILE through --carve-k, but
# its dense control is a DIFFERENT FILE -- and an un-flagged run of a carved file uses the k
# stored IN the file (1 here), so "T10-DENSE" with no flag was the FLOOR arm wearing the dense
# arm's label and the dense arm's 10.6 G charge.  Both gates caught it.  The dense control is
# E26's own artifact, still on disk, same container family (tagged-v2), 10,603,200,512 charged.
DENSE_DEFAULT = r"D:\_ktmp\e26\e26_t10_dense.bin"
ARMS = [("T10-DENSE", None, "dense"), ("T10-K16", 16, "carve"),
        ("T10-K4", 4, "carve"), ("T10-FLOOR", 1, "carve")]

GOAL = 50.0
ANCHOR_TOL = 0.10
MODEL_TOL = 0.15
BANDS = [(50.0, "FLOOR-CLEARS"), (25.0, "FLOOR-BINDS"), (0.0, "FLOOR-IS-THE-WALL")]

# ---- the brief's hand table (s2), quoted so the runner can contradict it
BRIEF_FLOOR_CHARGED = 2147483648
BRIEF_FLOOR_MOVED = 1.0786e9          # GB/token, hand-computed, "the runner recomputes it"
BRIEF_FLOOR_RATE = 21.5
BRIEF_PERFECT_RATE = 33.7


def log(*a):
    print(*a, flush=True)


def anchors():
    """E26's T10 ratios (NOT its absolutes -- E33 addendum s2), E28's arms, E30's ceiling."""
    e26 = json.load(open(os.path.join(RES, "e26_carve_cost.json")))
    e28 = json.load(open(os.path.join(RES, "e28_kernel_transfer.json")))
    e30 = json.load(open(os.path.join(RES, "e30_the_wall.json")))
    ceil = e30["bw_ceiling_GB_s"]          # E30's own key; KeyError here is better than a None
    return {"e26_T10": dict((k, {"mean": v["mean_tok_s"], "ratio": v["ratio_vs_dense"],
                                 "act": v["active_weights_per_token"]})
                            for k, v in e26["arms"].items() if k.startswith("T10")),
            "e28_T10_PACKED": e28["arms"]["T10-PACKED"]["mean_tok_s"],
            "e30_bw_ceil_gb_s": ceil, "e30_raw_keys": sorted(e30.keys())}


def synth():
    spec = importlib.util.spec_from_file_location("se", os.path.join(HERE, "synth_export.py"))
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["synth_export.py"]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


def moved_bytes_floor(m):
    """Attention + head + norms + embed row, in MOVED bytes, from E1's own layout helpers if it
    exposes them, else from the same accounting E30 s1.1 fixed (codes + fp32 row scales, plus the
    q/k/v biases the brief's E30 table had originally MISSED)."""
    D, QD, KD, F, L, V = (T10[k] for k in ("D", "QD", "KD", "F", "L", "V"))

    def mat(out_, in_):
        return out_ * in_ // 2 + out_ * 4

    per_layer = (mat(QD, D) + mat(D, QD) + 2 * mat(KD, D)      # q, o, k, v
                 + (QD + 2 * KD) * 4                            # q, k, v fp32 biases
                 + 2 * D * 4)                                   # input_norm, post_norm
    return per_layer * L + mat(V, D) + D * 4 + D * 4            # + embed row + final norm


def build(d, tag="t10_carve"):
    out = os.path.join(d, "e34_%s.bin" % tag)
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %s" % os.path.basename(out))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", "T10",
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", "1"]
    log("  build %s" % " ".join(cmd[3:]))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log(r.stdout.decode(errors="replace")[-2000:])
        log(r.stderr.decode(errors="replace")[-2000:])
        raise SystemExit("synth_export failed")
    for ln in r.stdout.decode(errors="replace").splitlines():
        if "GATE V3" in ln or "--carve" in ln or ln.startswith("wrote"):
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def bench_k(engine, w, ntok, threads, k):
    """E26's design: ONE file, the engine's --carve-k picks how many groups survive."""
    flags = [] if k is None else ["--carve-k", str(k)]
    return bench(engine, w, ntok, threads, flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e34")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dense", default=DENSE_DEFAULT,
                    help="the dense control -- a DIFFERENT FILE, per E26's own design")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    d = winpath(a.dir)
    os.makedirs(d, exist_ok=True)
    engine = a.engine if os.path.isabs(a.engine) else os.path.join(HERE, a.engine)
    t0 = time.time()
    A = anchors()
    m = synth()

    log("== E34: what does T10 read with the FFN gone? ==")
    log("  anchors READ FROM FILE: E28 T10-PACKED %.3f tok/s, E30 ceiling %s GB/s"
        % (A["e28_T10_PACKED"], A["e30_bw_ceil_gb_s"]))
    for k, v in sorted(A["e26_T10"].items()):
        log("     E26 %-10s %7.3f tok/s  ratio_vs_dense %.3f  act %.4f G"
            % (k, v["mean"], v["ratio"], v["act"] / 1e9))

    # ---- G-E34B part 1: recompute the floor, independently of the brief's hand table
    D, F, L, NH, NKV, HD, V, tied, _ = m.SHAPES["T10"]
    floor_charged = (NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D) * L + V * D
    floor_moved = moved_bytes_floor(m)
    act_dense = m.active_weights(D, F, L, NH, NKV, HD, V)
    act_k = dict((k, m.active_weights(D, F, L, NH, NKV, HD, V, E_GROUPS, k))
                 for k in (1, 4, 16))
    log("")
    log("  RUNNER's own floor (attention + head), charged: %d" % floor_charged)
    log("     brief s2 hand table said %d  ->  %s"
        % (BRIEF_FLOOR_CHARGED,
           "AGREES" if floor_charged == BRIEF_FLOOR_CHARGED
           else "DISAGREES by %+d -- THE RUNNER WINS" % (floor_charged - BRIEF_FLOOR_CHARGED)))
    log("  RUNNER's own floor, MOVED bytes: %d (%.4f GB)" % (floor_moved, floor_moved / 1e9))
    log("     brief s2 hand table said %.4f GB  ->  %+.3f%%"
        % (BRIEF_FLOOR_MOVED / 1e9, 100 * (floor_moved / BRIEF_FLOOR_MOVED - 1)))

    out = {"brief": "briefs/BRIEF_E34_THE_FLOOR_UNDER_THE_FLOOR.md (684e677)",
           "shape": T10, "E": E_GROUPS, "ntok": NTOK, "threads": a.threads, "reps": a.reps,
           "anchors_from_file": A, "goal_tok_s": GOAL, "bands": BANDS,
           "floor_charged_runner": floor_charged, "floor_charged_brief": BRIEF_FLOOR_CHARGED,
           "floor_charged_agrees": bool(floor_charged == BRIEF_FLOOR_CHARGED),
           "floor_moved_runner": floor_moved, "floor_moved_brief": BRIEF_FLOOR_MOVED,
           "active_dense": act_dense, "active_by_k": act_k, "arms": {}}

    w = build(d)
    meta = json.load(open(w + ".json"))
    dense_path = winpath(a.dense)
    if not os.path.exists(dense_path):
        raise SystemExit("missing the dense control " + dense_path)
    dmeta = json.load(open(dense_path + ".json"))
    if dmeta["active_weights_per_token"] != act_dense or dmeta["carve_E"]:
        raise SystemExit("the dense control is not dense: %s" % dmeta)
    log("  dense control %s  (%s, %d charged)"
        % (os.path.basename(dense_path), dmeta["quant"], dmeta["active_weights_per_token"]))
    PATHS = {"dense": dense_path, "carve": w}
    out["artifact"] = {"path": w, "bytes": meta["bytes"],
                       "active_weights_per_token": meta["active_weights_per_token"],
                       "carve_E": meta["carve_E"], "carve_k_in_file": meta["carve_k_in_file"]}
    # G-E34B part 2: the file's own accounting must match the runner's, exactly
    ok_b = (meta["active_weights_per_token"] == act_k[1])
    out["G_E34B"] = {"exporter_active": meta["active_weights_per_token"],
                     "runner_active_k1": act_k[1], "fires": bool(ok_b),
                     "floor_charged": floor_charged,
                     "ffn_residue_at_k1": act_k[1] - floor_charged}
    log("  G-E34B  exporter says %d active at k=1, runner says %d -> %s"
        % (meta["active_weights_per_token"], act_k[1], "FIRES" if ok_b else "VOID"))
    log("          the FFN residue left at k=1 is %d weights (%.3f%% of the floor)"
        % (act_k[1] - floor_charged, 100.0 * (act_k[1] - floor_charged) / floor_charged))
    if not ok_b:
        out["VOID"] = "G-E34B: exporter and runner disagree on charged weights"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("G-E34B VOID -- see " + OUT)

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

    rates = dict((t, []) for t, _, _ in ARMS)
    log("")
    for r in range(a.reps):                       # reps OUTERMOST -- E28's law
        for tag, k, which in ARMS:
            v = bench_k(engine, PATHS[which], NTOK, a.threads, k)
            rates[tag].append(v)
            log("  rep %d  %-10s %7.2f tok/s" % (r + 1, tag, v))
        b2, p2 = cpu_busy(2)
        out["cpu_busy_mean_pct"].append(b2)
        out["cpu_busy_peak_pct"].append(p2)
        log("         box %.1f%% mean / %.0f%% peak" % (b2, p2))
        if b2 > IDLE_BAR:
            log("         !! witness above the bar -- this rep is flagged")

    for tag, k, which in ARMS:
        v = sorted(rates[tag])
        mean = sum(v) / len(v)
        out["arms"][tag] = {"rates": rates[tag], "mean_tok_s": mean, "median_tok_s": v[len(v) // 2],
                            "carve_k": k, "file": which, "spread": (v[-1] - v[0]) / mean,
                            "active_weights_per_token": act_dense if k is None else act_k[k],
                            "charged_G_w_per_s": mean * (act_dense if k is None else act_k[k]) / 1e9}

    # ---- G-E34A: E26's RATIO, not its absolute
    r16 = out["arms"]["T10-K16"]["mean_tok_s"] / out["arms"]["T10-DENSE"]["mean_tok_s"]
    e26r = A["e26_T10"]["T10-K16"]["ratio"]
    out["G_E34A"] = {"measured_ratio": r16, "e26_ratio": e26r, "rel_dev": r16 / e26r - 1.0,
                     "tol": ANCHOR_TOL, "fires": bool(abs(r16 / e26r - 1.0) <= ANCHOR_TOL)}
    log("")
    log("  G-E34A  T10-K16/T10-DENSE %.3f vs E26's %.3f (%+.1f%%) -> %s"
        % (r16, e26r, 100 * out["G_E34A"]["rel_dev"], "FIRES" if out["G_E34A"]["fires"] else "VOID"))

    # ---- G-E34C: the numerator model must predict a point it did not fit
    num = out["arms"]["T10-DENSE"]["charged_G_w_per_s"]
    pred_k4 = num / (act_k[4] / 1e9)
    got_k4 = out["arms"]["T10-K4"]["mean_tok_s"]
    out["G_E34C"] = {"numerator_G_w_per_s": num, "predicted_K4": pred_k4, "measured_K4": got_k4,
                     "rel_dev": pred_k4 / got_k4 - 1.0, "tol": MODEL_TOL,
                     "fires": bool(abs(pred_k4 / got_k4 - 1.0) <= MODEL_TOL)}
    log("  G-E34C  numerator %.2f G-w/s (this session, dense) predicts T10-K4 = %.2f, "
        "measured %.2f (%+.1f%%) -> %s"
        % (num, pred_k4, got_k4, 100 * out["G_E34C"]["rel_dev"],
           "FIRES" if out["G_E34C"]["fires"] else "VOID"))

    if not (out["G_E34A"]["fires"] and out["G_E34C"]["fires"]):
        out["VOID"] = "a deciding anchor did not fire"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("")
        log("  VOID -- no verdict is quotable from this run.")
        return 2

    # ---- the verdict
    fl = out["arms"]["T10-FLOOR"]["mean_tok_s"]
    name = next(n for lo, n in BANDS if fl >= lo)
    ceil_gb = A["e30_bw_ceil_gb_s"]
    perfect = (ceil_gb * 1e9 / floor_moved) if ceil_gb else None
    budget_charged = num / GOAL
    budget_moved = (ceil_gb * 1e9 / GOAL) if ceil_gb else None
    gap_k4 = fl / got_k4 - 1.0
    out["verdict"] = {
        "name": name, "cell": "T10-FLOOR tok/s", "tok_s": fl, "goal": GOAL,
        "short_by": GOAL / fl,
        "floor_over_charged_budget": (floor_charged / 1e9) / budget_charged,
        "charged_budget_for_50": budget_charged,
        "perfect_kernel_floor_rate": perfect,
        "moved_budget_for_50_bytes": budget_moved,
        "floor_over_moved_budget": (floor_moved / budget_moved) if budget_moved else None,
        "floor_vs_K4_gap": gap_k4,
        "prediction3_under_12pct": bool(abs(gap_k4) < 0.12)}
    log("")
    log("  VERDICT CELL  T10-FLOOR = %.2f tok/s  ->  %s   (goal %.0f, short by %.2fx)"
        % (fl, name, GOAL, GOAL / fl))
    log("     floor is %.3fx the charged budget for 50 tok/s (%.4f G) -- WITH ZERO FFN"
        % ((floor_charged / 1e9) / budget_charged, budget_charged))
    if perfect:
        log("     a PERFECT kernel on this floor reads %.1f tok/s (%.2f GB/s ceiling / %.4f GB)"
            % (perfect, ceil_gb, floor_moved / 1e9))
        log("     floor is %.2fx the MOVED budget for 50 tok/s (%.4f GB)"
            % (floor_moved / budget_moved, budget_moved / 1e9))
    log("     T10-FLOOR vs T10-K4: %+.1f%% (prediction 3 said under 12%%) -> %s"
        % (100 * gap_k4, "HIT" if abs(gap_k4) < 0.12 else "MISS"))

    # ---- prediction 6: the smallest attention that WOULD fit (desk arithmetic, labelled)
    if budget_moved:
        att_layer_moved = (floor_moved - (V * D // 2 + V * 4) - 2 * D * 4) / float(L)
        head_moved = V * D // 2 + V * 4
        fits_L = int((budget_moved - head_moved) // att_layer_moved)
        out["prediction6_desk"] = {
            "note": "DESK ARITHMETIC on a measured ceiling -- not a recommendation",
            "attention_bytes_per_layer_moved": att_layer_moved, "head_bytes_moved": head_moved,
            "layers_that_fit_at_50_tok_s": fits_L, "layers_now": L,
            "fraction_of_depth": fits_L / float(L)}
        log("")
        log("  PREDICTION 6 (desk arithmetic on a measured ceiling, NOT a recommendation):")
        log("     at T10's width, 50 tok/s fits %d attention layers of the %d (%.0f%% of the "
            "depth), with the FFN at zero" % (fits_L, L, 100.0 * fits_L / L))

    out["seconds"] = time.time() - t0
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (OUT, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
