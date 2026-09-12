#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E36 -- ten billion parameters at fifty tokens a second, or the number that says why not.

Brief: docs/research/donor_adaptation/briefs/BRIEF_E36_TEN_BILLION_AT_FIFTY.md, pushed at
1738aa2 BEFORE this file existed.  Nothing here may contradict it.

Thirty-five probes priced the goal's sentence.  E35 closed the pricing: at 50 tok/s this box
affords 0.9451 G ACTIVE charged weights a token, and L=16 at 4096 wide reads 56.16 with its FFN
at the floor.  What has never happened is the obvious thing: build a file that is ACTUALLY ten
billion parameters, size its ACTIVE slice to that envelope, and run it.

Every 10 B-shaped artifact this programme has benched (T10, 10.74 B) is DENSE-ACTIVE: 10.6 G
charged a token, 11x over the envelope.  A10B is 9.999 B on disk and 0.93 G charged at k=3.
That gap -- 10.8x between what the file WEIGHS and what a token COSTS -- is the whole claim,
and it is the thing the engine has to actually deliver rather than the thing arithmetic says.

  python e36_ten_billion.py --build-only        # export + the three gates, no timing
  python e36_ten_billion.py --reps 5            # the measurement (idle box, operator idle)
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
OUT = os.path.join(RES, "e36_ten_billion.json")

GOAL = 50.0
E_GROUPS = 256
KS = [1, 2, 3, 4, 6]
K_IN_FILE = 3                     # the verdict cell's k, baked so an UNFLAGGED run is harmless
NTOK = 40                         # E35's, unchanged, so the control is comparable token for token
ANCHOR_TOL = 0.10
TEN_B_BAR = 9.9e9
BANDS = [(50.0, "TEN-B-AT-FIFTY"), (45.0, "TEN-B-NEAR-FIFTY"), (0.0, "TEN-B-SHORT")]
VERDICT_K = 3

BRIEF_PRED = {1: 55.1, 2: 52.9, 3: 50.9, 4: 49.0, 6: 45.7}
BRIEF_BAND = [49.0, 54.0]                                      # prediction 2, on the k=3 arm

CONTROL = ("T10-L16", r"D:\_ktmp\e35\e35_t10l16.bin")


def closed_form(k):
    """Brief s2's charged model, quoted here so the measurement can contradict MY arithmetic
    and not the other way round.  Checked against synth_export.active_weights() before the
    brief was pushed -- the E33 lesson: a gate that cannot be satisfied is worse than no gate."""
    return 822083584 + 35389440 * k


def log(*a):
    print(*a, flush=True)


def synth():
    spec = importlib.util.spec_from_file_location("se", os.path.join(HERE, "synth_export.py"))
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["synth_export.py"]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


def build(d):
    out = os.path.join(d, "e36_a10b.bin")
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  A10B  %s" % os.path.basename(out))
        return out
    cmd = [sys.executable, os.path.join(HERE, "synth_export.py"), "--shape", "A10B",
           "--out", out, "--codes", "mixed", "--seed", "1234",
           "--carve", str(E_GROUPS), "--carve-k", str(K_IN_FILE)]
    log("  build A10B  (this writes ~5.4 GB; E35's 2.2 GB arm took 99 s)")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True)
    txt = r.stdout.decode(errors="replace")
    if r.returncode != 0:
        log(txt[-3000:])
        log(r.stderr.decode(errors="replace")[-3000:])
        raise SystemExit("synth_export failed for A10B")
    for ln in txt.splitlines():
        if "GATE V3" in ln or ln.startswith("wrote") or "--carve" in ln:
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e36")
    ap.add_argument("--engine", default="./donor_engine_e26.exe")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--build-only", action="store_true")
    # ---- RUN 2 IS AN ORDER CONTROL ON MY OWN INSTRUMENT, NOT A SECOND CHANCE AT THE VERDICT.
    #
    # Run 1 read the verdict cell at 49.96 tok/s against a 50.0 bar, and its arms ran in a FIXED
    # order inside every rep (control, k=1, k=2, k=3, k=4, k=6).  The box drifted WITHIN reps --
    # rep 1 opened with the fastest control reading of the whole run (62.33) and closed with the
    # slowest k=6 (40.60) -- and a fixed order under a within-rep drift charges the drift to the
    # high-k arms, which is exactly the slope the verdict stands on.  --order reversed runs the
    # same arms back to front so the drift lands on the LOW-k arms instead.
    #
    # THE RULE, FIXED AND PUSHED BEFORE RUN 2 EXISTS, because a second run taken after seeing a
    # verdict miss its bar by 0.04 is the textbook way to fish:
    #   * Run 1 is the registered measurement.  Its verdict stands.
    #   * Run 2's ONLY question is whether the k-slope depends on arm order.
    #   * If the two slopes agree inside the reps' own dispersion, the verdict is UNCHANGED --
    #     run 2 may not promote 49.96 over the bar no matter what number it prints.
    #   * If they disagree, BOTH runs are reported and the verdict cell is declared unresolvable
    #     at this dispersion.  It is not promoted either way.
    ap.add_argument("--order", default="forward", choices=("forward", "reversed"),
                    help="arm order inside each rep; see the rule above")
    ap.add_argument("--out", default=None, help="override the result path (run 2 writes its own)")
    a = ap.parse_args()
    d = winpath(a.dir)
    os.makedirs(d, exist_ok=True)
    engine = a.engine if os.path.isabs(a.engine) else os.path.join(HERE, a.engine)
    t0 = time.time()
    m = synth()
    import e1_bpb_through_engine as E1

    # ---- anchors READ FROM FILE.  Nothing about E35 is typed in here as a constant.
    #
    # DO NOT USE e35["verdict"]["active_budget_at_50_G"].  That key is MISLABELLED: it holds
    # E34's dense numerator divided by 50 (0.9348 G), not the envelope E35's own s4 publishes
    # and this brief's s2 predicted against (0.9451 G, from the numerator INTERPOLATED between
    # the two arms that bracket the goal).  The two differ by 1.1% and the brief is built on
    # the second, so reading the first would score every arm against a model nobody registered.
    # The interpolation is recomputed here from E35's recorded ARMS, which are the raw record.
    e35 = json.load(open(os.path.join(RES, "e35_envelope.json")))
    ctrl_ref = e35["arms"]["T10-L16"]["mean_tok_s"]
    pts35 = sorted((A["charged"], A["mean_tok_s"], A["L"]) for A in e35["arms"].values())
    br = [(pts35[i], pts35[i + 1]) for i in range(len(pts35) - 1)
          if (pts35[i][1] - GOAL) * (pts35[i + 1][1] - GOAL) <= 0]
    if len(br) != 1:
        raise SystemExit("E35's arms no longer bracket %.0f tok/s exactly once" % GOAL)
    (c_lo, r_lo, L_lo), (c_hi, r_hi, L_hi) = br[0]
    num35 = (r_lo * c_lo + r_hi * c_hi) / 2.0               # E35 s4's interpolated numerator
    envelope = num35 / GOAL
    num35_e34key = e35["verdict"]["active_budget_at_50_G"] * GOAL * 1e9

    D, F, L, NH, NKV, HD, V, tied, src = m.SHAPES["A10B"]
    tied_eff = 0                                            # --head ternary unties, as E35 did
    tot = m.total_weights(D, F, L, NH, NKV, HD, V, tied_eff)

    log("== E36: ten billion parameters at fifty tokens a second ==")
    log("  anchors READ FROM FILE: E35 T10-L16 %.3f tok/s, envelope %.4f G, numerator %.3f G-w/s"
        % (ctrl_ref, envelope / 1e9, num35 / 1e9))
    log("     (interpolated between E35's L=%d at %.2f and L=%d at %.2f -- NOT the mislabelled"
        " JSON key, which holds %.3f G-w/s)"
        % (L_lo, r_lo, L_hi, r_hi, num35_e34key / 1e9))
    log("  A10B  D=%d F=%d L=%d V=%d  ->  %d weights on disk (%.4f B)"
        % (D, F, L, V, tot, tot / 1e9))
    log("  carve E=%d, group=%d neurons.  The model, registered in the brief:"
        % (E_GROUPS, F // E_GROUPS))
    for k in KS:
        ch = closed_form(k)
        log("     k=%d  %5d of %d neurons (%.2f%%)  charged %10d (%.4f G)  predicted %6.2f "
            "tok/s (brief s2 said %.1f)"
            % (k, k * (F // E_GROUPS), F, 100.0 * k * (F // E_GROUPS) / F, ch, ch / 1e9,
               num35 / ch, BRIEF_PRED[k]))

    out = {"brief": "briefs/BRIEF_E36_TEN_BILLION_AT_FIFTY.md (1738aa2)",
           "goal_tok_s": GOAL, "E": E_GROUPS, "ks": KS, "k_in_file": K_IN_FILE, "ntok": NTOK,
           "threads": a.threads, "reps": a.reps, "bands": BANDS, "verdict_k": VERDICT_K,
           "e35_control_tok_s": ctrl_ref, "e35_envelope_w": envelope,
           "numerator_w_per_s": num35,
           "numerator_source": "E35 s4, interpolated between L=%d (%.4f) and L=%d (%.4f)"
                               % (L_lo, r_lo, L_hi, r_hi),
           "numerator_from_mislabelled_e35_key": num35_e34key,
           "brief_predicted": BRIEF_PRED,
           "brief_band_k3": BRIEF_BAND,
           "shape": dict(zip(("D", "F", "L", "NH", "NKV", "HD", "V"),
                             (D, F, L, NH, NKV, HD, V))),
           "total_weights": int(tot), "arms": {}}

    log("")
    path = build(d)
    if not os.path.exists(CONTROL[1]):
        raise SystemExit("the planted control artifact is missing: " + CONTROL[1])
    meta = json.load(open(path + ".json"))

    # ---- G-E36B: the artifact must actually BE ten billion.
    #      Three independent readings, because "10 B" is the entire claim of this probe:
    #        1. the exporter's own count, from the file's JSON;
    #        2. the SHAPE READ BACK OUT OF THE FILE HEADER, recomputed here;
    #        3. the byte count on disk against E1's independently written v4 layout -- which is
    #           what proves the tensors are PRESENT and not merely declared.
    hdr = E1.read_header(path)
    hdr_tot = m.total_weights(hdr["D"], hdr["F"], hdr["L"], hdr["NH"], hdr["NKV"],
                              hdr["HD"], hdr["V"], hdr["tied"])
    want_bytes = E1.layout_bytes_v4(hdr["D"], hdr["F"], hdr["L"], hdr["NH"], hdr["NKV"],
                                    hdr["HD"], hdr["V"], hdr["tied"], E_GROUPS)
    got_bytes = os.path.getsize(path)
    gb = {"exporter_total": meta.get("total_weights"), "header_total": int(hdr_tot),
          "bar": TEN_B_BAR, "bytes_on_disk": got_bytes, "bytes_e1_layout": int(want_bytes),
          "header": dict((k, hdr[k]) for k in ("D", "F", "L", "NH", "NKV", "HD", "V", "tied",
                                               "quant"))}
    gb["fires"] = bool(gb["exporter_total"] == gb["header_total"] >= TEN_B_BAR
                       and got_bytes == want_bytes)
    out["G_E36B"] = gb
    log("")
    log("  G-E36B  exporter %s  header %d  bar %.2f G  |  bytes %d vs E1 layout %d  -> %s"
        % (gb["exporter_total"], gb["header_total"], TEN_B_BAR / 1e9, got_bytes,
           want_bytes, "FIRES" if gb["fires"] else "VOID"))

    # ---- G-E36C: charged accounting, zero tolerance, at every k.
    gc, ok_c = {}, True
    for k in KS:
        ex = m.active_weights(D, F, L, NH, NKV, HD, V, E_GROUPS, k)
        cf = closed_form(k)
        same = (ex == cf)
        ok_c = ok_c and same
        gc[str(k)] = {"exporter": int(ex), "closed_form": int(cf), "agrees": bool(same)}
    file_k = meta["active_weights_per_token"] == closed_form(K_IN_FILE)
    ok_c = ok_c and file_k
    gc["file_json_at_k%d" % K_IN_FILE] = {"exporter": meta["active_weights_per_token"],
                                          "closed_form": closed_form(K_IN_FILE),
                                          "agrees": bool(file_k)}
    out["G_E36C"] = {"per_k": gc, "fires": bool(ok_c)}
    for k in KS:
        log("  G-E36C  k=%d  exporter %10d  brief %10d -> %s"
            % (k, gc[str(k)]["exporter"], gc[str(k)]["closed_form"],
               "OK" if gc[str(k)]["agrees"] else "MISMATCH"))
    log("  G-E36C  file JSON at k=%d: %d vs %d -> %s"
        % (K_IN_FILE, meta["active_weights_per_token"], closed_form(K_IN_FILE),
           "OK" if file_k else "MISMATCH"))

    if not (gb["fires"] and ok_c):
        out["VOID"] = "G-E36B and/or G-E36C did not fire before any timing was taken"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID before timing -- see " + OUT)

    if a.build_only:
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  --build-only: no timing taken")
        return 0

    # ---- timing.  E34's void run 1 was caused by an UNFLAGGED run of a carved file inheriting
    #      the k stored in it.  Every arm here passes --carve-k explicitly, including the one
    #      whose k equals the file's.
    arms = [(CONTROL[0], CONTROL[1], [], None)] + \
           [("A10B-K%d" % k, path, ["--carve-k", str(k)], k) for k in KS]
    if a.order == "reversed":
        arms = arms[::-1]
    out["order"] = a.order

    busy, peak = cpu_busy(4)
    log("")
    log("  contention witness: %.1f%% mean / %.0f%% peak (bar %.0f%% on the MEAN)"
        % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)
    out["cpu_busy_mean_pct"], out["cpu_busy_peak_pct"] = [busy], [peak]

    rates = dict((t, []) for t, _, _, _ in arms)
    log("")
    for r in range(a.reps):                        # reps OUTERMOST
        for tag, w, flags, k in arms:
            v = bench(engine, w, NTOK, a.threads, flags)
            rates[tag].append(v)
            log("  rep %d  %-9s %7.2f tok/s" % (r + 1, tag, v))
        b2, p2 = cpu_busy(2)
        out["cpu_busy_mean_pct"].append(b2)
        out["cpu_busy_peak_pct"].append(p2)
        log("         box %.1f%% mean / %.0f%% peak%s"
            % (b2, p2, "   !! ABOVE THE BAR" if b2 > IDLE_BAR else ""))

    for tag, w, flags, k in arms:
        v = sorted(rates[tag])
        mean = sum(v) / len(v)
        ch = closed_form(k) if k else e35["arms"]["T10-L16"]["charged"]
        rec = {"rates": rates[tag], "mean_tok_s": mean, "median_tok_s": v[len(v) // 2],
               "spread": (v[-1] - v[0]) / mean, "charged": int(ch),
               "charged_G_w_per_s": mean * ch / 1e9}
        if k:
            rec.update({"k": k, "neurons_active": k * (F // E_GROUPS),
                        "activation_pct": 100.0 * k * (F // E_GROUPS) / F,
                        "predicted_tok_s": num35 / ch,
                        "rel_dev_vs_model": mean / (num35 / ch) - 1.0,
                        "brief_predicted": BRIEF_PRED[k]})
        out["arms"][tag] = rec

    # ---- G-E36A: the planted control
    got = out["arms"][CONTROL[0]]["mean_tok_s"]
    out["G_E36A"] = {"measured": got, "e35": ctrl_ref, "rel_dev": got / ctrl_ref - 1.0,
                     "tol": ANCHOR_TOL, "fires": bool(abs(got / ctrl_ref - 1.0) <= ANCHOR_TOL)}
    log("")
    log("  G-E36A  %s %.2f vs E35's %.2f (%+.1f%%) -> %s"
        % (CONTROL[0], got, ctrl_ref, 100 * out["G_E36A"]["rel_dev"],
           "FIRES" if out["G_E36A"]["fires"] else "VOID"))
    if not out["G_E36A"]["fires"]:
        out["VOID"] = "G-E36A: this session is not comparable to the envelope it is testing"
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
        log("  VOID -- no verdict is quotable from this run.")
        return 2

    log("")
    for k in KS:
        A = out["arms"]["A10B-K%d" % k]
        log("  k=%d  %5d/%d neurons (%.2f%%)  measured %6.2f   model %6.2f   %+6.1f%%   "
            "numerator %.3f G-w/s   spread %.1f%%"
            % (k, A["neurons_active"], F, A["activation_pct"], A["mean_tok_s"],
               A["predicted_tok_s"], 100 * A["rel_dev_vs_model"], A["charged_G_w_per_s"],
               100 * A["spread"]))

    # ---- prediction 3: does a GSZ=180 carve read FASTER per charged weight than E35's GSZ=56?
    nums = [out["arms"]["A10B-K%d" % k]["charged_G_w_per_s"] for k in KS]
    mean_num = sum(nums) / len(nums)
    out["numerator"] = {"a10b_mean_G_w_per_s": mean_num, "e35_G_w_per_s": num35 / 1e9,
                        "ratio": mean_num / (num35 / 1e9),
                        "faster_than_model": bool(mean_num > num35 / 1e9)}
    log("")
    log("  prediction 3: A10B's numerator %.3f G-w/s vs E35's %.3f  (%+.1f%%) -> %s"
        % (mean_num, num35 / 1e9, 100 * (out["numerator"]["ratio"] - 1.0),
           "FASTER, as registered" if out["numerator"]["faster_than_model"] else "SLOWER"))

    # ---- the verdict cell
    vk = out["arms"]["A10B-K%d" % VERDICT_K]["mean_tok_s"]
    name = next(nm for bar, nm in BANDS if vk >= bar)
    kstar = None
    pts = [(out["arms"]["A10B-K%d" % k]["charged"], out["arms"]["A10B-K%d" % k]["mean_tok_s"], k)
           for k in KS]
    for i in range(len(pts) - 1):
        if (pts[i][1] - GOAL) * (pts[i + 1][1] - GOAL) <= 0:
            n_star = (pts[i][1] * pts[i][0] + pts[i + 1][1] * pts[i + 1][0]) / 2.0
            kstar = (n_star / GOAL - 822083584.0) / 35389440.0
    out["verdict"] = {
        "name": name, "cell": "A10B-K%d" % VERDICT_K, "tok_s": vk,
        "in_brief_band": bool(BRIEF_BAND[0] <= vk <= BRIEF_BAND[1]),
        "k_star": kstar,
        "neurons_at_50": (kstar * (F // E_GROUPS)) if kstar else None,
        "activation_pct_at_50": (100.0 * kstar * (F // E_GROUPS) / F) if kstar else None,
        "total_weights": int(tot),
        "bracketed": bool(out["arms"]["A10B-K1"]["mean_tok_s"] >= GOAL
                          > out["arms"]["A10B-K6"]["mean_tok_s"])}
    log("")
    log("  VERDICT CELL  A10B-K%d = %.2f tok/s  ->  %s   (brief predicted %.1f, band [%g, %g])"
        % (VERDICT_K, vk, name, BRIEF_PRED[VERDICT_K], BRIEF_BAND[0], BRIEF_BAND[1]))
    if kstar:
        log("  crossing: k* = %.2f  ->  %.0f of %d neurons (%.2f%%) at exactly %.0f tok/s"
            % (kstar, out["verdict"]["neurons_at_50"], F,
               out["verdict"]["activation_pct_at_50"], GOAL))
    log("  the file weighs %.4f B parameters and a token costs %.4f G of them: %.1fx"
        % (tot / 1e9, closed_form(VERDICT_K) / 1e9, tot / float(closed_form(VERDICT_K))))
    log("")
    log("  REGISTERED BEFORE THE RUN (brief prediction 5): this is the SPEED half only.")
    log("  The weights are noise.  Nothing here says a model of this shape can be TRAINED,")
    log("  adapted from a donor, or is any good.  Quality is E24/E27/E29's axis, untouched.")

    # ---- the slope fit.  time(k) = a + b*k, least squares over the five arms.  `a` is what a
    # token costs before a single carve group is read (attention + head + router, 822,083,584
    # charged); `b` is what ONE group of 180 neurons costs (35,389,440 charged).  Their ratio is
    # the gather penalty at GSZ=180, measured PER GROUP instead of inferred from a curve fitted
    # at another granularity -- and it is the number run 2's order control exists to protect,
    # because a within-rep drift under a fixed arm order lands entirely on this slope.
    xs = [float(k) for k in KS]
    ys = [1000.0 / out["arms"]["A10B-K%d" % k]["mean_tok_s"] for k in KS]      # ms per token
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    bb = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    aa = my - bb * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (aa + bb * x)) ** 2 for x, y in zip(xs, ys))
    base_rate = 822083584.0 / (aa / 1000.0)
    grp_rate = 35389440.0 / (bb / 1000.0)
    out["slope_fit"] = {"ms_at_k0": aa, "ms_per_group": bb,
                        "r2": (1.0 - ss_res / ss_tot) if ss_tot else None,
                        "base_G_w_per_s": base_rate / 1e9, "group_G_w_per_s": grp_rate / 1e9,
                        "gather_penalty_at_gsz180": grp_rate / base_rate}
    log("")
    log("  slope fit  time = %.3f + %.4f*k ms  (R2 %.4f)" % (aa, bb, out["slope_fit"]["r2"]))
    log("     base (attention+head+router, 0.8221 G) reads %.3f G-w/s" % (base_rate / 1e9))
    log("     ONE carve group of 180 neurons (0.0354 G) reads %.3f G-w/s" % (grp_rate / 1e9))
    log("     gather penalty at GSZ=180: %.3f of the base rate"
        % out["slope_fit"]["gather_penalty_at_gsz180"])

    out["seconds"] = time.time() - t0
    dst = a.out or OUT
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=1)
    log("")
    log("  wrote %s  [%.0fs]" % (dst, out["seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
