#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E37 -- what E36's speed costs a model that was actually trained.

Brief: BRIEF_E37_WHAT_THE_SPEED_COSTS_A_TRAINED_MODEL.md (3072941)
Addendum: BRIEF_E37_ADDENDUM_TWO_GATES_AS_WRITTEN_ARE_WRONG.md (8cc405a)
Both pushed before this file existed.  Nothing here may contradict them.

THE GAP THIS CLOSES.  E21/E22/E23/E24/E27/E29 measured real trained donors and explicitly took
NO TIMING.  E25/E26B/E28/E30-E36 took every timing on synthetic noise.  No probe in this
programme has ever put a stopwatch and a BPB on the SAME file, so every "inside budget" claim
on the quality side is a weight-count claim priced with a numerator E32 revoked and E36 has
since split into two rates.

THE COINCIDENCE THAT MAKES IT ANSWERABLE AT 1.5 B.  S15 has F=8960 and the label set on disk
cuts it into E=256 groups of 35, so k=3 here is 3/256 = 1.17% -- the SAME activation rate A10B
needed at 10 B in E36.  E36's requirement can be put to a genuinely trained model.

TWO PHASES, because the two halves obey different laws:
  --phase quality   BPB through the engine.  DETERMINISTIC -> may be taken on a loaded box.
  --phase speed     tok/s.  Idle box, operator idle, reps outermost.  Never both at once.

  python e37_sparsity_cost.py --phase build
  python e37_sparsity_cost.py --phase quality
  python e37_sparsity_cost.py --phase speed --reps 5
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from e26_carve_cost import cpu_busy, IDLE_BAR                        # noqa: E402
from e28_kernel_transfer import bench, winpath                       # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "e37_sparsity_cost.json")
ENGINE = os.path.join(HERE, "donor_engine_e26.exe")

HF = "Qwen/Qwen2.5-1.5B"
REV = "8faed761d45a263340a0528343f099c05c9a4323"      # the anchor artifact's own revision
D, F, L, NH, NKV, HD, V = 1536, 8960, 28, 12, 2, 128, 151936
E_GROUPS = 256
GSZ = F // E_GROUPS                                   # 35
RULE, CALIB_SEQS = "R3", 32

# E1's protocol and convention, unchanged -- the same constants E32 used.
SEQLEN = 512
SCORED_BYTES, N_PRED = 51870, 12264
BYTES_PER_TOK = SCORED_BYTES / float(N_PRED)
CHANCE_BPB = math.log(V, 2) / BYTES_PER_TOK           # 4.069819

ANCHOR_BIN = r"D:\_ktmp\e1\qwen25-15b_tqh.bin"        # --fold layers, the published artifact
ANCHOR_IDS = r"D:\_ktmp\e1\ids_qwen25-15b_tqh.bin"
ANCHOR_BPB, ANCHOR_TOL = 3.475706372, 0.001           # G-E37B1

PARITY_TOL_BPB = 1e-4        # G-E37A': e26_parity_carve.py:49's TOL_P, adopted not invented
PARITY_TOL_L2 = 1e-4
PARITY_TOKENS = 8

KS = [256, 64, 32, 16, 8, 4, 3, 2, 1]
VERDICT_K = 3
BANDS = [(3.70, "SPARSITY-SURVIVES"), (CHANCE_BPB, "SPARSITY-DEGRADES"),
         (float("inf"), "SPARSITY-DESTROYS")]
NTOK, ANCHOR_SPEED_TOL = 40, 0.10


def log(*a):
    print(*a, flush=True)


def charged(k):
    """attention*L + head + (router + kept FFN)*L.  The runner's own arithmetic; G-E37C
    checks the exporter against it at zero tolerance."""
    attn = (NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D) * L
    ffn = (E_GROUPS * D + 3 * D * GSZ * k) * L
    return attn + ffn + V * D


def band(bpb):
    for bar, nm in BANDS:
        if bpb <= bar:
            return nm
    return BANDS[-1][1]


def run_bpb(weights, flags):
    cmd = [ENGINE, "--weights", weights, "--threads", "6", "--seqlen", str(SEQLEN),
           "--bpb", ANCHOR_IDS] + list(flags)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine --bpb failed (%d): %s" % (r.returncode, r.stderr[-2000:]))
    m = re.search(r"NATS_PER_TOKEN\s+([0-9.]+)", r.stdout)
    p = re.search(r"N_PREDICTED\s+(\d+)", r.stdout)
    if not m:
        raise SystemExit("no NATS_PER_TOKEN:\n" + r.stdout[-2000:])
    nats = float(m.group(1))
    npred = int(p.group(1)) if p else None
    if npred != N_PRED:
        raise SystemExit("N_PREDICTED %r != %d -- not E1's protocol" % (npred, N_PRED))
    return {"nats_per_token": nats, "bpb": nats / math.log(2) / BYTES_PER_TOK,
            "n_predicted": npred, "seconds": time.time() - t0}


def run_logits(weights, flags, out, n=PARITY_TOKENS):
    """The engine's own --logits, on the first n ids of the pinned slice."""
    idf = os.path.join(os.path.dirname(out), "parity_ids.bin")
    if not os.path.exists(idf):
        import numpy as np
        np.fromfile(ANCHOR_IDS, dtype="<i4")[:n].tofile(idf)
    cmd = [ENGINE, "--weights", weights, "--threads", "6"] + list(flags) + \
          ["--logits", idf, str(n), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("engine --logits failed: " + r.stderr[-2000:])
    return out


def build(d, tag, extra):
    out = os.path.join(d, "e37_%s.bin" % tag)
    if os.path.exists(out) and os.path.exists(out + ".json"):
        log("  have  %-12s %s" % (tag, os.path.basename(out)))
        return out
    cmd = [sys.executable, os.path.join(HERE, "qwen_export.py"),
           "--model", HF, "--revision", REV, "--rule", RULE, "--head-ternary",
           "--calib-seqs", str(CALIB_SEQS), "--fold", "none", "--out", out] + extra
    log("  build %-12s %s" % (tag, " ".join(extra)))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-3000:])
        log(r.stderr[-3000:])
        raise SystemExit("qwen_export failed for " + tag)
    for ln in r.stdout.splitlines():
        if "carved" in ln or ln.startswith("wrote") or "GATE" in ln:
            log("        " + ln.strip())
    log("        [%.0fs]" % (time.time() - t0))
    return out


def load(path):
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(path) else {}


def save(o):
    json.dump(o, open(OUT, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="D:/_ktmp/e37")
    ap.add_argument("--routers", default="D:/_ktmp/e37/e37_routers_E256.npz")
    ap.add_argument("--labels",
                    default=os.path.abspath(os.path.join(
                        HERE, "..", "density", "results", "d0c_labels", "labels_E256.npz")))
    ap.add_argument("--phase", required=True, choices=("build", "quality", "speed"))
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    d = winpath(a.dir)
    os.makedirs(d, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    out = load(OUT)
    out.setdefault("brief", "BRIEF_E37 (3072941) + addendum (8cc405a)")
    out.setdefault("protocol", {"seqlen": SEQLEN, "scored_bytes": SCORED_BYTES,
                                "n_pred": N_PRED, "bytes_per_token": BYTES_PER_TOK,
                                "chance_bpb": CHANCE_BPB, "donor": HF, "revision": REV,
                                "rule": RULE, "fold": "none", "E": E_GROUPS, "GSZ": GSZ})
    out.setdefault("bands", [[b, n] for b, n in BANDS])

    dense = os.path.join(d, "e37_dense_nf.bin")
    carved = os.path.join(d, "e37_carved_nf.bin")

    # ================================================================= BUILD
    if a.phase == "build":
        log("== E37 build: two artifacts, --fold none, rule %s ==" % RULE)
        log("  the fold is NOT a choice: qwen_export.py:390 refuses --quant carved unless")
        log("  --fold none, because the labels were fitted on the UNFOLDED donor.  The")
        log("  addendum (8cc405a) registered this and split G-E37B accordingly.")
        if not os.path.exists(a.routers):
            raise SystemExit("fitted routers missing: %s -- run e37_fit_routers.py" % a.routers)
        build(d, "dense_nf", ["--quant", "packed"])
        build(d, "carved_nf", ["--quant", "carved", "--carve-labels", a.labels,
                               "--carve-router", a.routers, "--carve-k", str(E_GROUPS)])

        # ---- G-E37C: charged accounting, zero tolerance, at every k
        meta = json.load(open(carved + ".json"))
        gc, ok = {}, True
        for k in KS:
            want = charged(k)
            gc[str(k)] = {"runner": int(want)}
        want_file = charged(E_GROUPS)
        got_file = meta.get("active_weights_per_token")
        ok = (got_file == want_file)
        out["G_E37C"] = {"per_k": gc, "file_k": E_GROUPS, "exporter": got_file,
                         "runner": int(want_file), "fires": bool(ok)}
        log("")
        log("  G-E37C  exporter %s  runner %d  -> %s"
            % (got_file, want_file, "FIRES" if ok else "MISMATCH"))
        log("  charged/token: k=%d %.4f G   k=%d %.4f G   dense %.4f G"
            % (E_GROUPS, charged(E_GROUPS) / 1e9, VERDICT_K, charged(VERDICT_K) / 1e9,
               ((NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D + 3 * D * F) * L + V * D) / 1e9))
        out["artifacts"] = {"dense_nf": dense, "carved_nf": carved,
                            "anchor": ANCHOR_BIN, "routers": a.routers}
        out["build_seconds"] = time.time() - t0
        save(out)
        return 0 if ok else 2

    # ================================================================= QUALITY
    if a.phase == "quality":
        log("== E37 quality: BPB through the engine.  DETERMINISTIC -- a loaded box is legal ==")
        log("  chance line %.6f at %d scored bytes / %d predictions"
            % (CHANCE_BPB, SCORED_BYTES, N_PRED))
        for p in (dense, carved):
            if not os.path.exists(p):
                raise SystemExit("missing artifact %s -- run --phase build" % p)

        # ---- G-E37B1: the published artifact, unchanged, reproduces the record
        log("")
        log("  G-E37B1  running the PUBLISHED artifact unchanged ...")
        r = run_bpb(ANCHOR_BIN, [])
        b1 = {"bpb": r["bpb"], "anchor": ANCHOR_BPB, "diff": r["bpb"] - ANCHOR_BPB,
              "tol": ANCHOR_TOL, "fires": bool(abs(r["bpb"] - ANCHOR_BPB) <= ANCHOR_TOL),
              "seconds": r["seconds"]}
        out["G_E37B1"] = b1
        log("  G-E37B1  %.9f vs %.9f  (%+.9f)  -> %s  [%.0fs]"
            % (b1["bpb"], ANCHOR_BPB, b1["diff"], "FIRES" if b1["fires"] else "VOID",
               b1["seconds"]))
        if not b1["fires"]:
            out["VOID"] = "G-E37B1: this session does not reproduce the record"
            save(out)
            return 2

        # ---- the fold-matched dense control (G-E37B2) and the measured fold cost
        log("")
        log("  S15-DENSE-NF (--fold none) ...")
        rd = run_bpb(dense, [])
        out["S15_DENSE_NF"] = rd
        fold_cost = rd["bpb"] - b1["bpb"]
        out["fold_cost_bpb"] = fold_cost
        log("  S15-DENSE-NF %.9f   fold cost vs the published --fold layers artifact: %+.9f"
            % (rd["bpb"], fold_cost))
        log("     (REPORTED, NOT GATED -- addendum 8cc405a registered this and did not")
        log("      predict its sign or size.)")

        # ---- G-E37A': parity at k=E, scored the way E26 scored the identical comparison
        log("")
        log("  G-E37A'  parity at k=E=%d (nothing dropped) ..." % E_GROUPS)
        rk = run_bpb(carved, ["--carve-k", str(E_GROUPS)])
        dbpb = abs(rk["bpb"] - rd["bpb"])
        lo_d = run_logits(dense, [], os.path.join(d, "lg_dense.bin"))
        lo_c = run_logits(carved, ["--carve-k", str(E_GROUPS)], os.path.join(d, "lg_carved.bin"))
        import numpy as np
        A = np.fromfile(lo_d, dtype="<f4").reshape(PARITY_TOKENS, -1)
        B = np.fromfile(lo_c, dtype="<f4").reshape(PARITY_TOKENS, -1)
        rel = float(np.max(np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)))
        top1 = float(np.mean(A.argmax(1) == B.argmax(1)))
        ga = {"bpb_carved_kE": rk["bpb"], "bpb_dense": rd["bpb"], "bpb_diff": dbpb,
              "bpb_tol": PARITY_TOL_BPB, "worst_rel_l2": rel, "l2_tol": PARITY_TOL_L2,
              "top1": top1, "tokens": PARITY_TOKENS,
              "fires": bool(dbpb < PARITY_TOL_BPB and rel < PARITY_TOL_L2 and top1 == 1.0)}
        out["G_E37A"] = ga
        log("  G-E37A'  BPB %.9f vs %.9f (|d| %.3e, bar %.0e) | worst rel L2 %.3e (bar %.0e) | "
            "top-1 %.4f -> %s" % (rk["bpb"], rd["bpb"], dbpb, PARITY_TOL_BPB, rel,
                                  PARITY_TOL_L2, top1, "FIRES" if ga["fires"] else "VOID"))
        if not ga["fires"]:
            out["VOID"] = ("G-E37A': the carved container is not inert at k=E, so damage at "
                           "lower k cannot be attributed to the sparsity")
            save(out)
            return 2

        # ---- the sweep
        log("")
        arms = out.get("quality_arms", {})
        arms["K%d" % E_GROUPS] = dict(rk, k=E_GROUPS, charged=charged(E_GROUPS),
                                      activation_pct=100.0 * E_GROUPS / E_GROUPS)
        for k in KS:
            if k == E_GROUPS:
                continue
            rr = run_bpb(carved, ["--carve-k", str(k)])
            arms["K%d" % k] = dict(rr, k=k, charged=charged(k),
                                   activation_pct=100.0 * k / E_GROUPS)
            log("  k=%-4d %5d/%d neurons (%5.2f%%)  charged %.4f G  BPB %.6f  (%+.6f vs dense, "
                "%s chance)  [%.0fs]"
                % (k, k * GSZ, F, 100.0 * k / E_GROUPS, charged(k) / 1e9, rr["bpb"],
                   rr["bpb"] - rd["bpb"], "under" if rr["bpb"] < CHANCE_BPB else "OVER",
                   rr["seconds"]))
            out["quality_arms"] = arms
            save(out)
        out["quality_arms"] = arms

        vb = arms["K%d" % VERDICT_K]["bpb"]
        out["verdict"] = {"cell": "S15-K%d" % VERDICT_K, "bpb": vb, "name": band(vb),
                          "vs_dense": vb - rd["bpb"], "vs_chance": vb - CHANCE_BPB,
                          "activation_pct": 100.0 * VERDICT_K / E_GROUPS}
        # prediction 3's two numbers
        cheap = [k for k in KS if arms["K%d" % k]["bpb"] - rd["bpb"] <= 0.10]
        beats = [k for k in KS if arms["K%d" % k]["bpb"] < CHANCE_BPB]
        out["knee"] = {"largest_k_within_0.10_of_dense": max(cheap) if cheap else None,
                       "smallest_k_within_0.10_of_dense": min(cheap) if cheap else None,
                       "smallest_k_beating_chance": min(beats) if beats else None}
        log("")
        log("  VERDICT CELL  S15-K%d  BPB %.6f  ->  %s   (dense %.6f, chance %.6f)"
            % (VERDICT_K, vb, out["verdict"]["name"], rd["bpb"], CHANCE_BPB))
        log("  knee: largest k within +0.10 of dense = %s ; smallest k beating chance = %s"
            % (out["knee"]["smallest_k_within_0.10_of_dense"] and
               out["knee"]["largest_k_within_0.10_of_dense"],
               out["knee"]["smallest_k_beating_chance"]))
        out["quality_seconds"] = time.time() - t0
        save(out)
        return 0

    # ================================================================= SPEED
    log("== E37 speed: tok/s.  Idle box, operator idle, reps outermost ==")
    for p in (dense, carved):
        if not os.path.exists(p):
            raise SystemExit("missing artifact %s -- run --phase build" % p)
    busy, peak = cpu_busy(4)
    log("  contention witness: %.1f%% mean / %.0f%% peak (bar %.0f%%)" % (busy, peak, IDLE_BAR))
    if busy > IDLE_BAR:
        raise SystemExit("the box is at %.1f%% -- a contended timing is not a timing.  STOP."
                         % busy)
    out["speed_cpu_busy_mean_pct"], out["speed_cpu_busy_peak_pct"] = [busy], [peak]

    arms = [("DENSE-NF", dense, [], None)] + \
           [("K%d" % k, carved, ["--carve-k", str(k)], k) for k in KS]
    rates = dict((t, []) for t, _, _, _ in arms)
    for r in range(a.reps):
        for tag, w, flags, k in arms:
            v = bench(ENGINE, w, NTOK, 6, flags)
            rates[tag].append(v)
            log("  rep %d  %-9s %7.2f tok/s" % (r + 1, tag, v))
        b2, p2 = cpu_busy(2)
        out["speed_cpu_busy_mean_pct"].append(b2)
        out["speed_cpu_busy_peak_pct"].append(p2)
        log("         box %.1f%% mean / %.0f%% peak%s"
            % (b2, p2, "   !! ABOVE THE BAR" if b2 > IDLE_BAR else ""))

    dense_charged = ((NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D + 3 * D * F) * L + V * D)
    sp = {}
    for tag, w, flags, k in arms:
        v = sorted(rates[tag])
        mean = sum(v) / len(v)
        ch = charged(k) if k else dense_charged
        sp[tag] = {"rates": rates[tag], "mean_tok_s": mean, "median_tok_s": v[len(v) // 2],
                   "spread": (v[-1] - v[0]) / mean, "charged": int(ch),
                   "charged_G_w_per_s": mean * ch / 1e9, "k": k}
    out["speed_arms"] = sp
    log("")
    for tag, w, flags, k in arms:
        log("  %-9s %7.2f tok/s   charged %.4f G   numerator %.3f G-w/s   spread %.1f%%"
            % (tag, sp[tag]["mean_tok_s"], sp[tag]["charged"] / 1e9,
               sp[tag]["charged_G_w_per_s"], 100 * sp[tag]["spread"]))

    # ---- the exchange rate: the whole reason both phases exist
    if "quality_arms" in out and "S15_DENSE_NF" in out:
        qa, bd = out["quality_arms"], out["S15_DENSE_NF"]["bpb"]
        sd = sp["DENSE-NF"]["mean_tok_s"]
        xr = {}
        for k in KS:
            q, s = qa.get("K%d" % k), sp.get("K%d" % k)
            if not q or not s:
                continue
            dt, db = s["mean_tok_s"] - sd, q["bpb"] - bd
            xr["K%d" % k] = {"d_tok_s": dt, "d_bpb": db,
                             "bpb_per_tok_s": (db / dt) if dt else None,
                             "speedup": s["mean_tok_s"] / sd}
        out["exchange_rate"] = xr
        log("")
        log("  THE EXCHANGE RATE -- both axes on ONE artifact, the first time in this branch")
        log("  %-6s %8s %10s %10s %12s" % ("k", "tok/s", "speedup", "BPB", "dBPB/dtok/s"))
        for k in KS:
            key = "K%d" % k
            if key in xr:
                log("  %-6d %8.2f %9.2fx %10.6f %12.6f"
                    % (k, sp[key]["mean_tok_s"], xr[key]["speedup"], qa[key]["bpb"],
                       xr[key]["bpb_per_tok_s"] or float("nan")))

    out["speed_seconds"] = time.time() - t0
    save(out)
    log("")
    log("  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
