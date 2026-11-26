#!/usr/bin/env python3
# Sparse Capacity, phase 3 (the method): the verdict, from the stream run and its phase 2 reference, choosing nothing.
#   python benchmarks/sparse_capacity/sc3_verdict.py [results/sparse_capacity]
# Rules: docs/research/SPARSE_CAPACITY/F3_STREAM_PROTOCOL.md, section 5 (registered before the run).
import json, math, os, statistics, sys

SIGMA = 0.005            # sigma_BPB: the R1 floor, the plan's gate
SIGMA_SEED = 0.0019      # phase 2, measured between seeds (pooled over the gate arms)
STRICT = 2 * math.sqrt(2) * SIGMA_SEED   # 2 sd of the difference of two runs: 0.0054 (review item 3), one-sided
STEPS = 7324
PACE_MIN = 0.5           # PLAN, phase 3: pace at least 50% of the GPU-only one
HOST_BYTES = 16          # host bytes per bank parameter: fp32 weight, gradient, two AdamW moments (+ the stage: 4/L)


def jl(p):
    return [json.loads(l) for l in open(p)]


def tok_s(r):
    return r["tokens"] / (r["train_hours"] * 3600)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "..", "results",
                                                                "sparse_capacity")
    rd, sd, m1d = (os.path.join(root, x) for x in ("f2_m16_s0", "f3_stream_m16_s0", "f2_m1_s0"))
    for d in (rd, sd):
        if not os.path.exists(os.path.join(d, "result.json")):
            print(f"VALIDITY: missing {d}"); print("VERDICT: VOID"); return
    R, S = json.load(open(os.path.join(rd, "result.json"))), json.load(open(os.path.join(sd, "result.json")))
    cfg, rcfg = json.load(open(os.path.join(sd, "config.json"))), json.load(open(os.path.join(rd, "config.json")))
    LR_, LS = jl(os.path.join(rd, "train.jsonl")), jl(os.path.join(sd, "train.jsonl"))
    bad = []
    if cfg["mode"] != "RUN" or not cfg["pushed"] or cfg["dirty_files"]: bad.append("stream run not RUN, not pushed or dirty")
    if S["steps"] != STEPS or R["steps"] != STEPS: bad.append("steps")
    if any(r.get("kind") == "nonfinite" for r in LS): bad.append("non-finite stop")
    p0 = [r for r in LS if r.get("kind") == "planted_step0"]
    if not p0 or not p0[0]["ok"]: bad.append(f"planted step 0: {p0[0] if p0 else 'missing'}")
    for f, h in rcfg["src_sha256"].items():
        if cfg["src_sha256"].get(f) != h: bad.append(f"{f} differs from the reference's")
    if cfg["torch"] != rcfg["torch"]: bad.append("torch differs from the reference's")
    if cfg["ts_sha256_first256MiB"] != rcfg["ts_sha256_first256MiB"]: bad.append("data differ from the reference's")
    cr = {r["step"]: r["bpb"] for r in LR_ if r.get("kind") == "curve"}
    cs = {r["step"]: r["bpb"] for r in LS if r.get("kind") == "curve"}
    if sorted(cr) != sorted(cs): bad.append("curve points differ")
    if bad:
        print("VALIDITY:", "; ".join(bad)); print("VERDICT: VOID"); return

    d_bpb = S["val"]["bpb"] - R["val"]["bpb"]
    d_curve = {k: cs[k] - cr[k] for k in sorted(cr)}
    worst = max(abs(v) for v in d_curve.values())
    pace = tok_s(S) / tok_s(R)
    print(f"  BPB val: stream {S['val']['bpb']:.4f}, reference {R['val']['bpb']:.4f}, diff {d_bpb:+.4f} "
          f"= {d_bpb / SIGMA:+.1f} sigma; top-1 {S['val']['top1'] - R['val']['top1']:+.4f}")
    print(f"  temperature control (half B, fitted T): stream {S['val']['bpb_B_tfit']:.4f}, reference "
          f"{R['val']['bpb_B_tfit']:.4f}")
    print("  curve differences (stream - reference):", {k: round(v, 4) for k, v in d_curve.items()})
    print(f"  pace: stream {tok_s(S):.0f} tok/s, reference {tok_s(R):.0f} tok/s, ratio {pace:.3f} "
          f"(hours {S['train_hours']:.2f} vs {R['train_hours']:.2f})")
    if "p62" in S and "p62" in R:
        print(f"  reading P62: stream {S['p62']['bpb']:.4f}, reference {R['p62']['bpb']:.4f}")
    print(f"  reading K primary: stream {S['probes']['k_primary']:.3f}, reference {R['probes']['k_primary']:.3f}; "
          f"R primary: stream {S['probes']['r_primary']:.3f}, reference {R['probes']['r_primary']:.3f}")
    tm = [r["times"] for r in LS if r.get("kind") == "step" and r.get("times")]
    if tm:
        med = {k: statistics.median(t[k] for t in tm) for k in tm[0] if not k.startswith("n_sel")}
        P = cfg["host_bank_params"]
        print("  step time, median over timed steps (s):", {k: round(v, 4) for k, v in med.items()},
              f"| experts per layer in U: min {min(t['n_sel_min'] for t in tm)}, max {max(t['n_sel_max'] for t in tm)}")
        over = med["fetch"] + med["grad_out"] + med["host_adam"]
        ns = over / P * 1e9
        print(f"  method cost per step: fetch {med['fetch']:.3f} ({med['fetches_per_layer']:.0f} per layer) + gradients "
              f"out {med['grad_out']:.3f} + host AdamW {med['host_adam']:.3f} = {over:.3f} s = {ns:.2f} ns per bank "
              f"parameter ({P / 1e6:.1f} M)")
        if os.path.exists(os.path.join(m1d, "result.json")):
            t_m1 = 8192 / tok_s(json.load(open(os.path.join(m1d, "result.json"))))
            for gb in (20, 40):
                Pb = gb * 1e9 / HOST_BYTES
                c = Pb * ns * 1e-9
                print(f"  reading for phase 4 (linear, a LOWER BOUND on the cost: no VRAM pressure, every expert touched): "
                      f"bank + state {gb} GB = {Pb / 1e9:.2f} G parameters -> +{c:.1f} s per step; against m1's GPU step "
                      f"({t_m1:.2f} s, the active compute) the pace would be at most {t_m1 / (t_m1 + c):.2f}")
    print("\nVALIDITY: ok")
    eq_plan = abs(d_bpb) <= 2 * SIGMA and worst <= 2 * SIGMA
    eq_strict = d_bpb <= STRICT
    pc = pace >= PACE_MIN
    print(f"  equivalence (plan): |BPB diff| {abs(d_bpb):.4f} and worst curve diff {worst:.4f} vs 2 sigma "
          f"{2 * SIGMA:.3f} -> {'PASS' if eq_plan else 'FAIL'}")
    print(f"  equivalence (strict, one-sided): BPB diff {d_bpb:+.4f} vs {STRICT:.4f} -> {'PASS' if eq_strict else 'FAIL'}")
    print(f"  pace: {pace:.3f} vs {PACE_MIN} -> {'PASS' if pc else 'FAIL'}")
    if eq_plan and eq_strict and pc:
        print("VERDICT: PASS (the method trains as the GPU-only trainer, at a pace above the gate): phase 4 may be "
              "detailed (PLAN)")
    else:
        fails = [x for x, ok in (("equivalence (plan)", eq_plan), ("equivalence (strict)", eq_strict), ("pace", pc)) if not ok]
        print("VERDICT: FAIL (" + ", ".join(fails) + "): one page with one recommendation (rule 7)")


if __name__ == "__main__":
    main()
