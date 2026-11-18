#!/usr/bin/env python3
# Sparse Capacity, phase 1a (D-0): the verdict, from the runner's numbers, without choosing anything.
#   python benchmarks/sparse_capacity/sc1a_verdict.py results/sparse_capacity/f1a_<stamp>
# Rules: docs/research/SPARSE_CAPACITY/F1A_RUNG1_GATE_PROTOCOL.md, section 3 (registered before the reading).
import json, os, sys

SIGMA = 0.005            # R1 calibration, sigma_seed of BPB
PLANT_TOL = 0.002        # the recomputed internal val must match the recorded stage-exit BPB


def main():
    d = sys.argv[1]
    S = {r["stage"]: r for r in map(json.loads, open(os.path.join(d, "stages.jsonl")))}
    cfg = json.load(open(os.path.join(d, "config.json")))
    bad = []
    for s, r in S.items():
        if abs(r["val_bpb"] - r["recorded_val_bpb"]) > PLANT_TOL:
            bad.append(f"stage {s}: internal val {r['val_bpb']:.5f} vs recorded {r['recorded_val_bpb']:.5f}")
        if r["p62_bytes_scored"] + r["p62_first_tok_bytes"] != cfg["p62_bytes"]:
            bad.append(f"stage {s}: P62 bytes {r['p62_bytes_scored']} + {r['p62_first_tok_bytes']} != {cfg['p62_bytes']}")
    if set(S) != {"D", "E", "F"}:
        bad.append(f"stages found: {sorted(S)}")
    print("  stage  step     params    val(rec)  val(now)  P62 BPB   top-1    T fit  P62-B T=1  P62-B T fit")
    for s in sorted(S):
        r = S[s]
        print(f"  {s}      {r['gstep']:<7}  {r['params'] / 1e6:6.2f} M  {r['recorded_val_bpb']:.5f}   {r['val_bpb']:.5f}   "
              f"{r['p62_bpb']:.5f}   {r['p62_top1']:.4f}   {r['t_fit']:.2f}   {r['p62_B_bpb_t1']:.5f}    {r['p62_B_bpb_tfit']:.5f}")
    if bad:
        print("\nVALIDITY:", "; ".join(bad)); print("VERDICT: VOID"); return
    dE = S["E"]["p62_bpb"] - S["D"]["p62_bpb"]
    dF = S["F"]["p62_bpb"] - S["D"]["p62_bpb"]
    dE_T = S["E"]["p62_B_bpb_tfit"] - S["D"]["p62_B_bpb_tfit"]
    print(f"\n  E - D on P62: {dE:+.5f} BPB = {dE / SIGMA:+.1f} sigma (half B, fitted T: {dE_T:+.5f}); "
          f"top-1 {S['E']['p62_top1'] - S['D']['p62_top1']:+.4f}")
    print(f"  F - D on P62: {dF:+.5f} BPB = {dF / SIGMA:+.1f} sigma (reading: F is annealed at lr x0.1, D is not); "
          f"top-1 {S['F']['p62_top1'] - S['D']['p62_top1']:+.4f}")
    print("\nVALIDITY: ok")
    if dE > 2 * SIGMA:
        print("VERDICT: BANDIERA ROSSA (the bank stage E is worse than dense D by more than 2 sigma on P62): "
              "talk before phase 2 (PLAN, phase 1a)")
    elif dE < -2 * SIGMA:
        print("VERDICT: VERDE (E better than D by more than 2 sigma): phase 2 starts")
    else:
        print("VERDICT: PARI (E within 2 sigma of D): phase 2 starts")


if __name__ == "__main__":
    main()
