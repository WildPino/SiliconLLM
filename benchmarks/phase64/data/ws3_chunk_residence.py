#!/usr/bin/env python3
"""Stage-3 pre-launch gate: does the resident-chunk window sweep the whole slice, or dwell on a prefix?

WHY (Architect, stage-3 branch-point). The alpha arms sample INSIDE the resident logit window
(KDChunks, resident=2), which advances one chunk every --chunk-steps steps. If the step budget is too
small for the window to sweep all the chunks, the alpha trend is measured on a PREFIX of the slice, not
a representative sample -- and the trend would then confound alpha with "which 60% of the corpus". Same
failure family as every plausible artefact this rung has produced: the run looks healthy, the loss curve
is clean, and it answers a question about a distorted subset.

This simulates the EXACT advance schedule from the trainer's own constants (no training, no GPU): pos
starts at 0, advances (pos+1) % n_chunks whenever gstep % chunk_steps == 0, and the resident set at each
step is {pos, pos+1}. It counts, per chunk, how many steps it spends resident, and fails if any chunk is
never resident (a hole in the slice) or the coverage is too uneven to call the trend representative.

Run: python benchmarks/phase64/data/ws3_chunk_residence.py --steps 15106 --chunk-steps 200
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGITS = os.path.join(ROOT, "results", "phase64", "rung1", "logits_s0", "manifest.json")


def residence(steps, chunk_steps, n_chunks, resident=2):
    """Per-chunk resident-step counts, replaying the trainer's advance rule exactly."""
    counts = np.zeros(n_chunks, dtype=np.int64)
    pos = 0
    for gstep in range(steps):
        if gstep and gstep % chunk_steps == 0:
            pos = (pos + 1) % n_chunks
        for d in range(min(resident, n_chunks)):
            counts[(pos + d) % n_chunks] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15106, help="per-arm stage-C budget (prereg)")
    ap.add_argument("--chunk-steps", type=int, default=0,
                    help="0 = DERIVE exactly as the trainer does (steps // n_chunks // sweeps); a positive "
                         "value checks an explicit override.")
    ap.add_argument("--sweeps", type=int, default=3, help="the trainer's --chunk-sweeps K (default 3)")
    ap.add_argument("--resident", type=int, default=2, help="KDChunks resident default")
    ap.add_argument("--max-cov", type=float, default=0.15, help="max coefficient of variation to pass")
    a = ap.parse_args()

    n = len(json.load(open(LOGITS))["chunks"]) if os.path.isfile(LOGITS) else 121
    # Derive with the SAME formula the trainer uses, so the gate checks the value that will actually run.
    if a.chunk_steps <= 0:
        a.chunk_steps = max(1, a.steps // (n * a.sweeps))
        print(f"  chunk-steps DERIVED = {a.steps} // ({n} x {a.sweeps}) = {a.chunk_steps}\n")
    adv = a.steps // a.chunk_steps
    sweeps = adv / n
    c = residence(a.steps, a.chunk_steps, n, a.resident)

    resident_chunks = int((c > 0).sum())
    starved = int((c == 0).sum())
    cov = float(c.std() / max(c.mean(), 1e-9))
    print(f"chunk-residence gate   {n} chunks, resident={a.resident}   "
          f"{a.steps} steps / {a.chunk_steps} chunk-steps = {adv} advances = {sweeps:.2f} sweeps\n")
    print(f"  chunks ever resident : {resident_chunks}/{n}"
          + ("" if starved == 0 else f"   <-- {starved} NEVER sampled"))
    print(f"  residence per chunk  : min {int(c.min())}  max {int(c.max())}  mean {c.mean():.0f} steps")
    print(f"  coefficient of var   : {cov:.3f}   (bar {a.max_cov})")
    # a coarse ASCII sparkline of residence across the chunk ring. Normalised from ZERO, not from the
    # minimum: a blank must mean "never resident", so a starved chunk cannot hide behind a low-but-nonzero
    # one. (Normalising from the min made every min-residence chunk render blank -- a PASS looked like a
    # hole.) blocks[0] is the space, reached only by a true zero.
    blocks = " .:-=+*#@"
    hi = max(int(c.max()), 1)
    spark = "".join(blocks[0] if v == 0 else blocks[1 + min(len(blocks) - 2, int(v / hi * (len(blocks) - 2)))]
                    for v in c)
    print(f"  residence(chunk 0..{n-1}): {spark}")

    ok = starved == 0 and cov <= a.max_cov
    print(f"\n  verdict: {'PASS -- window sweeps the slice representatively' if ok else 'FAIL'}")
    if not ok:
        need = int(np.ceil(a.steps / (n * 2)))     # >=2 sweeps
        print(f"  the alpha trend would be measured on a PREFIX, not the slice. To sweep >=2x, "
              f"--chunk-steps <= {need}.")
        # show the fix passes, so the number is not just asserted
        c2 = residence(a.steps, need, n, a.resident)
        print(f"  at --chunk-steps {need}: {int((c2>0).sum())}/{n} chunks, "
              f"CoV {c2.std()/max(c2.mean(),1e-9):.3f}, {(a.steps//need)/n:.2f} sweeps")
    print("\nSTOP. No commit.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
