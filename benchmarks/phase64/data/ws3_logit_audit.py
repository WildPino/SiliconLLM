#!/usr/bin/env python3
"""WS3 -- audit the finished teacher-logit artifact ON DISK, before five arms are built on it.

WHY THIS AND NOT THE RUN'S OWN SELF-CHECK. mve_logits.py printed hit@32 = 96.9% from tensors it had in
hand, inside the loop that produced them. That number can be perfect while the artifact on disk is
useless: a wrong j0, a chunk written twice, a gap at a resume boundary, or a truncated final write are
all invisible to an in-flight check and all produce a KD signal that is silently misaligned with the
student's targets. Misaligned KD does not crash -- it trains, converges, and answers a different
question, which is this project's characteristic failure.

So the artifact is re-read from disk and scored against teacher_ids_s0.i32, which the scoring loop never
consulted for its own check.

THE PLANTED POSITIVE. A hit-rate computed from disk is only evidence if it can fall. `--shift N` offsets
the row index by N before looking up the true token: the identical code path, the identical bytes, one
deliberate misalignment. If shift=1 does not collapse the rate, the check is not reading alignment at
all and its pass means nothing.

Run: python benchmarks/phase64/data/ws3_logit_audit.py --tag s0 --sample 200000
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "results", "phase64", "rung1")


def check_layout(man, ldir, n_tea):
    """Contiguity, coverage, and byte-length of every chunk. Any failure here voids the sample below."""
    ch = sorted(man["chunks"], key=lambda c: c["j0"])
    problems, expect = [], 1
    for c in ch:
        if c["j0"] != expect:
            problems.append(f"{c['path']}: starts at row {c['j0']}, expected {expect} "
                            f"({'GAP' if c['j0'] > expect else 'OVERLAP'} of {abs(c['j0']-expect)} rows)")
        p = os.path.join(ldir, c["path"])
        if not os.path.isfile(p):
            problems.append(f"{c['path']}: MISSING")
        else:
            got = os.path.getsize(p)
            if got != c["bytes"]:
                problems.append(f"{c['path']}: {got} bytes on disk, manifest says {c['bytes']} (truncated write?)")
        expect = c["j0"] + c["rows"]
    end = expect
    # Row 0 is never scored: the first teacher token has no preceding position to predict it from.
    covered = end - 1
    print(f"  chunks {len(ch)}   rows [1, {end})   covered {covered} of {n_tea - 1} scoreable "
          f"({100*covered/max(n_tea-1,1):.4f}%)")
    if covered != n_tea - 1:
        problems.append(f"coverage: {covered} rows stored, {n_tea-1} teacher tokens are scoreable")
    return ch, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--sample", type=int, default=200_000, help="rows to re-score, spread over all chunks")
    ap.add_argument("--shift", type=int, default=0, help="deliberate row misalignment; the planted positive")
    a = ap.parse_args()

    ldir = os.path.join(DATA, f"logits_{a.tag}")
    man = json.load(open(os.path.join(ldir, "manifest.json")))
    meta = json.load(open(os.path.join(DATA, f"meta_{a.tag}.json")))
    n_tea = int(meta["n_teacher_tok"])
    tids = np.fromfile(os.path.join(DATA, f"teacher_ids_{a.tag}.i32"), dtype=np.int32)
    print(f"WS3 logit audit   tag={a.tag}   teacher={man['teacher']} {man['quant']}   K={man['K']}")
    print(f"  teacher_ids {len(tids)} tok   meta says {n_tea}   "
          f"{'match' if len(tids) == n_tea else 'MISMATCH -- the ids file is not this slice'}")
    if len(tids) != n_tea:
        sys.exit("ERROR: the token stream and the scored slice disagree. Refusing to audit.")

    ch, problems = check_layout(man, ldir, n_tea)
    if problems:
        print("\n  LAYOUT PROBLEMS:")
        for p in problems: print("   ", p)
        sys.exit("\n==== logit audit: FAIL (layout) ====")
    print("  layout: contiguous, complete, every chunk full-length")

    per = max(1, a.sample // len(ch))
    rng = np.random.default_rng(20260719)
    hit, tot, nan_rows, bad_id, p1 = 0, 0, 0, 0, []
    V_tea = 151936      # Qwen2.5 embedding rows; ids must land inside it
    for c in ch:
        z = np.load(os.path.join(ldir, c["path"]))
        ids, pq, j0 = z["ids"], z["pq"], int(z["j0"])
        if j0 != c["j0"]:
            sys.exit(f"ERROR: {c['path']} stores j0={j0}, manifest says {c['j0']}. "
                     f"The manifest is not describing this file.")
        if len(ids) != c["rows"]:
            sys.exit(f"ERROR: {c['path']} holds {len(ids)} rows, manifest says {c['rows']}.")
        k = min(per, len(ids))
        sel = rng.choice(len(ids), size=k, replace=False)
        sub_ids, sub_pq = ids[sel], pq[sel]
        true = tids[j0 + sel + a.shift]
        hit += int((sub_ids == true[:, None]).any(1).sum()); tot += k
        bad_id += int(((sub_ids < 0) | (sub_ids >= V_tea)).sum())
        nan_rows += int((sub_pq.max(1) == 0).sum())          # an all-zero row = no mass anywhere = dead write
        p1.append(sub_pq[:, 0].astype(np.float64).mean() / 255.0)

    rate = 100.0 * hit / max(tot, 1)
    print(f"\n  re-scored {tot} rows from disk against teacher_ids (shift={a.shift})")
    print(f"    true token in stored top-32   {rate:.2f}%")
    print(f"    mean p(top-1)                 {np.mean(p1):.3f}")
    print(f"    ids outside [0,{V_tea})       {bad_id}")
    print(f"    all-zero probability rows     {nan_rows}")

    if a.shift == 0:
        ok = rate >= 90.0 and bad_id == 0 and nan_rows == 0
        print(f"\n  the run's own in-flight self-check said 96.9%; from disk it is {rate:.2f}%")
        print("\n==== logit audit: " + ("PASS" if ok else "FAIL") + " ====")
        print("  positive control not yet run. Re-run with --shift 1: the rate MUST collapse,")
        print("  otherwise this check does not read alignment and its PASS is worth nothing.")
        sys.exit(0 if ok else 1)
    else:
        fired = rate < 60.0
        print(f"\n  planted misalignment of {a.shift} row(s): {'COLLAPSED as required' if fired else 'DID NOT COLLAPSE'}")
        print("\n==== positive control: " + ("FIRED" if fired else "BLIND") + " ====")
        sys.exit(0 if fired else 1)


if __name__ == "__main__":
    main()
