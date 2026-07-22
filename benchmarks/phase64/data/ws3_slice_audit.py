#!/usr/bin/env python3
"""WS3 -- does the scored logit slice touch the reserved validation band?

WHY IT MATTERS. All five screening arms train on this slice. If the band was not excluded at draw time,
~1.68% of the windows would be band content, and the in-corpus record-only validation set would have
been trained on by the whole block -- which destroys the one thing it exists to do (decompose temporal
shift from modelling capacity). The build claims to exclude it; a claim in a log line is not a check.

THE POSITIVE CONTROL, and why it needs no extra compute. Widen the band: documents with
h(blob_id) in [0.01, 0.02) were NOT excluded by the build, so counting them exercises the identical
doc -> byte -> window mapping on identical data and MUST return a non-zero count. A counter that has
never been seen to fire is an assumption; this makes it fire on demand.

Consistency is checked first, and it is the part that would silently invalidate everything else: the
document list is reconstructed by re-walking the shards under the same rule, and its cumulative byte
offsets must reproduce `docbound_<tag>.i64` EXACTLY. If they do not, the reconstruction is not the
corpus that was built and no count computed from it means anything.

Run: python benchmarks/phase64/data/ws3_slice_audit.py --tag s0
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase64", "mve"))
from fetch_corpus import is_val, h01, VAL_BAND_HI      # noqa: E402
from mve_logits import windows                         # noqa: E402

RAW = os.path.join(ROOT, "data", "phase64", "raw_python")
DATA = os.path.join(ROOT, "results", "phase64", "rung1")


def rewalk(target_bytes):
    """Reproduce the document list build_raw() produced: same shard order, same rule, same stop."""
    ids, sizes, nb = [], [], 0
    for sh in sorted(glob.glob(os.path.join(RAW, "shard*.jsonl"))):
        for line in open(sh, encoding="utf-8"):
            d = json.loads(line)
            if is_val(d["blob_id"]): continue
            b = d["text"].encode("utf-8")
            if not b.endswith(b"\n"): b += b"\n"
            ids.append(d["blob_id"]); sizes.append(len(b)); nb += len(b)
            if nb >= target_bytes: break
        if nb >= target_bytes: break
    return ids, np.array(sizes, dtype=np.int64)


def touched(win_starts, L, tstart, docbound, flag):
    """How many scoring windows touch at least one document whose flag is set."""
    cum = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    n = len(tstart)
    b0 = tstart[np.asarray(win_starts)]
    b1 = tstart[np.minimum(np.asarray(win_starts) + L, n) - 1]
    d0 = np.clip(np.searchsorted(docbound, b0, "right") - 1, 0, len(flag) - 1)
    d1 = np.clip(np.searchsorted(docbound, b1, "right") - 1, 0, len(flag) - 1)
    return int(((cum[d1 + 1] - cum[d0]) > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(DATA, f"meta_{a.tag}.json")))
    docbound = np.fromfile(os.path.join(DATA, f"docbound_{a.tag}.i64"), dtype=np.int64)
    tstart = np.fromfile(os.path.join(DATA, f"teacher_start_{a.tag}.i64"), dtype=np.int64)
    n_tea = int(meta["n_teacher_tok"]); N = int(meta["bytes"])
    print(f"WS3 slice audit   tag={a.tag}   {N/2**20:.1f} MiB, {len(docbound)-1} docs, {n_tea} teacher tok")

    ids, sizes = rewalk(N)
    bd = np.concatenate([[0], np.cumsum(sizes)])
    ok_docs = len(ids) == len(docbound) - 1
    ok_bytes = ok_docs and np.array_equal(bd, docbound)
    print(f"  reconstruction: {len(ids)} docs vs {len(docbound)-1} recorded -> {'match' if ok_docs else 'MISMATCH'}; "
          f"byte offsets {'IDENTICAL' if ok_bytes else 'DIFFER'}")
    if not ok_bytes:
        sys.exit("ERROR: the re-walk does not reproduce the built corpus. Any count from it would be "
                 "about a different document list. Refusing to report a number.")

    h = np.array([h01(b) for b in ids])
    st, L = windows(n_tea, a.ctx, a.stride)
    print(f"  windows: {len(st)} of length {L}\n")

    band = h < VAL_BAND_HI                                   # what the build claims to have excluded
    ctrl = (h >= VAL_BAND_HI) & (h < 2 * VAL_BAND_HI)        # planted positive: present by construction
    nb_ = touched(st, L, tstart, docbound, band)
    nc_ = touched(st, L, tstart, docbound, ctrl)
    print(f"  {'stratum':34s} {'docs':>8s} {'windows touching':>17s}")
    print(f"  {'reserved band  h < ' + str(VAL_BAND_HI):34s} {int(band.sum()):8d} {nb_:17d}")
    print(f"  {'control band   ' + str(VAL_BAND_HI) + ' <= h < ' + str(2*VAL_BAND_HI):34s} "
          f"{int(ctrl.sum()):8d} {nc_:17d}")

    fired = nc_ > 0
    clean = nb_ == 0
    print(f"\n  positive control fired: {'YES' if fired else 'NO -- the counter is blind, its zero means nothing'}"
          f"  ({100*nc_/max(len(st),1):.2f}% of windows)")
    print(f"  reserved band in the slice: {'NONE' if clean else str(nb_) + ' windows -- FILTER CONSUME-SIDE'}")
    print("\n==== slice audit: " + ("PASS" if (fired and clean) else "FAIL") + " ====")
    print("\nSTOP. No commit.")
    sys.exit(0 if (fired and clean) else 1)


if __name__ == "__main__":
    main()
