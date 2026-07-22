#!/usr/bin/env python3
"""WS3 -- how many bytes are STRICT-PERMISSIVE? Metadata only: no content is fetched.

THE DECISION THIS UNBLOCKS. the-stack-v2-dedup carries two license classes: `permissive` and
`no_license`. `no_license` means no license was DETECTED -- which is not permission, it is absence of
evidence. The plan allows both with a strict-permissive option, and the first shard measured the split
at 23/77 on Python. If strict-permissive alone clears the ~40 GB target we can simply take it; if it
does not, the trade goes to the Architect with numbers instead of adjectives.

This must run BEFORE the fetch, not after: the answer decides which files get downloaded, and 15 h of
downloading the wrong set is the expensive way to learn it.

METHOD, with its weakness stated. Exact totals would mean streaming 47 M metadata rows per language.
Instead: sample N rows per language, measure the permissive fraction and the mean size of permissive
files, and extrapolate over the config's true `num_examples` (which the dataset info gives exactly).
The sample is drawn through a shuffle buffer because The Stack is ordered BY REPOSITORY -- reading the
head of the stream would sample a handful of repos and their licenses, not the corpus. The shuffle
decorrelates shard and row order; it does not make the draw uniform, so the interval below is a
sampling interval, NOT a guarantee. Treat the result as an estimate good to its order, which is all the
decision needs.

Run: python benchmarks/phase64/data/ws3_license_census.py [--n 4000]
"""
import argparse, itertools, os, sys
import numpy as np

LANGS = ("Python", "C", "C++", "Rust", "Go", "JavaScript", "TypeScript", "Java", "Shell")
PERMISSIVE = "permissive"


def census(lang, n, seed, token):
    from datasets import load_dataset, load_dataset_builder
    info = load_dataset_builder("bigcode/the-stack-v2-dedup", lang, token=token).info
    total = info.splits["train"].num_examples
    ds = load_dataset("bigcode/the-stack-v2-dedup", lang, split="train", streaming=True, token=token)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    sizes_all, is_perm = [], []
    for r in itertools.islice(ds, n):
        sizes_all.append(int(r.get("length_bytes", 0) or 0))
        is_perm.append(1.0 if r.get("license_type") == PERMISSIVE else 0.0)
    k = len(sizes_all)
    if k == 0: return None
    # The size array and the licence mask must stay ROW-ALIGNED. Built separately (a count of permissive
    # rows expanded into a sorted mask) they pair every size with someone else's licence, which silently
    # estimates E[size]*P(perm) instead of E[size AND perm] -- and the two differ whenever permissive
    # files are not the average size, which is exactly the case. The tell was the point estimate landing
    # OUTSIDE its own bootstrap interval.
    a = np.array(sizes_all, dtype=np.float64)
    m = np.array(is_perm, dtype=np.float64)
    frac = float(m.mean())
    gb_all = total * float(a.mean()) / 2**30
    gb_perm = total * float((a * m).mean()) / 2**30
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(200):
        idx = rng.integers(0, k, k)
        boot.append(total * float((a[idx] * m[idx]).mean()) / 2**30)
    return dict(lang=lang, files=total, sample=k, frac=frac, gb_all=gb_all, gb_perm=gb_perm,
                lo=float(np.percentile(boot, 5)), hi=float(np.percentile(boot, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="metadata rows sampled per language")
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--languages", default=",".join(LANGS))
    ap.add_argument("--target-gb", type=float, default=40.0)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN absent from the environment. Metadata is gated; nothing to count without it.")

    print(f"WS3 license census   metadata only, no content fetched   n={a.n}/language  seed={a.seed}\n")
    print(f"  {'language':12s} {'files':>12s} {'perm%':>7s} {'GB all':>9s} {'GB strict-perm':>15s} "
          f"{'90% interval':>18s}")
    rows = []
    for lang in [x for x in a.languages.split(",") if x]:
        try:
            r = census(lang, a.n, a.seed, token)
        except Exception as e:
            print(f"  {lang:12s} FAILED: {type(e).__name__} {str(e)[:50]}"); continue
        if not r: continue
        rows.append(r)
        iv = "%.1f - %.1f" % (r["lo"], r["hi"])
        print(f"  {r['lang']:12s} {r['files']:12d} {100*r['frac']:6.1f}% {r['gb_all']:9.1f} "
              f"{r['gb_perm']:15.1f} {iv:>18s}", flush=True)

    if not rows: sys.exit("nothing counted")
    tot_all = sum(r["gb_all"] for r in rows)
    tot_p = sum(r["gb_perm"] for r in rows)
    lo = sum(r["lo"] for r in rows); hi = sum(r["hi"] for r in rows)
    print(f"\n  {'TOTAL':12s} {sum(r['files'] for r in rows):12d} "
          f"{100*tot_p/max(tot_all,1e-9):6.1f}% {tot_all:9.1f} {tot_p:15.1f} {f'{lo:.1f} - {hi:.1f}':>18s}")
    print(f"\n  target = {a.target_gb:.0f} GB")
    if lo >= a.target_gb:
        print(f"  STRICT-PERMISSIVE CLEARS THE TARGET even at the low end of the interval "
              f"({lo:.1f} GB >= {a.target_gb:.0f}). Take permissive only; no trade needed.")
    elif hi < a.target_gb:
        print(f"  STRICT-PERMISSIVE DOES NOT REACH THE TARGET even at the high end "
              f"({hi:.1f} GB < {a.target_gb:.0f}). The trade is real and belongs to the Architect.")
    else:
        print(f"  STRADDLES the target ({lo:.1f} - {hi:.1f} GB). The sample cannot decide it; either widen "
              f"the language set or raise --n before treating this as an answer.")
    print("\n  Note: `no_license` is absence of detected licence, not permission. That is the whole "
          "reason this count exists.")
    print("\nSTOP. WS3 licence census above. No commit.")


if __name__ == "__main__":
    main()
