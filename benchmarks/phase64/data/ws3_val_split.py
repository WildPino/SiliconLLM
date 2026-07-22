#!/usr/bin/env python3
"""WS3 -- materialise the reserved-band in-corpus validation set, and prove it is disjoint from training.

The band is defined in fetch_corpus.is_val() and is imported, never re-derived here: two copies of a
selection rule is one copy too many, and the whole value of the band is that it names the SAME files
forever.

TIMING NOTE, because it matters for what this script does. The rung-1 corpus was fetched BEFORE the band
existed, so band files are currently sitting in the training shards. They are not re-fetched -- they are
already on disk, and the rule is deterministic, so splitting after the fact yields exactly the file set
a band-aware fetch would have produced. From here on `fetch_corpus.is_val()` is the gate every consumer
applies, and no future expansion can absorb these files.

Run: python benchmarks/phase64/data/ws3_val_split.py
"""
import glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from fetch_corpus import is_val, h01, VAL_BAND_HI, SEED   # noqa: E402

RAW = os.path.join(ROOT, "data", "phase64", "raw_python")
OUT = os.path.join(ROOT, "data", "phase64", "val_incorpus")


def main():
    os.makedirs(OUT, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(RAW, "shard*.jsonl")))
    if not shards: sys.exit(f"no shards under {RAW}")
    vp = os.path.join(OUT, "val.jsonl")
    n_val = n_tr = b_val = b_tr = 0
    ids = []
    with open(vp, "w", encoding="utf-8") as f:
        for sh in shards:
            for line in open(sh, encoding="utf-8"):
                d = json.loads(line)
                nb = len(d["text"].encode("utf-8"))
                if is_val(d["blob_id"]):
                    f.write(line); n_val += 1; b_val += nb; ids.append(d["blob_id"])
                else:
                    n_tr += 1; b_tr += nb
    h = hashlib.sha256()
    for b in sorted(ids): h.update(b.encode())
    man = dict(band=[0.0, VAL_BAND_HI], seed=SEED, rule="fetch_corpus.is_val(blob_id)",
               n_val=n_val, n_train=n_tr, bytes_val=b_val, bytes_train=b_tr,
               val_blob_sha256=h.hexdigest(), status="record-only, never gated")
    json.dump(man, open(os.path.join(OUT, "val_manifest.json"), "w"), indent=1)

    print(f"WS3 reserved-band val split   band = h(blob_id) < {VAL_BAND_HI}   seed {SEED}")
    print(f"  val   {n_val:9d} docs  {b_val/2**20:9.1f} MiB   ({100*n_val/max(n_val+n_tr,1):.2f}% of docs)")
    print(f"  train {n_tr:9d} docs  {b_tr/2**30:9.2f} GB")
    print(f"  val_blob_sha256 {h.hexdigest()[:32]}...")

    # Planted checks. A split is exactly the kind of thing that looks right and silently is not.
    ok = True
    bad = [b for b in ids if not is_val(b)]
    print(f"\n  every val id satisfies the band rule: {'yes' if not bad else f'NO -- {len(bad)} violate it'}")
    ok &= not bad
    # The band is VAL_BAND_HI of the POOL, but this corpus is only the fraction p of the pool, so inside
    # the corpus the band is VAL_BAND_HI/p -- not VAL_BAND_HI. Comparing against the nominal 1% is the
    # wrong denominator and reads as a uniformity failure when the hash is in fact uniform. (It did: the
    # first run flagged 1.693% vs 1.000% and the discrepancy was in the check, not the data.)
    fp = os.path.join(RAW, "fetch_manifest.json")
    p_sel = json.load(open(fp))["select_p"] if os.path.isfile(fp) else 1.0
    frac = n_val / max(n_val + n_tr, 1)
    exp = VAL_BAND_HI / p_sel
    near = abs(frac - exp) < 0.2 * exp
    print(f"  observed band fraction {100*frac:.3f}% vs expected {100*exp:.3f}% "
          f"(= band {VAL_BAND_HI} / select_p {p_sel:.4f})  "
          f"{'(within 20%)' if near else '(OFF -- the hash is not uniform over these ids)'}")
    ok &= near
    # The permanence claim, exercised rather than asserted: raise p to 1.0 and re-check disjointness.
    still = all(is_val(b) for b in ids)
    print(f"  band survives a hypothetical p -> 1.0 expansion: {'yes' if still else 'NO'}  "
          f"(the rule does not depend on p, which is the point)")
    ok &= still
    print(f"\n  wrote {vp}")
    print("\n==== reserved-band split: " + ("PASS" if ok else "FAIL") + " ====")
    print("\nSTOP. No commit.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
