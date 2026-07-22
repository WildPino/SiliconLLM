#!/usr/bin/env python3
"""WS3 -- retrain the student BPEs on the TRAINING SPLIT ONLY, after the reserved band was carved out.

WHY. The V2048/V4096 tokenizers were trained on 8 MiB sampled from the corpus BEFORE the reserved band
existed, so roughly 1.7% of that sample could be band content. Merges learned from held-out text survive
the removal of the documents that taught them -- which is exactly the argument used for training the
tokenizers AFTER decontamination, applied to the band.

Proportionality, stated so the fix is not mistaken for a scare: this does NOT touch the canonical P62
gate (temporally outside the corpus, unreachable) and does NOT distort the vocab A/B (both arms are
treated identically). It affects only the record-only decomposition, and slightly. It is done because it
is coherent and costs minutes.

Run: python benchmarks/phase64/data/ws3_retrain_bpe.py
"""
import glob, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
from fetch_corpus import is_val, VAL_BAND_HI     # noqa: E402
from cartography import train_bpe, Bpe           # noqa: E402

RAW = os.path.join(ROOT, "data", "phase64", "raw_python")
OUT = os.path.join(ROOT, "data", "phase64", "corpus")
VOCABS = (2048, 4096)
TRAIN_BYTES = 8 << 20


def main():
    shards = sorted(glob.glob(os.path.join(RAW, "shard*.jsonl")))
    buf, nb, nd, nskip = [], 0, 0, 0
    for sh in shards:
        for line in open(sh, encoding="utf-8"):
            d = json.loads(line)
            if is_val(d["blob_id"]):
                nskip += 1; continue          # the band never teaches a merge
            b = d["text"].encode("utf-8")
            buf.append(b); nb += len(b); nd += 1
            if nb >= TRAIN_BYTES: break
        if nb >= TRAIN_BYTES: break
    blob = b"".join(buf)[:TRAIN_BYTES]

    print(f"WS3 BPE retrain on the training split only")
    print(f"  sample {len(blob)/2**20:.1f} MiB from {nd} train docs; {nskip} band docs skipped "
          f"(band = h < {VAL_BAND_HI})\n")
    # LABEL CAREFULLY. B/tok computed on the training blob is an IN-SAMPLE fit: tokens x B/tok is exactly
    # the sample size, by construction. It is not the corpus packing and must never be quoted as such --
    # a tokenizer packs its own training bytes better than the corpus, so the in-sample figure flatters.
    # `corpus_bytes_per_tok` in the manifest, measured on disjoint bytes, is the packing number.
    print(f"  {'vocab':>6} {'B/tok(in-sample)':>17} {'tokens':>12} {'sec':>7}  sha256")
    rec = {}
    for V in VOCABS:
        p = os.path.join(OUT, f"bpe{V}_code.bin")
        old = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else ""
        t0 = time.time()
        bpe = train_bpe(blob, target_vocab=V)
        bpe.save(p)
        ids = bpe.encode(blob)
        new = hashlib.sha256(open(p, "rb").read()).hexdigest()
        rec[V] = dict(path=p, sha256=new, prev_sha256=old, changed=(old != new and bool(old)),
                      train_bytes=len(blob), n_tok=len(ids), bytes_per_tok=len(blob) / max(len(ids), 1),
                      seconds=time.time() - t0, trained_on="training split only (reserved band excluded)")
        print(f"  {V:6d} {rec[V]['bytes_per_tok']:17.3f} {len(ids):12d} {rec[V]['seconds']:7.1f}  {new[:16]}...")
        if old:
            print(f"         {'CHANGED from ' + old[:16] + '...' if old != new else 'identical to the previous tokenizer'}")

    mp = os.path.join(OUT, "corpus_manifest.json")
    if os.path.isfile(mp):
        man = json.load(open(mp))
        # MERGE, do not replace: corpus_bytes_per_tok is measured separately on disjoint bytes and a
        # wholesale overwrite here would silently drop the only honest packing figure in the manifest.
        prev = man.get("tokenizers", {})
        man["tokenizers"] = {str(k): {**prev.get(str(k), {}), **v} for k, v in rec.items()}
        for k in man["tokenizers"]:
            man["tokenizers"][k].pop("corpus_bytes_per_tok", None)   # stale: the tokenizer just changed
            man["tokenizers"][k].pop("corpus_sample_bytes", None)
        man["tokenizer_note"] = ("retrained on the training split after the reserved validation band was "
                                 "carved out; merges must not be learned from held-out text")
        man["val_band"] = [0.0, VAL_BAND_HI]
        json.dump(man, open(mp, "w"), indent=1)
        print(f"\n  manifest updated: {mp}")
    r = rec[4096]["bytes_per_tok"] / rec[2048]["bytes_per_tok"]
    print(f"\n  at equal TOKEN budget the V4096 arm sees {100*(r-1):+.1f}% more BYTES "
          f"-- the declared qualifier on the vocab screening claim")
    print("\nSTOP. No commit.")


if __name__ == "__main__":
    main()
