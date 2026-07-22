#!/usr/bin/env python3
"""WS3 stages 1-4 -- everything downstream of the fetch, and none of it blocked on credentials.

    exact-hash  ->  MinHash dedup (5-shingle, 128 perm, J=0.7)  ->  P62 DECONTAMINATION (J=0.5,
    MANDATORY)  ->  per-domain BPE at BOTH V=2048 and V=4096  ->  manifest + hash

WHY DECONTAMINATION IS MANDATORY AND NOT A PRECAUTION: the P62 code-val is CPython source, and CPython
is *in* The Stack. Contamination is not a risk here, it is a certainty absent this stage -- and every
rung-1 BPB gate reads that val set. A leak would not produce an obviously wrong number; it would produce
a good one. P62 itself is NEVER modified: it is the fixed reference, and the training corpus is what
moves.

DECLARED CHOICE, because it changes the strictness: P62's code_val is a concatenation without reliable
per-file delimiters (2 path headers in 1.5 MB), so it is shingled in fixed 4 KiB windows rather than per
file. Finer windows make decontamination STRICTER (more chances to trip J=0.5), which is the safe
direction for a mandatory filter.

Everything here runs today on any local corpus -- see ws3_smoke.py. The point is that when the owner's
credentials land, the only unproven stage is the fetch.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))

NON_ALPHA = re.compile(r"\W+")
SHINGLE = 5              # BigCode default
NPERM = 128              # BigCode default
J_DEDUP = 0.7            # BigCode default
J_DECON = 0.5            # plan §90: MANDATORY
MERSENNE = (1 << 61) - 1
P62_VAL = os.path.join(ROOT, "results", "phase62", "code_val.txt")
VOCABS = (2048, 4096)    # the vocab A/B needs both


# ------------------------------------------------------------------ MinHash --------------------------
def shingles(text, k=SHINGLE):
    w = [t for t in NON_ALPHA.split(text) if t]
    if len(w) < k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def _perms(rng):
    a = rng.integers(1, MERSENNE, size=NPERM, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=NPERM, dtype=np.uint64)
    return a, b


def signature(text, a, b):
    sh = shingles(text)
    if not sh:
        return np.full(NPERM, np.uint64(MERSENNE), dtype=np.uint64)
    h = np.array([int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "little") & 0xFFFFFFFF
                  for s in sh], dtype=np.uint64)
    # (a*h + b) mod (2^61-1), vectorised over the 128 permutations
    v = (np.multiply.outer(h, a) + b) % np.uint64(MERSENNE)
    return v.min(axis=0)


def bands(nperm=NPERM, j=J_DEDUP):
    """Pick (b, r) with b*r = nperm whose S-curve threshold (1/b)^(1/r) is closest to the target J.

    Chosen rather than hard-coded because the threshold is the parameter the plan states; b and r are an
    implementation detail that must SERVE it. Hard-coding b=16,r=8 would silently target J=0.71."""
    best = None
    for r in range(1, nperm + 1):
        if nperm % r: continue
        bb = nperm // r
        t = (1.0 / bb) ** (1.0 / r)
        if best is None or abs(t - j) < abs(best[2] - j):
            best = (bb, r, t)
    return best


def lsh_buckets(sigs, b, r):
    """band -> {hash: [doc indices]}. Candidate pairs are those colliding in any band."""
    out = []
    for bi in range(b):
        d = {}
        for i, s in enumerate(sigs):
            key = hashlib.sha1(s[bi * r:(bi + 1) * r].tobytes()).digest()[:12]
            d.setdefault(key, []).append(i)
        out.append(d)
    return out


def jacc(x, y):
    return float((x == y).mean())


# ------------------------------------------------------------------ stages ---------------------------
def stage_exact(docs):
    """Exact-content dedup by sha256. Cheap, and it removes the mass that would otherwise dominate the
    MinHash candidate pairs."""
    seen, keep = set(), []
    for d in docs:
        h = d.get("sha256") or hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); keep.append(d)
    return keep, len(docs) - len(keep)


def stage_minhash(docs, seed, j=J_DEDUP):
    """Near-duplicate removal. Candidates from LSH, then the signature Jaccard is VERIFIED on each pair
    -- LSH banding produces false positives by design and dropping them unverified would delete real
    training data on a hash collision."""
    rng = np.random.default_rng(seed)
    a, b_ = _perms(rng)
    sigs = [signature(d["text"], a, b_) for d in docs]
    nb, nr, thr = bands(j=j)
    dead = set()
    for band in lsh_buckets(sigs, nb, nr):
        for idxs in band.values():
            if len(idxs) < 2: continue
            for i in idxs[1:]:
                if i in dead: continue
                if jacc(sigs[idxs[0]], sigs[i]) >= j: dead.add(i)
    keep = [d for i, d in enumerate(docs) if i not in dead]
    return keep, len(dead), dict(bands=nb, rows=nr, curve_threshold=thr)


def val_windows(path=P62_VAL, win=4096):
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    return [raw[i:i + win] for i in range(0, len(raw), win)]


def stage_decontam(docs, seed, j=J_DECON, val_path=P62_VAL, win=4096):
    """Remove any training doc containing a P62 val window. MANDATORY. P62 is never modified.

    WINDOW-vs-WINDOW, not doc-vs-window, and this is the whole correctness of the stage. Jaccard is
    symmetric in set SIZE: a 24 KiB file that contains a 4 KiB val window verbatim scores
    J ~ 4/24 ~ 0.17, far under any sane threshold -- so a doc-level comparison waves through exactly the
    leak it exists to catch. (The smoke's planted verbatim val slice survived the first version of this
    function. That is what the planted positive is for.)

    Cutting BOTH sides into equal windows makes Jaccard behave like containment. The training side uses
    50% overlap so a leak straddling a boundary still lands whole inside some window."""
    if not os.path.isfile(val_path):
        sys.exit(f"ERROR: P62 val not found at {val_path}. Decontamination is a MANDATORY stage and a "
                 f"corpus built without it must not be used for any gate. Refusing to continue.")
    rng = np.random.default_rng(seed)
    a, b_ = _perms(rng)
    vs = [signature(w, a, b_) for w in val_windows(val_path, win)]
    nb, nr, thr = bands(j=j)
    vbands = lsh_buckets(vs, nb, nr)
    dead, nwin = set(), 0
    for i, d in enumerate(docs):
        t = d["text"]
        for off in range(0, max(len(t) - win, 0) + 1, win // 2):
            s = signature(t[off:off + win], a, b_); nwin += 1
            hit = False
            for bi in range(nb):
                key = hashlib.sha1(s[bi * nr:(bi + 1) * nr].tobytes()).digest()[:12]
                for vi in vbands[bi].get(key, []):
                    if jacc(s, vs[vi]) >= j: hit = True; break
                if hit: break
            if hit: dead.add(i); break
    keep = [d for i, d in enumerate(docs) if i not in dead]
    ex = [dict(path=docs[i].get("path", ""), language=docs[i].get("language", ""),
               bytes=len(docs[i]["text"].encode("utf-8"))) for i in sorted(dead)[:20]]
    return keep, len(dead), dict(val_windows=len(vs), doc_windows=nwin, window=win, stride=win // 2,
                                 bands=nb, rows=nr, curve_threshold=thr, examples=ex)


def stage_tokenizers(docs, out_dir, vocabs=VOCABS, train_bytes=8 << 20):
    """One BPE per vocab size, learned on the DEDUPED+DECONTAMINATED text only.

    Order matters and is not cosmetic: a tokenizer learned before decontamination would carry merges
    fitted to val content into every downstream run, which is contamination that survives deleting the
    documents."""
    from cartography import train_bpe
    os.makedirs(out_dir, exist_ok=True)
    buf = "".join(d["text"] for d in docs).encode("utf-8")[:train_bytes]
    made = {}
    for V in vocabs:
        t0 = time.time()
        bpe = train_bpe(buf, target_vocab=V)
        p = os.path.join(out_dir, f"bpe{V}_code.bin")
        bpe.save(p)
        ids = bpe.encode(buf)
        made[V] = dict(path=p, sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                       train_bytes=len(buf), n_tok=len(ids),
                       bytes_per_tok=len(buf) / max(len(ids), 1), seconds=time.time() - t0)
    return made


def manifest(docs, stats, toks, out_dir, seed):
    """The corpus is never committed or redistributed: this manifest + the scripts + the hashes ARE the
    artefact. `corpus_sha256` is verified at every training start -- that check is what makes 'the run
    used the registered corpus' a fact rather than a belief."""
    h = hashlib.sha256()
    for d in docs:
        h.update((d.get("sha256") or hashlib.sha256(d["text"].encode("utf-8")).hexdigest()).encode())
    nb = sum(len(d["text"].encode("utf-8")) for d in docs)
    m = dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=seed,
             policy="corpora NEVER committed or redistributed; manifest+scripts+hash only; data/ gitignored",
             n_docs=len(docs), bytes=nb, corpus_sha256=h.hexdigest(),
             params=dict(shingle=SHINGLE, nperm=NPERM, j_dedup=J_DEDUP, j_decontam=J_DECON,
                         decontam_ref=os.path.relpath(P62_VAL, ROOT), decontam_window=4096),
             stages=stats, tokenizers={str(k): v for k, v in toks.items()})
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "corpus_manifest.json")
    json.dump(m, open(p, "w"), indent=1)
    return p, m


def run(docs, out_dir, seed, decontam_stop=0.05):
    stats = {}
    n0 = len(docs)
    print(f"  in: {n0} docs, {sum(len(d['text']) for d in docs)/2**20:.1f} MiB")

    docs, n = stage_exact(docs)
    stats["exact"] = dict(removed=n, kept=len(docs))
    print(f"  exact-hash   : -{n:6d}  -> {len(docs)}")

    docs, n, info = stage_minhash(docs, seed)
    stats["minhash"] = dict(removed=n, kept=len(docs), **info)
    print(f"  minhash J>={J_DEDUP}: -{n:6d}  -> {len(docs)}   "
          f"(b={info['bands']} r={info['rows']}, curve thr {info['curve_threshold']:.3f})")

    pre = len(docs)
    docs, n, info = stage_decontam(docs, seed)
    frac = n / max(pre, 1)
    stats["decontam"] = dict(removed=n, kept=len(docs), fraction=frac, **info)
    print(f"  P62 decontam J>={J_DECON}: -{n:6d}  -> {len(docs)}   "
          f"({info['val_windows']} val windows vs {info['doc_windows']} doc windows)   "
          f"= {100*frac:.2f}% of docs   [MANDATORY]")

    # The asymmetry stays deliberate -- removing too much is cheap, missing a leak invalidates every gate
    # -- but a filter that is firing wide has to be SEEN before it is trained on, not explained after.
    # Zero collateral on a 68-document smoke says nothing about the false-positive rate at 40 GB.
    if frac > decontam_stop:
        print(f"\n  first {len(info['examples'])} removed:")
        for e in info["examples"]:
            print(f"    {e['bytes']:9d} B  {e['language']:12s} {e['path']}")
        sys.exit(f"\nSTOP: decontamination removed {100*frac:.2f}% of documents, over the {100*decontam_stop:.0f}% "
                 f"review threshold. Examples above. This is not a failure and the threshold is NOT to be "
                 f"raised to get past it -- it is the pre-registered point at which the Architect looks at "
                 f"what is being cut before any of it trains.")

    if not docs:
        sys.exit("ERROR: every document was removed. A corpus of zero documents is not a result; check "
                 "the input before touching the thresholds.")

    toks = stage_tokenizers(docs, out_dir)
    for V, t in toks.items():
        print(f"  BPE V={V:5d}   : {t['bytes_per_tok']:.3f} B/tok on {t['train_bytes']/2**20:.1f} MiB "
              f"({t['seconds']:.1f}s)  sha {t['sha256'][:16]}...")
    print(f"  V4096 vs V2048 sequence length: "
          f"{100*(toks[4096]['n_tok']/toks[2048]['n_tok']-1):+.1f}%  (plan T2 expects about -17%)")

    p, m = manifest(docs, stats, toks, out_dir, seed)
    print(f"  manifest     : {p}\n                 corpus_sha256 {m['corpus_sha256'][:32]}...")
    return docs, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", default=[], help="shard(s) from the fetch")
    ap.add_argument("--dir", default="", help="directory of shard*.jsonl (avoids a 70-arg command line)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "phase64", "corpus"))
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--decontam-stop", type=float, default=0.05,
                    help="halt and show examples if decontamination removes more than this fraction")
    a = ap.parse_args()
    import glob as _glob
    paths = list(a.jsonl) + (sorted(_glob.glob(os.path.join(a.dir, "shard*.jsonl"))) if a.dir else [])
    if not paths:
        sys.exit("no shards given (--jsonl or --dir). For a credential-free exercise, run ws3_smoke.py.")
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            docs += [json.loads(l) for l in f]
    print(f"  loaded {len(docs)} docs from {len(paths)} shard(s)")
    print(f"WS3 pipeline   seed={a.seed}")
    run(docs, a.out, a.seed, a.decontam_stop)
    print("\nSTOP. WS3 pipeline above. No commit.")


if __name__ == "__main__":
    main()
