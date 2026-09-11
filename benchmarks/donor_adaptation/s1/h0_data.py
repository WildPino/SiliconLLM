#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H0 step 2 (CPU, free) -- pre-tokenize the training stream the T4 will read.

Plan: docs/research/donor_adaptation/decisions/T4_HEALING_PROPOSAL.md s3 (H0).

The T4 gets tokens, not text: the tokenizer is a CPU cost with no reason to be paid on a GPU
quota, and shipping ids makes the training stream a pinned, hashable artefact rather than
something re-derived differently on the other side.

PROVENANCE, WHICH IS THE WHOLE POINT OF THIS FILE BEING SEPARATE.

  * drawn from `calib.txt` ONLY.  `heldout.txt` is the frozen eval half (24x512, ids sha
    a1a48dc9...) and nothing trained here may ever see it.  The two halves are disjoint by
    construction in the corpus manifest, so this is structural, not a promise.
  * seed 90011, deliberately NOT 42424 -- that is the seed of the 32x512 calibration slice the
    factors were fitted on.  Overlap with it would not be fatal (same half, and the factors are
    a fixed initialisation rather than a held-out quantity) but it costs nothing to avoid and
    keeps "what fitted the factors" and "what trains them" separable.
  * offsets are drawn uniformly over a file that is itself a global shuffle of 8 KiB chunks, so
    neither file order nor block order can be a confound -- the same argument common.make_slice
    makes for the eval slice.

Unlike common.make_slice this does NOT do the decode/re-encode byte-accounting round trip.  That
check exists so BPB has an exact denominator; training tokens have no denominator and do not
need it.  Byte counts are therefore not emitted here, and no BPB may be computed from this file.

Env: H0_TOKENS (default 16.0e6), H0_SEQLEN (512), H0_SEED (90011), H0_SMOKE (1)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DENSDIR = os.path.abspath(os.path.join(HERE, "..", "density"))
sys.path.insert(0, DENSDIR)
import common as C                                          # noqa: E402

SMOKE = os.environ.get("H0_SMOKE", "0") == "1"
N_TOKENS = int(float(os.environ.get("H0_TOKENS", "2.0e5" if SMOKE else "16.0e6")))
SEQLEN = int(os.environ.get("H0_SEQLEN", "512"))
SEED = int(os.environ.get("H0_SEED", "90011"))
PART = "calib"                                  # never heldout; see the docstring
OUTDIR = os.path.join(HERE, "results", "h0")
OUT = os.path.join(OUTDIR, "h0_train%s.npz" % ("_smoke" if SMOKE else ""))
META = os.path.join(OUTDIR, "h0_train%s.json" % ("_smoke" if SMOKE else ""))

# the frozen eval slice this must never touch
EVAL_SHA = "a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"


def log(m):
    print(m, flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    n_seq = int(np.ceil(N_TOKENS / SEQLEN))
    log("== H0 data: %d sequences x %d tokens from %s.txt, seed %d ==" % (n_seq, SEQLEN, PART, SEED))

    model, tok = C.load_model()
    del model                                    # tokenizer only; do not hold 6 GB for nothing

    path = os.path.join(C.CORPUS, "%s.txt" % PART)
    size = os.path.getsize(path)
    manifest = json.load(open(os.path.join(C.CORPUS, "manifest.json")))
    rng = np.random.default_rng(SEED)
    span = SEQLEN * 8                            # ~4 bytes/token here; 8x is always enough

    seqs, tries, rejected = [], 0, 0
    with open(path, "rb") as f:
        while len(seqs) < n_seq and tries < n_seq * 40:
            tries += 1
            off = int(rng.integers(0, size - span))
            f.seek(off)
            txt = f.read(span).decode("utf-8", "ignore")
            enc = tok(txt, add_special_tokens=False)["input_ids"]
            if len(enc) < SEQLEN:
                rejected += 1
                continue
            seqs.append(enc[:SEQLEN])
            if len(seqs) % 2000 == 0:
                log("   %d/%d  (%.0fs)" % (len(seqs), n_seq, time.time() - t0))

    ids = np.asarray(seqs, dtype=np.int32)
    sha = hashlib.sha256(ids.tobytes()).hexdigest()
    if sha == EVAL_SHA:
        log("  ABORT: this stream hashes to the frozen EVAL slice.")
        return 2

    np.savez(OUT, ids=ids)
    meta = {
        "plan": "decisions/T4_HEALING_PROPOSAL.md s3 (H0)",
        "what": "pre-tokenized H0 training stream; ids only, NO byte accounting, no BPB from this",
        "model": C.MODEL_ID, "revision": C.REVISION, "smoke": SMOKE,
        "part": PART, "seed": SEED, "seq_len": SEQLEN,
        "n_seq": int(ids.shape[0]), "n_tokens": int(ids.size),
        "n_rejected": rejected, "tries": tries,
        "ids_sha256": sha,
        "corpus_sha256": manifest["parts"][PART]["sha256"],
        "eval_slice_sha256_NOT_THIS": EVAL_SHA,
        "calib_seed_deliberately_avoided": 42424,
        "seconds": time.time() - t0,
    }
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log("")
    log("  %d seq x %d = %d tokens, %d rejected" % (ids.shape[0], SEQLEN, ids.size, rejected))
    log("  ids_sha256 %s" % sha)
    log("  disjoint from the eval half by corpus construction (%s.txt vs heldout.txt)" % PART)
    log("  %s  (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
    log("  total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
