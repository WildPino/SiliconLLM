# -*- coding: utf-8 -*-
"""B2 prerequisite: does the CACHED calib slice survive the Qwen2.5 -> Coder tokenizer crossing?

`common.get_slice` keys its disk cache on (part, n_seq, seq_len, seed) and NOT on the tokenizer.
`slice_calib_32x512_s42424.pt` was built on 2026-08-22 under a Qwen2.5 tokenizer, months before
Coder-7B entered the programme.  `qwen_export.py:132` calls exactly that key for R3's calibration
pass.  So a B2 export would silently calibrate on whatever ids that file holds.

E15 s4 verified the HELDOUT slice crosses cleanly.  That does not transfer: make_slice's REJECTION
loop is tokenizer-dependent (`len(enc) < seq_len`, and the decode/re-encode round-trip), so a
different tokenizer can reject a different candidate and shift every offset after it.  This
re-derives the calib slice under the Coder tokenizer and compares.

Cheap: tokenizers only, no model is loaded.
"""
import os
import sys

DENS = r"D:\_THINGS\Progetti\SiliconLLM\benchmarks\donor_adaptation\density"
sys.path.insert(0, DENS)
import common as CD  # noqa: E402
import torch         # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CACHE = os.path.join(CD.RESULTS, "slice_calib_32x512_s42424.pt")

d = torch.load(CACHE)
ids_cached, byts_cached, meta_cached = d["ids"], d["byts"], d["meta"]
print("cached  %s" % CACHE)
print("  ids_sha256      %s" % meta_cached["ids_sha256"][:16])
print("  offsets_sha256  %s" % meta_cached["offsets_sha256"][:16])
print("  n_seq %d  seq_len %d  scored_bytes %d  bytes/tok %.6f  rejected %d"
      % (meta_cached["n_seq"], meta_cached["seq_len"], meta_cached["total_scored_bytes"],
         meta_cached["bytes_per_token"], meta_cached["n_rejected"]))
print("  corpus_sha256   %s" % meta_cached["corpus_sha256"][:16])

for name in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-Coder-7B"):
    tk = AutoTokenizer.from_pretrained(name)
    ids, byts, meta = CD.make_slice(tk, "calib", 32, 512, 42424)
    same_ids = bool(ids.shape == ids_cached.shape and torch.equal(ids, ids_cached))
    same_byt = bool(byts.shape == byts_cached.shape and torch.equal(byts, byts_cached))
    print("\n%s  (len %d)" % (name, len(tk)))
    print("  ids_sha256      %s   ids identical to cache: %s"
          % (meta["ids_sha256"][:16], same_ids))
    print("  offsets_sha256  %s   offsets identical:      %s"
          % (meta["offsets_sha256"][:16],
             meta["offsets_sha256"] == meta_cached["offsets_sha256"]))
    print("  scored_bytes %d  bytes/tok %.6f  rejected %d   byte counts identical: %s"
          % (meta["total_scored_bytes"], meta["bytes_per_token"], meta["n_rejected"], same_byt))
    print("  max id %d" % int(ids.max()))
