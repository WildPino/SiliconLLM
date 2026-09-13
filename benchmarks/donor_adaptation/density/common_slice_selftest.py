# -*- coding: utf-8 -*-
"""Planted controls for the tokenizer guard in `common.get_slice`.

The guard's job is to make a hazard that E15 and E16 caught BY HAND -- `get_slice` keys its
disk cache on (part, n_seq, seq_len, seed) and not on the tokenizer -- impossible to walk into
silently.  A guard that never refuses anything is indistinguishable from no guard, so this file
shows it refusing, and shows it NOT refusing the crossing that is genuinely safe.

Cheap: tokenizers and the corpus only.  No model is loaded.  Run it from anywhere:

    python density/common_slice_selftest.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                              # noqa: E402
from transformers import AutoTokenizer          # noqa: E402

SLICE = ("heldout", 2, 512, 1234)               # the smallest pinned slice on disk
rows = []


def check(name, ok, note):
    rows.append((name, ok, note))
    print("  %-28s %-6s %s" % (name, "FIRES" if ok else "FAILS", note))


def main():
    base = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.REVISION)
    fp_base = C.tok_fingerprint(base)

    # --- T1: the fingerprint is stable, and it is a function of the tokenizer -------------
    check("T1 stable", C.tok_fingerprint(base) == fp_base, "same tokenizer -> same fingerprint")

    mut = AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.REVISION)
    mut.add_tokens(["<<<ctrl_slice_guard>>>"])
    check("T2 separates", C.tok_fingerprint(mut) != fp_base,
          "one added token moves the fingerprint")

    # --- T3: the Coder crossing is SAFE, and now we know why ------------------------------
    coder = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")
    identical = (base.get_vocab() == coder.get_vocab()
                 and base.get_added_vocab() == coder.get_added_vocab())
    check("T3 coder identical", identical and C.tok_fingerprint(coder) == fp_base,
          "Qwen2.5-1.5B and Qwen2.5-Coder-7B are the SAME tokenizer, vocab and added tokens")

    # --- T4: the guard serves the pinned slice to the tokenizer that built it -------------
    ids, _, meta = C.get_slice(base, *SLICE)
    check("T4 serves base", fp_base in (meta.get("tok_fp_ok") or []),
          "ids %s, verified fingerprints %s" % (meta["ids_sha256"][:12],
                                                meta.get("tok_fp_ok")))

    # --- T5: and to Coder too, because re-derivation AGREES, not because it was waved past -
    ids_c, _, meta_c = C.get_slice(coder, *SLICE)
    check("T5 crossing allowed", bool((ids == ids_c).all()),
          "the safe crossing is not blocked")

    # --- T6: THE CONTROL THAT MATTERS -- a genuinely different tokenizer is REFUSED --------
    try:
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:                       # offline and not cached
        check("T6 refuses gpt2", False, "SKIPPED, gpt2 not available (%s)" % type(e).__name__)
    else:
        fired = False
        msg = ""
        try:
            C.get_slice(gpt2, *SLICE)
        except RuntimeError as e:
            fired, msg = True, str(e).split(".")[0]
        check("T6 refuses gpt2", fired,
              msg or "THE GUARD HANDED A GPT-2 CALLER THE QWEN SLICE")

    bad = [r for r in rows if not r[1]]
    print("\n  slice-guard self-test : %s  (%d checks)"
          % ("ALL FIRE" if not bad else "%d FAILED" % len(bad), len(rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
