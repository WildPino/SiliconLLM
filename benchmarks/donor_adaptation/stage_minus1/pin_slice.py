#!/usr/bin/env python3
"""pin_slice.py -- turn "the eval slice is pinned" from a claim into a file.

B1 has been open through three Controller reviews. Review #3: "Sec 3.5b lists the required fields
and supplies not one value. No corpus is named. No sha256 exists. No stride, no context length, no
BPB_donor, no B." This script produces the values and WRITES THEM BACK into prereg.yaml, so that
check_prereg.py -- not prose -- decides whether the slice is pinned.

WHAT IT PINS
  * the exact bytes evaluated:  '\\n\\n'.join(wikitext-2-raw-v1 test['text']) at the pinned dataset
    revision, truncated to the last complete context window -- and their sha256
  * the byte range, token counts and window count under the method's own eval protocol
    (context 2048, stride 2048, tail dropped, no per-window BOS) read from pt2_llm/eval_ppl.py
  * B -- BYTES PER TOKEN, measured, never assumed and specifically never frozen at 4.0
  * BOS behaviour of the donor tokenizer, verified empirically rather than asserted
  * fast/slow tokenizer parity (the method builds its tokenizer with use_fast=False; this script
    needs the fast one for byte offsets, so the two must be shown to agree)
  * decontamination: byte-shingle overlap of the eval slice against (a) the calibration source
    (wikitext-2 TRAIN) and (b) the pinned P62 code-val
  * B for each publicly accessible ANCHOR model's tokenizer -- because converting an anchor's PPL
    ratio to b/byte needs THAT model's bytes/token, not the donor's

WHAT IT DOES NOT PIN
  BPB_donor and PPL_donor. Those require a forward pass, hence the CUDA install, which is not
  authorized. They stay UNPINNED with that reason, and check_prereg.py keeps reporting them.

USAGE
  python pin_slice.py                  # pin, print everything, write back into prereg.yaml
  python pin_slice.py --dry-run        # compute and print, write nothing
  python pin_slice.py --no-anchors     # skip the anchor-tokenizer pass (avoids a 500 MB download)

Network: reads huggingface.co only. Downloads the wikitext parquet files and tokenizer files.
No gated repository is touched and no authentication is attempted.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "prereg.yaml"
REPO_ROOT = HERE.parents[2]

# ---- pinned identities. These MUST agree with prereg.yaml; the script asserts it. --------------
CORPUS_REPO = "wikitext"
CORPUS_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TEST_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
TRAIN_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
DONOR_REPO = "Qwen/Qwen2.5-1.5B"
DONOR_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
CONTEXT_LENGTH = 2048
STRIDE = 2048
P62_CODEVAL = REPO_ROOT / "results" / "phase62" / "code_val.txt"

SHINGLE_BYTES = 64
SHINGLE_STRIDE = 16
SLOW_PARITY_CHARS = 200_000   # slow tokenizers are O(n) with a large constant; parity on a prefix

# Anchor models whose bytes/token we want. Gated repos are deliberately absent from this list.
ANCHOR_TOKENIZERS = [
    ("LLaMA-7B (control 1b anchor)", "huggyllama/llama-7b",
     "4782ad278652c7c71b72204d462d6d01eaaf7549"),
]


# ==================================================================================================
# YAML write-back -- targeted line replacement, so comments and ordering survive
# ==================================================================================================
def _indent_of(line):
    return len(line) - len(line.lstrip(" "))


def _yaml_scalar(value):
    """Render a Python scalar as a single-line YAML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return '"%s"' % text


KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_]+):(\s|$)(.*)$")
BLOCK_MARKERS = (">-", ">", "|", "|-", "|+", ">+")


def iter_key_lines(lines):
    """Yield (index, dotted_path, indent, value_part) for every mapping key line.

    Uses an explicit indent stack, and SKIPS the bodies of block scalars (>- and | blocks), whose
    prose frequently contains 'Word: text' and would otherwise be mistaken for a key.
    """
    stack = []                       # list of (indent, key)
    skip_below = None                # indent of an open block scalar; skip anything deeper
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        if skip_below is not None:
            if indent > skip_below:
                continue
            skip_below = None
        if stripped.startswith("- "):
            continue                 # sequence item: not on any path we rewrite
        match = KEY_RE.match(line)
        if not match:
            continue
        key, value_part = match.group(2), match.group(4).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        yield idx, ".".join(k for _, k in stack), indent, value_part
        if value_part in BLOCK_MARKERS:
            skip_below = indent


def set_scalar(lines, dotted, value):
    """Replace the single-line scalar at `dotted` in `lines` (list of str, no newlines).

    Returns the index of the replaced line. Raises if the path is missing, ambiguous, or not a
    single-line scalar -- silence here would reintroduce exactly the failure this exercise exists
    to prevent.
    """
    hits = [(idx, part) for idx, path, _, part in iter_key_lines(lines) if path == dotted]
    if not hits:
        raise KeyError("path not found in prereg.yaml: %s" % dotted)
    if len(hits) > 1:
        raise KeyError("path is AMBIGUOUS in prereg.yaml (%d matches): %s" % (len(hits), dotted))
    idx, value_part = hits[0]
    if value_part in BLOCK_MARKERS or value_part == "":
        raise ValueError("%s is a block scalar or a nested mapping, not a single-line scalar; "
                         "pin_slice.py will not rewrite it" % dotted)
    indent, key = _indent_of(lines[idx]), dotted.split(".")[-1]
    lines[idx] = "%s%s: %s" % (" " * indent, key, _yaml_scalar(value))
    return idx


# ==================================================================================================
# helpers
# ==================================================================================================
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def char_to_byte_prefix(text):
    """Return a list P where P[i] is the UTF-8 byte offset of character i (len == len(text)+1)."""
    import numpy as np
    widths = np.fromiter(
        (1 if ord(c) < 0x80 else (2 if ord(c) < 0x800 else (4 if ord(c) > 0xFFFF else 3))
         for c in text),
        dtype=np.int64, count=len(text))
    prefix = np.zeros(len(text) + 1, dtype=np.int64)
    np.cumsum(widths, out=prefix[1:])
    return prefix


def shingle_hashes(data, size=SHINGLE_BYTES, stride=SHINGLE_STRIDE):
    out = set()
    for i in range(0, max(0, len(data) - size + 1), stride):
        out.add(hashlib.blake2b(data[i:i + size], digest_size=8).digest())
    return out


def read_parquet_text(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["text"])
    return table.column("text").to_pylist()


# ==================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Pin the stage -1 eval slice into prereg.yaml.")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--no-anchors", action="store_true", help="skip anchor-tokenizer bytes/token")
    ap.add_argument("--prereg", default=str(PREREG))
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as exc:
        print("FATAL: %s  (need huggingface_hub + transformers)" % exc, file=sys.stderr)
        return 2

    prereg_path = Path(args.prereg)
    import yaml
    tree = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))

    # ---- assert this script and the artefact agree on WHAT is being pinned --------------------
    checks = [
        ("eval_slice.corpus_id", CORPUS_REPO),
        ("eval_slice.corpus_revision_sha", CORPUS_REVISION),
        ("eval_slice.corpus_file", TEST_FILE),
        ("eval_slice.tokenizer_repo_id", DONOR_REPO),
        ("eval_slice.tokenizer_revision_sha", DONOR_REVISION),
        ("eval_slice.context_length", CONTEXT_LENGTH),
        ("eval_slice.stride", STRIDE),
    ]
    for dotted, expected in checks:
        node = tree
        for part in dotted.split("."):
            node = node[part]
        if node != expected:
            print("FATAL: prereg.yaml %s = %r but this script pins %r. Refusing to run: the "
                  "artefact is the authority and the two must agree." % (dotted, node, expected),
                  file=sys.stderr)
            return 2
    print("[ok] prereg.yaml and pin_slice.py agree on corpus, revision, tokenizer, context, stride")

    # ---- 1. the corpus ------------------------------------------------------------------------
    print("\n[1/6] downloading %s @ %s" % (CORPUS_REPO, CORPUS_REVISION[:12]))
    test_path = hf_hub_download(CORPUS_REPO, TEST_FILE, repo_type="dataset",
                                revision=CORPUS_REVISION)
    train_path = hf_hub_download(CORPUS_REPO, TRAIN_FILE, repo_type="dataset",
                                 revision=CORPUS_REVISION)
    test_rows = read_parquet_text(test_path)
    train_rows = read_parquet_text(train_path)
    print("      test rows: %d   train rows: %d" % (len(test_rows), len(train_rows)))

    # The join rule is READ FROM pt2_llm/data.py@9e943e6, not chosen here.
    joined = "\n\n".join(test_rows)
    joined_bytes = joined.encode("utf-8")
    sha_joined = sha256_bytes(joined_bytes)
    print("      joined test text: %d chars, %d bytes" % (len(joined), len(joined_bytes)))
    print("      sha256(joined)  : %s" % sha_joined)

    # ---- 2. tokenize --------------------------------------------------------------------------
    print("\n[2/6] tokenizing with %s @ %s (fast, for byte offsets)"
          % (DONOR_REPO, DONOR_REVISION[:12]))
    tok_fast = AutoTokenizer.from_pretrained(DONOR_REPO, revision=DONOR_REVISION, use_fast=True)
    enc = tok_fast(joined, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n_tokens_total = len(ids)
    print("      n_tokens_total  : %d" % n_tokens_total)

    # BOS: verified, not asserted.
    bos_id = tok_fast.bos_token_id
    prepends_bos = bool(ids and bos_id is not None and ids[0] == bos_id and
                        offsets[0][1] - offsets[0][0] == 0)
    bos_verified = ("VERIFIED: add_special_tokens=True does NOT prepend BOS "
                    "(first token id %d, bos_token_id %r, first offset %s). No per-window BOS is "
                    "inserted anywhere: the stream is tokenized once and sliced."
                    % (ids[0], bos_id, tuple(offsets[0])))
    if prepends_bos:
        bos_verified = ("VERIFIED: the tokenizer DOES prepend BOS (id %d) to the joined stream. "
                        "Only window 0 carries it; windows 1..n-1 do not. Recorded because it "
                        "makes window 0 asymmetric with the rest." % bos_id)
    print("      %s" % bos_verified.split(".")[0])

    # ---- 3. the window geometry (read from pt2_llm/eval_ppl.py@9e943e6) -----------------------
    n_windows = n_tokens_total // CONTEXT_LENGTH
    n_tokens_evaluated = n_windows * CONTEXT_LENGTH
    n_target_tokens = n_windows * (CONTEXT_LENGTH - 1)
    print("\n[3/6] window geometry: context=%d stride=%d tail=DROPPED"
          % (CONTEXT_LENGTH, STRIDE))
    print("      n_windows       : %d" % n_windows)
    print("      n_tokens_eval   : %d   (tail dropped: %d tokens)"
          % (n_tokens_evaluated, n_tokens_total - n_tokens_evaluated))
    print("      n_target_tokens : %d   (2047 per window; token 0 of each window is unscored)"
          % n_target_tokens)

    prefix = char_to_byte_prefix(joined)
    end_char = offsets[n_tokens_evaluated - 1][1]
    end_byte = int(prefix[end_char])
    evaluated_bytes = joined_bytes[:end_byte]
    sha_eval = sha256_bytes(evaluated_bytes)
    print("      byte range      : [0, %d)  of %d" % (end_byte, len(joined_bytes)))
    print("      sha256(eval)    : %s" % sha_eval)

    # Bytes spanned by the TARGET tokens only: within each window, tokens 1..2047.
    target_bytes = 0
    for w in range(n_windows):
        first_target = w * CONTEXT_LENGTH + 1
        last_target = (w + 1) * CONTEXT_LENGTH - 1
        start_char = offsets[first_target][0]
        stop_char = offsets[last_target][1]
        target_bytes += int(prefix[stop_char]) - int(prefix[start_char])

    B = target_bytes / n_target_tokens
    B_whole = len(joined_bytes) / n_tokens_total
    print("\n[4/6] B -- MEASURED, never assumed")
    print("      bytes_of_target_region : %d" % target_bytes)
    print("      B (target region)      : %.6f bytes/token" % B)
    print("      B (whole stream)       : %.6f bytes/token" % B_whole)
    print("      agreement              : %.3f%%  (must be < 1%%)"
          % (100.0 * abs(B - B_whole) / B_whole))
    print("      NOTE: the illustrative anchor column in prereg.yaml uses B=4.0. Measured B is "
          "%.3f, so every anchor b/byte value moves by x%.3f." % (B, 4.0 / B))

    # ---- fast/slow parity ---------------------------------------------------------------------
    parity = "NOT RUN"
    try:
        tok_slow = AutoTokenizer.from_pretrained(DONOR_REPO, revision=DONOR_REVISION,
                                                 use_fast=False)
        head = joined[:SLOW_PARITY_CHARS]
        a = tok_fast(head, add_special_tokens=True)["input_ids"]
        b = tok_slow(head, add_special_tokens=True)["input_ids"]
        if a == b:
            parity = ("PASS: fast and slow tokenizers produce IDENTICAL ids on the first %d chars "
                      "(%d tokens). The method builds its tokenizer with use_fast=False; this "
                      "script uses the fast one only because offset mapping requires it."
                      % (SLOW_PARITY_CHARS, len(a)))
        else:
            first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            parity = ("FAIL: fast and slow tokenizers DIVERGE at token %d (%r vs %r) on the first "
                      "%d chars. This is a BLOCK: the byte accounting below is computed under a "
                      "different tokenization than the method will run."
                      % (first, a[first], b[first], SLOW_PARITY_CHARS))
    except Exception as exc:
        parity = ("NOT RUN: the slow tokenizer could not be constructed (%s: %s). The method's "
                  "data.py uses use_fast=False, so this parity check MUST be completed before any "
                  "arm runs." % (type(exc).__name__, str(exc)[:160]))
    print("\n[5/6] fast/slow parity: %s" % parity.split(":")[0])

    # ---- 5. decontamination -------------------------------------------------------------------
    print("\n[6/6] decontamination -- %d-byte shingles at stride %d"
          % (SHINGLE_BYTES, SHINGLE_STRIDE))
    eval_sh = shingle_hashes(evaluated_bytes)
    train_joined = " ".join(train_rows).encode("utf-8")   # the join data.py uses for TRAIN
    train_sh = shingle_hashes(train_joined)
    hit_train = len(eval_sh & train_sh)
    frac_train = hit_train / max(1, len(eval_sh))
    decon_calib = ("MEASURED: %d/%d eval shingles (%.4f%%) also occur in the calibration source "
                   "(wikitext-2-raw-v1 TRAIN, ' '.join, same pinned revision). Splits are disjoint "
                   "by construction; this measures the residual boilerplate overlap."
                   % (hit_train, len(eval_sh), 100.0 * frac_train))
    print("      vs calibration (train split): %d/%d = %.4f%%"
          % (hit_train, len(eval_sh), 100.0 * frac_train))

    if P62_CODEVAL.is_file():
        code_sh = shingle_hashes(P62_CODEVAL.read_bytes())
        hit_code = len(eval_sh & code_sh)
        frac_code = hit_code / max(1, len(eval_sh))
        decon_p62 = ("MEASURED: %d/%d eval shingles (%.4f%%) also occur in the pinned P62 code-val "
                     "(%s, %d bytes)." % (hit_code, len(eval_sh), 100.0 * frac_code,
                                          P62_CODEVAL.as_posix(), P62_CODEVAL.stat().st_size))
        print("      vs P62 code-val             : %d/%d = %.4f%%"
              % (hit_code, len(eval_sh), 100.0 * frac_code))
    else:
        frac_code = None
        decon_p62 = ("NOT RUN: %s not found. The P62 code-val decontamination leg MUST be run "
                     "before any arm." % P62_CODEVAL.as_posix())
        print("      vs P62 code-val             : NOT RUN (file absent)")

    threshold = 0.005
    for label, frac in (("calibration", frac_train), ("P62 code-val", frac_code)):
        if frac is not None and frac > threshold:
            print("      *** THRESHOLD EXCEEDED on %s: %.4f%% > %.2f%%. Per prereg.yaml the slice "
                  "must be re-cut." % (label, 100.0 * frac, 100.0 * threshold))

    # ---- 6. anchor tokenizers -----------------------------------------------------------------
    anchor_lines = []
    if not args.no_anchors:
        print("\n[+] anchor bytes/token (each anchor's PPL ratio converts at ITS OWN B)")
        for label, repo, rev in ANCHOR_TOKENIZERS:
            try:
                t = AutoTokenizer.from_pretrained(repo, revision=rev, use_fast=True)
                n = len(t(joined, add_special_tokens=True)["input_ids"])
                nw = n // CONTEXT_LENGTH
                b_anchor = len(joined_bytes) / n
                anchor_lines.append("%s [%s@%s]: B=%.4f bytes/token on the same pinned slice "
                                    "(%d tokens, %d windows)" % (label, repo, rev[:12], b_anchor,
                                                                 n, nw))
                print("    %s: B=%.4f  (%d tokens)" % (label, b_anchor, n))
            except Exception as exc:
                anchor_lines.append("%s [%s]: UNAVAILABLE (%s: %s)"
                                    % (label, repo, type(exc).__name__, str(exc)[:100]))
                print("    %s: UNAVAILABLE (%s)" % (label, type(exc).__name__))
        anchor_lines.append("Gated anchors (meta-llama/Llama-2-*) are permanently absent: their "
                            "B cannot be measured without accepting the licence, so their b/byte "
                            "values remain derived-under-assumption.")
    anchor_value = " | ".join(anchor_lines) if anchor_lines else \
        "UNPINNED: --no-anchors was passed; the anchor-tokenizer pass did not run."

    # ---- write back ---------------------------------------------------------------------------
    updates = {
        "eval_slice.sha256_joined_text": sha_joined,
        "eval_slice.sha256_evaluated_bytes": sha_eval,
        "eval_slice.joined_text_bytes_total": len(joined_bytes),
        "eval_slice.byte_range_evaluated": "[0, %d) of %d bytes -- the prefix spanned by the %d "
                                           "tokens of the %d complete windows; the trailing %d "
                                           "tokens are dropped by the method's own eval loop"
                                           % (end_byte, len(joined_bytes), n_tokens_evaluated,
                                              n_windows, n_tokens_total - n_tokens_evaluated),
        "eval_slice.n_tokens_total": n_tokens_total,
        "eval_slice.n_windows": n_windows,
        "eval_slice.n_tokens_evaluated": n_tokens_evaluated,
        "eval_slice.n_target_tokens": n_target_tokens,
        "eval_slice.bytes_of_target_region": target_bytes,
        "eval_slice.bos_handling_verified": bos_verified,
        "eval_slice.tokenizer_fast_slow_parity_result": parity,
        "eval_slice.decontamination_result_vs_calibration": decon_calib,
        "eval_slice.decontamination_result_vs_p62_codeval": decon_p62,
        "measured_prerequisites.B_bytes_per_token.value": round(B, 6),
        "measured_prerequisites.B_secondary_whole_stream.value": round(B_whole, 6),
        "measured_prerequisites.B_per_anchor_model.value": anchor_value,
    }

    print("\n" + "=" * 90)
    print("VALUES PINNED (%d fields)" % len(updates))
    print("=" * 90)
    for key, value in updates.items():
        shown = value if not isinstance(value, str) or len(value) <= 96 else value[:93] + "..."
        print("  %-58s = %s" % (key, shown))

    if args.dry_run:
        print("\n--dry-run: prereg.yaml NOT modified.")
        return 0

    lines = prereg_path.read_text(encoding="utf-8").split("\n")
    for key, value in updates.items():
        set_scalar(lines, key, value)
    with io.open(prereg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("\nWROTE %s" % prereg_path)

    yaml.safe_load(prereg_path.read_text(encoding="utf-8"))   # parse-back guard
    print("re-parse OK. Now run:  python check_prereg.py")

    sidecar = HERE / "pin_slice_report.json"
    sidecar.write_text(json.dumps({k: v for k, v in updates.items()}, indent=2), encoding="utf-8")
    print("sidecar: %s" % sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
