#!/usr/bin/env python3
"""Assert this package's identity BEFORE training. Run it as the first cell of the notebook.

THE FAILURE CLASS IT EXISTS FOR: a run that is perfectly healthy and measures the wrong thing. The two
stage-1 arms differ only in which vocabulary they carry, so a swapped dataset produces a clean loss
curve, a plausible BPB, and an answer to a question nobody asked. Nothing in the training log would look
wrong. This is the logistics form of a CE control silently receiving KD loss, and the only defence is
that the package declares its full configuration tuple and verifies it against the bytes on disk.

Checked: every file's sha256; the vocabulary the caller expects; the arm id; the corpus manifest; and
for stages 2-3 the parent checkpoint sha and the alpha value -- because a chained arm branching from the
wrong parent is the same failure one stage later, where it is more expensive.

Usage in the notebook (fail closed -- do not train if this raises):
    import assert_package
    assert_package.check(expect_arm="arm1_V2048", expect_vocab=2048)
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()


def check(expect_arm=None, expect_vocab=None, expect_parent_sha=None, expect_alpha=None,
          root=None, verbose=True):
    root = root or HERE
    mp = os.path.join(root, "PACKAGE_MANIFEST.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"PACKAGE_MANIFEST.json missing at {root}. An unidentified package must not train.")
    man = json.load(open(mp))
    ddir = os.path.join(root, "data")
    bad = []
    for name, rec in man["files"].items():
        p = os.path.join(ddir, name)
        if not os.path.isfile(p):
            bad.append(f"{name}: MISSING"); continue
        got = _sha(p)
        if got != rec["sha256"]:
            bad.append(f"{name}: sha {got[:16]}... != declared {rec['sha256'][:16]}...")
    if bad:
        raise SystemExit("PACKAGE INTEGRITY FAILED -- do not train:\n  " + "\n  ".join(bad))

    def want(field, exp):
        if exp is None: return
        got = man.get(field)
        if str(got) != str(exp):
            raise SystemExit(
                f"PACKAGE IDENTITY MISMATCH -- do not train.\n"
                f"  {field}: package says {got!r}, this run expects {exp!r}.\n"
                f"  A run on the wrong package finishes cleanly and answers a different question.")
    want("arm", expect_arm)
    want("V_student", expect_vocab)
    want("parent_sha256", expect_parent_sha)
    want("alpha", expect_alpha)

    if verbose:
        print(f"package OK: arm={man['arm']} stage={man['stage']} V={man['V_student']} "
              f"files={len(man['files'])}")
        print(f"  corpus {man['corpus_sha256'][:16]}...  raw {man['raw_sha256'][:16]}...  "
              f"bpe {man['bpe_sha256'][:16]}...")
        if man.get("parent_sha256"):
            print(f"  parent ckpt {man['parent_sha256'][:16]}...  alpha={man.get('alpha')}")
    return man


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm"); ap.add_argument("--vocab", type=int)
    ap.add_argument("--parent-sha"); ap.add_argument("--alpha")
    a = ap.parse_args()
    check(a.arm, a.vocab, a.parent_sha, a.alpha)
