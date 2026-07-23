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


def data_generation_sha(files):
    """One sha over the package's DATA files -- the generation pin. It changes the moment the file set
    changes, which is exactly the failure it exists to catch: the stage-1 upload was a generation behind
    (6 files, no p62), yet every per-file sha it did carry was internally consistent, so nothing local to
    the manifest could tell it was stale. A hash over the SET makes 'which generation is this' a value the
    cell can pin. Order-independent by sorting: it is a property of the content, not the walk."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(f"{name} {files[name]['sha256']}\n".encode())
    return h.hexdigest()


def check(expect_arm=None, expect_vocab=None, expect_parent_sha=None, expect_alpha=None,
          expect_data_sha=None, root=None, verbose=True):
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

    # Generation pin. Recompute from the files actually on disk (not from a field the manifest could
    # carry stale), verify the manifest's own record agrees, THEN compare to the cell's pinned value.
    # --require-p62 catches the specific failure (that one file missing); this catches the whole class
    # (any generation other than the one the cell was written against).
    gen = data_generation_sha(man["files"])
    if man.get("data_sha256") and man["data_sha256"] != gen:
        raise SystemExit(f"MANIFEST INCONSISTENT: recorded data_sha256 {man['data_sha256'][:16]}... "
                         f"!= recomputed {gen[:16]}... -- the manifest does not describe its own files.")
    if expect_data_sha is not None and gen != expect_data_sha:
        raise SystemExit(
            f"DATA GENERATION MISMATCH -- do not train.\n"
            f"  package data generation {gen}\n"
            f"  this cell was written for {expect_data_sha}\n"
            f"  The attached dataset is a different generation than the notebook expects -- the stage-1\n"
            f"  waste was exactly this: an older package that trained cleanly and computed no decider.")

    if verbose:
        print(f"package OK: arm={man['arm']} stage={man['stage']} V={man['V_student']} "
              f"files={len(man['files'])}")
        if man.get("code"):
            print("  NOTE: this manifest's embedded code shas are SUPERSEDED and are not checked here -- "
                  "the data packages were uploaded before later trainer fixes. Code identity is verified "
                  "against the separate bundle by check_code(); that is the authority.")
        print(f"  corpus {man['corpus_sha256'][:16]}...  raw {man['raw_sha256'][:16]}...  "
              f"bpe {man['bpe_sha256'][:16]}...")
        if man.get("parent_sha256"):
            print(f"  parent ckpt {man['parent_sha256'][:16]}...  alpha={man.get('alpha')}")
    return man


def check_code(code_root, expect_sha, verbose=True):
    """Verify the attached code bundle reproduces the sha the notebook pins.

    THE FAILURE IT EXISTS FOR, and it already happened: PACKAGE_MANIFEST.json recorded a sha256 for every
    code file and nothing ever read them. The stage-1 datasets were uploaded carrying a trainer that
    predates the val-split assert, the CPU refusal and the affirmative P62 invariant -- so the prereg
    conditions held on the Builder's machine and would NOT have held on the machine that produces the gate
    number, with no symptom anywhere. Recording a hash is not verifying it; this is the reader.
    """
    mp = os.path.join(code_root, "CODE_MANIFEST.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"CODE_MANIFEST.json missing at {code_root}. Attach the code bundle dataset.")
    man = json.load(open(mp))
    bad = []
    for rel, want in man["files"].items():
        p = os.path.join(code_root, "code", rel) if rel != "assert_package.py" else os.path.join(code_root, rel)
        if not os.path.isfile(p): bad.append(f"{rel}: MISSING"); continue
        got = _sha(p)
        if got != want: bad.append(f"{rel}: {got[:16]}... != declared {want[:16]}...")
    if bad:
        raise SystemExit("CODE BUNDLE CORRUPT -- do not train:\n  " + "\n  ".join(bad))
    if man["code_sha256"] != expect_sha:
        raise SystemExit(
            f"CODE VERSION MISMATCH -- do not train.\n"
            f"  bundle   {man['code_sha256']}\n"
            f"  expected {expect_sha}\n"
            f"  This notebook was written against a different version of the trainer. Training would run\n"
            f"  cleanly and produce a gate number from code nobody registered.")
    if verbose:
        print(f"code OK: {len(man['files'])} files, sha {man['code_sha256'][:16]}...")
    return man


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm"); ap.add_argument("--vocab", type=int)
    ap.add_argument("--parent-sha"); ap.add_argument("--alpha")
    a = ap.parse_args()
    check(a.arm, a.vocab, a.parent_sha, a.alpha)
