#!/usr/bin/env python3
"""Build the CODE bundle as its own small Kaggle dataset, decoupled from the 4 GB of data.

WHY IT IS SEPARATE NOW. The stage-1 data packages were uploaded on 2026-07-21 and carry a COPY of the
trainer. Every fix landed since -- the val-split assert, the CPU refusal, the affirmative P62 invariant --
is absent from that copy, so the prereg v7 conditions would have been satisfied locally and not on the
machine that produces the gate number. Re-uploading 4.2 GB to ship 220 KB of Python is the wrong trade,
and it would recur on the next fix.

THE GAP THIS ALSO CLOSES, which is the more serious half. `PACKAGE_MANIFEST.json` records a sha256 for
every code file, and `assert_package.check()` verified only the DATA files -- the code shas were written
and never read. A stale trainer was therefore not merely possible, it was undetectable: identity check
passes, training runs, the gate number comes out of the wrong code. Recording a hash is not verifying it.

The anchor moves to the notebook cell, which is text and costs nothing to regenerate: the cell pins
CODE_SHA and refuses to train unless the attached bundle reproduces it.

Run: python benchmarks/phase64/data/pack_code_bundle.py
"""
import hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from pack_kaggle import CODE_FILES, sha, assert_whitelist_complete   # one source of truth for the transitive closure  # noqa: E402

OUT = os.path.join(ROOT, "kaggle_rung1", "code_bundle")


def combined(man):
    """One sha over the whole tree. Order-independent by sorting, so it is a property of the CONTENT."""
    h = hashlib.sha256()
    for rel in sorted(man):
        h.update(f"{rel} {man[rel]}\n".encode())
    return h.hexdigest()


def main():
    # BEFORE anything is copied: refuse if mve/ holds a module nobody has classified. Placed first so the
    # refusal costs nothing and cannot half-build a bundle.
    assert_whitelist_complete()
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    man = {}
    for rel in CODE_FILES:
        d = os.path.join(OUT, "code", os.path.dirname(rel))
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(OUT, "code", rel))
        man[rel] = sha(os.path.join(OUT, "code", rel))
    shutil.copyfile(os.path.join(HERE, "assert_package.py"), os.path.join(OUT, "assert_package.py"))
    man["assert_package.py"] = sha(os.path.join(OUT, "assert_package.py"))
    cs = combined(man)
    json.dump(dict(files=man, code_sha256=cs), open(os.path.join(OUT, "CODE_MANIFEST.json"), "w"), indent=1)

    tot = sum(os.path.getsize(os.path.join(OUT, "code", r)) for r in CODE_FILES)
    print(f"code bundle -> {OUT}")
    print(f"  {len(CODE_FILES)} modules + assert_package.py, {tot/1024:.0f} KiB total\n")
    for rel in sorted(man):
        print(f"    {man[rel][:16]}  {rel}")
    print(f"\n  CODE_SHA = '{cs}'")
    print("\n  Paste that value into the notebook cells (they already carry it if generated after this run).")
    print("  Upload kaggle_rung1/code_bundle/ as ONE PRIVATE dataset, attached to every stage-1 notebook.")
    print("\nSTOP. No commit.")


if __name__ == "__main__":
    main()
