#!/usr/bin/env python3
"""Package the teacher logits as ONE shared Kaggle dataset for the stage-3 alpha curve.

WHY SHARED, not copied into each arm (Architect, stage 3). The three alpha arms {0, 0.25, 0.5} read the
SAME 17.976 GB of logits. A per-arm copy would triple that to 54 GB AND -- the stronger reason -- it would
create three places a wrong generation could hide, each verified only against its own sha. One physical
artefact makes the logits byte-identical across the arms BY CONSTRUCTION, which is what "the curve is one
variable" requires: every input but alpha must be the same file, not three files that happen to match.
Same argument as the shared code bundle, and the same lesson as the stale-package incident.

The integrity pin is LOGITS_SHA: a sha over the per-chunk shas plus the manifest, verified against the
attached dataset by assert_package.check_logits() in the cell -- registered AND checked, never just
registered. The chunks are large, so the per-chunk sha is over the bytes on disk; the bundle sha is a
pure function of the content and independent of walk order.

Run: python benchmarks/phase64/data/pack_logits_bundle.py
"""
import hashlib, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from pack_kaggle import sha, TAG        # noqa: E402  -- one source of the sha helper and the tag

SRC = os.path.join(ROOT, "results", "phase64", "rung1", f"logits_{TAG}")
OUT = os.path.join(ROOT, "kaggle_rung1", "logits_bundle")
LINK = True     # symlink/junction the 18 GB by default; --copy forces a real copy for a portable upload


def generation_sha(files):
    """One sha over the file set -- the same shape as the code/data generation pins."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(f"{name} {files[name]}\n".encode())
    return h.hexdigest()


def main():
    copy = "--copy" in sys.argv
    if not os.path.isdir(SRC):
        sys.exit(f"no logits at {SRC} -- produce them first (mve_logits.py).")
    man = json.load(open(os.path.join(SRC, "manifest.json")))
    if os.path.isdir(OUT): shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, f"logits_{TAG}"), exist_ok=True)

    files = {}
    # The chunk files + the manifest ARE the artefact. Hash every one; a truncated or swapped chunk must
    # change the bundle sha, because a corrupt logit is a silently-wrong KD target, not a crash.
    names = ["manifest.json"] + [c["path"] for c in man["chunks"]]
    total = 0
    for i, name in enumerate(names):
        srcp = os.path.join(SRC, name)
        files[name] = sha(srcp)
        total += os.path.getsize(srcp)
        dstp = os.path.join(OUT, f"logits_{TAG}", name)
        if copy:
            shutil.copyfile(srcp, dstp)
        else:
            # A junction (Windows) / symlink (POSIX) avoids duplicating 18 GB locally. The uploader still
            # has to push the real bytes; --copy makes a self-contained tree when that matters.
            try:
                if os.name == "nt":
                    import subprocess
                    subprocess.run(["cmd", "/c", "mklink", "/H", dstp, srcp], capture_output=True)
                else:
                    os.link(srcp, dstp)
            except Exception:
                shutil.copyfile(srcp, dstp)
        if (i + 1) % 40 == 0 or i + 1 == len(names):
            print(f"  hashed {i+1}/{len(names)} files ...")

    gsha = generation_sha(files)
    json.dump(dict(files=files, logits_sha256=gsha, n_chunks=len(man["chunks"]),
                   slice_sha256=man.get("slice_sha256", ""), K=man["K"],
                   n_teacher_tok=man["n_teacher_tok"], total_bytes=total,
                   note="shared teacher-logit dataset for the stage-3 alpha curve; read via --logits-dir"),
              open(os.path.join(OUT, "LOGITS_MANIFEST.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(HERE, "assert_package.py"), os.path.join(OUT, "assert_package.py"))
    print(f"\n  logits bundle -> {OUT}")
    print(f"  {len(man['chunks'])} chunks + manifest, {total/2**30:.2f} GB  ({'linked' if not copy else 'copied'})")
    print(f"  LOGITS_SHA = '{gsha}'")
    print("\n  Upload kaggle_rung1/logits_bundle/ as ONE PRIVATE dataset, attached to all three alpha cells.")
    print("  (linked build: the upload needs the real bytes -- re-run with --copy for a self-contained tree.)")
    print("\nSTOP. No commit.")


if __name__ == "__main__":
    main()
