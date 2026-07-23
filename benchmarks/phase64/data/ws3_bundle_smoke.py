#!/usr/bin/env python3
"""Can the code bundle be imported and run from a READ-ONLY location?

WHY THIS EXISTS, and it is a confession. The bundle was rehearsed locally by running the trainer straight
out of kaggle_rung1/code_bundle/ -- on a writable disk. It passed. On Kaggle the same bundle is mounted at
/kaggle/input, which is read-only, and five of the eight modules call os.makedirs on a results directory
AT IMPORT TIME. The first launch died in the import, before a single step, with OSError: Read-only file
system. The rehearsal was faithful in everything except the one property that broke.

So the property is now tested directly: deny write on the staging copy and require the run to reach
training anyway. A rehearsal that cannot fail the way production fails is not a rehearsal.

Run: python benchmarks/phase64/data/ws3_bundle_smoke.py
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BUNDLE = os.path.join(ROOT, "kaggle_rung1", "code_bundle")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY): PY = sys.executable


def set_readonly(path, on=True):
    """Deny write to this user, recursively. A DENY ace outranks ownership, which a read-only ATTRIBUTE
    does not -- and the point is to reproduce a mount the process genuinely cannot write to."""
    who = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if os.name == "nt":
        # NOT the simple right (W). icacls's W is FILE_GENERIC_WRITE, which contains SYNCHRONIZE -- and
        # SYNCHRONIZE is required to OPEN a file for synchronous I/O at all, so denying W makes the tree
        # unreadable and the interpreter cannot even load the script. That is a different failure wearing
        # the same word ("Permission denied"), and it would have been easy to read as the fix not working.
        # Deny the specific mutation rights instead: write/append data, write EA/attributes, delete.
        flag = "/deny" if on else "/remove:d"
        arg = f"{who}:(OI)(CI)(WD,AD,WEA,WA,DC,DE)" if on else who
        r = subprocess.run(["icacls", path, flag, arg, "/T", "/C", "/Q"],
                           capture_output=True, text=True)
        return r.returncode == 0
    mode = 0o555 if on else 0o755
    for d, _, fs in os.walk(path):
        for f in fs: os.chmod(os.path.join(d, f), 0o444 if on else 0o644)
        os.chmod(d, mode)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="kaggle_rung1/account_1/arm1_V2048/data")
    ap.add_argument("--bundle", default=BUNDLE,
                    help="bundle to test. Used to point the smoke at a deliberately broken copy, which is "
                         "how it was shown to fire rather than merely to pass.")
    a = ap.parse_args()
    if not os.path.isdir(a.bundle):
        sys.exit(f"no bundle at {a.bundle} -- run pack_code_bundle.py first.")

    tmp = tempfile.mkdtemp(prefix="ro_bundle_")
    stage = os.path.join(tmp, "code_bundle")
    shutil.copytree(a.bundle, stage)
    work = os.path.join(tmp, "work"); os.makedirs(work)

    if not set_readonly(stage, True):
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("could not make the staging copy read-only; refusing to report a PASS from a writable "
                 "directory -- that is the exact mistake this smoke exists to prevent.")
    # Prove the denial is real before trusting the result: this is the planted control OF THE HARNESS.
    probe_ok = False
    try:
        open(os.path.join(stage, "code", "_probe.tmp"), "w").close()
    except OSError:
        probe_ok = True
    if not probe_ok:
        set_readonly(stage, False); shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("the staging copy is still writable -- the read-only emulation did not take, so a PASS "
                 "here would mean nothing.")
    print(f"read-only emulation VERIFIED on {stage}\n  (a write probe raised OSError, as it must)\n")

    train = os.path.join(stage, "code", "benchmarks", "phase64", "mve", "mve_train.py")
    p = subprocess.run([PY, train, "--tag", "s0", "--arm", "ce", "--recall", "off", "--stages", "C",
                        "--steps", "1", "--seq", "512", "--batch", "2",
                        "--data-dir", os.path.join(ROOT, a.data_dir),
                        "--out", os.path.join(work, "s.pt"), "--ckpt-dir", work],
                       capture_output=True, text=True, timeout=3600)
    out = (p.stdout or "") + (p.stderr or "")
    set_readonly(stage, False)
    shutil.rmtree(tmp, ignore_errors=True)

    imported = "student S0:" in out
    trained = "stage C" in out
    if "Read-only file system" in out:
        print("  IMPORT-TIME WRITE still present:")
        for l in out.splitlines():
            if "Read-only" in l or "makedirs" in l: print("    " + l.strip())
    print(f"  imports resolved from the read-only tree : {'YES' if imported else 'NO'}")
    print(f"  training reached                          : {'YES' if trained else 'NO'}")
    if not (imported and trained):
        print("\n  last 12 lines:\n    " + "\n    ".join(out.strip().splitlines()[-12:]))
    print("\n==== read-only bundle smoke: " + ("PASS" if (imported and trained) else "FAIL") + " ====")
    print("\nSTOP. No commit.")
    sys.exit(0 if (imported and trained) else 1)


if __name__ == "__main__":
    main()
