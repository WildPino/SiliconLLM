#!/usr/bin/env python3
"""Positive control for the run-start val-split assert (prereg v7, condition (b)).

WHY THIS EXISTS. The assert added to mve_train.py verifies that the first n_train_tok tokens of THIS
vocabulary cover exactly `val_split_byte` bytes -- the property that makes the two arms evaluate
byte-identical text. It passed on both packages on the first try. A check that has only ever passed is
indistinguishable from a check that cannot fail, and this one guards the metric that decides the
screening: if it were inert, two arms could silently evaluate different text and the comparison would
run cleanly and mean nothing.

So: build a package view that is byte-identical to the real one EXCEPT for a single wrong number in
meta_s0.json, and require the trainer to refuse it. The big arrays are hard-linked, not copied -- the
control must exercise the real data, and 1.1 GB per run is not a thing to duplicate.

Run: python benchmarks/phase64/data/ws3_valsplit_control.py --pkg kaggle_rung1/account_1/arm1_V2048
"""
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRAIN = os.path.join(ROOT, "benchmarks", "phase64", "mve", "mve_train.py")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(PY): PY = sys.executable


def link_view(src, dst, meta_patch):
    """A package view: every array hard-linked from the original, meta_s0.json rewritten."""
    os.makedirs(dst, exist_ok=True)
    for n in os.listdir(src):
        s, d = os.path.join(src, n), os.path.join(dst, n)
        if os.path.exists(d): os.remove(d)
        if n == "meta_s0.json":
            m = json.load(open(s)); m.update(meta_patch); json.dump(m, open(d, "w"), indent=1)
        else:
            try: os.link(s, d)
            except OSError: shutil.copy2(s, d)      # different volume: fall back, correctness first
    return dst


def run(data_dir, tmp):
    p = subprocess.run([PY, TRAIN, "--tag", "s0", "--arm", "ce", "--recall", "off", "--stages", "C",
                        "--steps", "1", "--seq", "512", "--batch", "2", "--data-dir", data_dir,
                        "--out", os.path.join(tmp, "ctrl.pt"), "--ckpt-dir", tmp],
                       capture_output=True, text=True, timeout=1800)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="kaggle_rung1/account_1/arm1_V2048")
    a = ap.parse_args()
    src = os.path.join(ROOT, a.pkg, "data")
    truth = json.load(open(os.path.join(src, "meta_s0.json")))["val_split_byte"]
    print(f"positive control for the val-split assert   package={a.pkg}   declared split={truth}\n")

    tmp = tempfile.mkdtemp(prefix="valsplit_ctrl_")
    # Off by ONE byte. A large corruption would prove only that the assert notices catastrophes; the
    # failure it must catch is a split that moved slightly -- a different document boundary, a tokenizer
    # rebuilt after the cut -- which is precisely the plausible-artifact case.
    cases = [("off by +1 byte", truth + 1), ("off by -1 byte", truth - 1),
             ("split outside the slice", 10 ** 12)]
    ok = True
    for name, v in cases:
        d = link_view(src, os.path.join(tmp, f"v_{v}"), {"val_split_byte": v})
        rc, out = run(d, tmp)
        refused = rc != 0 and "not the one this arm was registered against" in out
        print(f"  {name:26s} val_split_byte={v:<15d} -> "
              f"{'REFUSED (correct)' if refused else 'ACCEPTED -- THE ASSERT IS INERT'}")
        if not refused:
            print("      " + "\n      ".join(out.strip().splitlines()[-6:]))
        ok &= refused

    d = link_view(src, os.path.join(tmp, "v_true"), {})
    rc, out = run(d, tmp)
    passed = "val split VERIFIED byte-identical" in out
    print(f"  {'unmodified (negative case)':26s} val_split_byte={truth:<15d} -> "
          f"{'ACCEPTED (correct)' if passed else 'REFUSED -- the assert rejects valid data'}")
    ok &= passed

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n==== val-split assert: " + ("VERIFIED (fires on wrong data, passes on right data)"
                                         if ok else "FAILED") + " ====")
    print("\nSTOP. No commit.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
