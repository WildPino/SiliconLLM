#!/usr/bin/env python3
"""epsilon-identity gate: prove a trainer edit is behaviour-preserving, instead of asserting it.

WHY THIS EXISTS, in the words of the case that created it. Stage 4 (the record-only order probe) needs
`--kd-resident`, a flag exposing a constructor parameter that was previously hard-wired to 2. The flag
defaults to 2, so BY INSPECTION the edit is neutral and every arm that does not pass it behaves exactly as
before. But the probe compares its result against arm4, which ran under CODE_SHA 232267f2 -- a different
trainer generation. "By inspection" is precisely the standard this project does not accept: the same
report that proposed this check also found a hard-wired 2-sigma threshold that looked fine for weeks and
encoded a rule choosing the OPPOSITE vocabulary to the sealed one. A constant that seemed right and was
not is the argument for running the check rather than reasoning about it.

WHAT IT RUNS -- three arms, because two would not be a test:
    A  OLD trainer (extracted from git at the pinned commit), N steps, seed S
    B  NEW trainer, --kd-resident 2 (the old hard-wired value made explicit), same N, same S
    C  NEW trainer, --kd-resident 4                                          <- THE PLANTED CONTROL

A == B is the claim. C != A is what makes the claim mean anything: without it, "identical" is equally
consistent with a comparison that cannot see the window at all -- comparing two runs that would match no
matter what the flag did. The control is not optional decoration; it is the difference between a test and
a ritual. (Mechanism, verified in the source rather than assumed: KDChunks.window() bounds where batches
are drawn from -- mve_train.py's "sample INSIDE the resident chunk window" -- so a wider resident set moves
the sampled positions even at alpha=0, where the KD loss block is skipped entirely.)

Run:  python benchmarks/phase64/data/ws3_epsilon_identity.py --commit <sha> [--steps 200]
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MVE = os.path.join(ROOT, "benchmarks", "phase64", "mve")
TRAIN_REL = "benchmarks/phase64/mve/mve_train.py"

# The old trainer must run from the mve/ directory: it does sys.path.insert(0, HERE) and imports mve_model
# from there. A copy in a scratch directory would import the wrong module or none at all.
OLD_NAME = "_eps_old_mve_train.py"


def base_args(out, data, logits, steps, seed, ckdir):
    """arm4's launch arguments, shortened. Everything that could differ between the runs is fixed here in
    ONE place, so the three invocations cannot drift into being three experiments.

    ckdir is PER-RUN and that is load-bearing, not tidiness. Without it all three runs inherit the trainer's
    DEFAULT resume-checkpoint path and write to the same file. I looked at that once and judged it harmless
    because nothing READS it -- wrong on the write side: two runs racing on the same path die with
    'WinError 32: the file is in use by another process' mid-save, and the gate then concludes nothing from
    a crashed run. Isolation also removes any chance a stale resume from a previous invocation is picked up.
    """
    return ["--tag", "s0", "--arm", "kd", "--alpha", "0.0", "--recall", "off", "--stages", "C",
            "--steps", str(steps), "--seq", "512", "--batch", "8", "--accum", "1",
            "--warmup", "200", "--max-nonfinite", "50",
            "--xproj-rank", "26", "--chunk-steps", "0",
            "--logits-dir", logits, "--data-dir", data,
            "--ckpt-dir", ckdir, "--resume-ckpt", os.path.join(ckdir, "resume.pt"),
            "--out", out, "--seed", str(seed), "--allow-cpu", "--device", "cpu"]


def run(script, args, tag):
    print(f"  [{tag}] {os.path.basename(script)} {' '.join(a for a in args if a.startswith('--kd-resident') or a == '2' or a == '4')}", flush=True)
    r = subprocess.run([sys.executable, script] + args, cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit(f"  [{tag}] FAILED (exit {r.returncode}) -- the gate cannot conclude from a crashed run.")
    return r.stdout


def compare(pa, pb):
    """Bit-exact comparison of the saved weights. torch.equal, not allclose: epsilon-identity means
    IDENTICAL, and a tolerance would quietly accept the very drift the gate exists to detect."""
    import torch
    a = torch.load(pa, map_location="cpu", weights_only=False)["model"]
    b = torch.load(pb, map_location="cpu", weights_only=False)["model"]
    ka, kb = set(a), set(b)
    if ka != kb:
        return False, f"key sets differ: only-A={sorted(ka-kb)[:4]} only-B={sorted(kb-ka)[:4]}", 0.0
    worst, wname = 0.0, ""
    for k in sorted(ka):
        if a[k].shape != b[k].shape:
            return False, f"{k}: shape {tuple(a[k].shape)} != {tuple(b[k].shape)}", float("inf")
        if not torch.equal(a[k], b[k]):
            d = (a[k].float() - b[k].float()).abs().max().item()
            if d > worst: worst, wname = d, k
    if worst == 0.0:
        return True, "every tensor bit-identical", 0.0
    return False, f"diverges, worst at {wname}", worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # NO DEFAULT, and HEAD is refused. The first run of this gate passed --commit HEAD while the Media
    # Manager committed the very edit under test, so "HEAD" may have resolved to the NEW file and the gate
    # would then have compared the new trainer to ITSELF -- printing PASS on a vacuous claim, with the
    # planted control (C) still firing to make it look healthy. A gate that accepts a MOVING reference can
    # silently compare a thing to itself. Same family as "recording a hash is not verifying it".
    ap.add_argument("--commit", required=True,
                    help="IMMUTABLE commit sha holding the trainer generation to compare against. Branch "
                         "names and HEAD are refused: they move, and a gate must pin what it compares.")
    ap.add_argument("--data", default=os.path.join("kaggle_rung1", "stage3_data", "data"))
    ap.add_argument("--logits", default=os.path.join("kaggle_rung1", "logits_bundle"))
    a = ap.parse_args()

    data = os.path.join(ROOT, a.data); logits = os.path.join(ROOT, a.logits)
    for p in (data, logits):
        if not os.path.isdir(p):
            sys.exit(f"missing input tree: {p}")

    if a.commit.upper() == "HEAD" or "/" in a.commit:
        sys.exit(f"--commit {a.commit!r} is a moving reference. Pass an immutable sha: the gate must pin "
                 f"what it compares, or it can end up comparing the new trainer to itself.")
    rev = subprocess.run(["git", "rev-parse", a.commit], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    if rev.returncode != 0:
        sys.exit(f"cannot resolve commit {a.commit}")
    resolved = rev.stdout.strip()

    # EXTRACT AS BYTES. text=True decodes with the LOCALE encoding (cp1252 on this machine), and this file
    # carries UTF-8 non-ASCII (an em-dash at offset 43); round-tripping it through cp1252 and back out as
    # UTF-8 rewrote those bytes. The extracted "old trainer" was therefore a corrupted copy of the blob --
    # it still parsed, because the damage landed in a comment and stayed valid UTF-8, which is precisely why
    # nothing complained. It also guaranteed the VOID control below could never fire. Bytes in, bytes out.
    old = os.path.join(MVE, OLD_NAME)
    src = subprocess.run(["git", "show", f"{resolved}:{TRAIN_REL}"], cwd=ROOT, stdout=subprocess.PIPE)
    if src.returncode != 0:
        sys.exit(f"cannot extract {TRAIN_REL} from {resolved}")
    with open(old, "wb") as f:
        f.write(src.stdout)
    new = os.path.join(MVE, "mve_train.py")

    # PLANTED CONTROL ON THE GATE'S OWN INPUT, and the one whose absence made the first run unfalsifiable:
    # if the extracted OLD file is byte-identical to the CURRENT one, there is no edit under test and any
    # "IDENTICAL" verdict is a tautology. Refuse before burning three runs on it.
    #
    # LINE ENDINGS ARE NORMALISED FIRST, and the first version of this control was USELESS without it:
    # core.autocrlf=true means `git show` hands back LF while the working copy on disk is CRLF, so a raw
    # byte comparison ALWAYS differs and the control could never fire. Verified by pointing it at a commit
    # that does contain the edit -- it sailed straight past and started running. A control that cannot fire
    # is the exact failure this project keeps re-learning, and it had reappeared inside the fix for it.
    def _norm(p):
        with open(p, "rb") as f:
            return f.read().replace(b"\r\n", b"\n")

    if _norm(old) == _norm(new):
        os.remove(old)
        sys.exit(f"VOID: {TRAIN_REL} at {resolved[:12]} is byte-identical to the working copy. There is no\n"
                 f"  edit under test -- the gate would compare the trainer to itself and print PASS.")
    tmp = tempfile.mkdtemp(prefix="eps_")
    print(f"epsilon-identity gate   steps={a.steps} seed={a.seed}\n"
          f"  old = {resolved} : {TRAIN_REL}\n"
          f"        (immutable sha, and verified byte-DIFFERENT from the working copy)\n")
    try:
        pa = os.path.join(tmp, "A_old.pt"); pb = os.path.join(tmp, "B_new_r2.pt"); pc = os.path.join(tmp, "C_new_r4.pt")
        ca, cb, cc = (os.path.join(tmp, d) for d in ("ck_A", "ck_B", "ck_C"))
        for d in (ca, cb, cc): os.makedirs(d, exist_ok=True)
        run(old, base_args(pa, data, logits, a.steps, a.seed, ca), "A old")
        run(new, base_args(pb, data, logits, a.steps, a.seed, cb) + ["--kd-resident", "2"], "B new r=2")
        run(new, base_args(pc, data, logits, a.steps, a.seed, cc) + ["--kd-resident", "4"], "C new r=4")

        same_ab, why_ab, d_ab = compare(pa, pb)
        same_ac, why_ac, d_ac = compare(pa, pc)
        print(f"\n  CLAIM    A(old) vs B(new,r=2): {'IDENTICAL' if same_ab else 'DIFFERS'}  -- {why_ab}"
              + (f"  max|delta| {d_ab:.3e}" if not same_ab else ""))
        print(f"  CONTROL  A(old) vs C(new,r=4): {'IDENTICAL' if same_ac else 'DIFFERS'}  -- {why_ac}"
              + (f"  max|delta| {d_ac:.3e}" if not same_ac else ""))

        if same_ab and not same_ac:
            print("\n  VERDICT: PASS -- the flag is behaviour-preserving at its default, and the comparison is\n"
                  "  demonstrably able to detect a window change. The probe may be compared against arm4.")
            rc = 0
        elif not same_ab:
            print("\n  VERDICT: FAIL -- the edit is NOT neutral. The probe would carry a second variable and\n"
                  "  could not be read against arm4. Fix the trainer, do not relabel the probe.")
            rc = 1
        else:
            print("\n  VERDICT: VOID -- the planted control did NOT fire: r=4 produced the same weights as the\n"
                  "  old trainer, so this comparison cannot see the window and its 'identical' proves nothing.\n"
                  "  Diagnose the harness before trusting ANY result from it.")
            rc = 2
        print("\nSTOP. No commit.")
        return rc
    finally:
        if os.path.isfile(old): os.remove(old)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
