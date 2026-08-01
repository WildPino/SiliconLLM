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
import argparse, hashlib, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MVE = os.path.join(ROOT, "benchmarks", "phase64", "mve")
TRAIN_REL = "benchmarks/phase64/mve/mve_train.py"

# The extracted old trainer lives in a SCRATCH DIR, never inside mve/ -- and that is a correctness rule,
# not tidiness. mve/ is a directory the run-packs read from, and packaging copies from the FILESYSTEM, not
# from git: a stale trainer left there can ride inside the GB uploaded to Kaggle. .gitignore would not help
# and would actively hurt -- it protects the repo, not the package, so the file would become invisible to
# history while still travelling. (It did not leak this time only because pack_code_bundle enumerates an
# explicit 8-file list instead of walking the directory -- a property of one packager that nobody had
# asserted. That is the stage-1 incident's exact shape.) Extracting outside every packaged path makes the
# question un-askable next time.
#
# Imports still resolve because the subprocess gets a PYTHONPATH (see run()): the old trainer's own
# sys.path.insert(0, HERE) now points at the scratch dir, which holds no modules, so `mve_model` and
# `cartography` fall through to the injected entries. Its ROOT-derived DATA/CKPT are overridden by
# --data-dir/--ckpt-dir (mve_train.py:272-273); the one ROOT path that is NOT overridable is the
# benchmarks/phase62 insert for cartography (line 354), which is exactly why PYTHONPATH carries it too.
OLD_NAME = "_eps_old_mve_train.py"
SUBPROC_PATH = [MVE, os.path.join(ROOT, "benchmarks", "phase62")]


def mode_args(mode):
    """The arm under test, and the knob its planted control perturbs.

    MODE MATTERS, and picking it wrong makes the gate test nothing. For the conditional-KD-load edit the
    changed behaviour is a SKIP that happens only when the KD apparatus is not needed -- so in KD mode the
    arrays are still loaded and that branch is trivially unchanged: testing it would be testing the code
    that was not modified. CE is where the edit lives, so CE is the claim; KD rides along as a free second
    point confirming the untouched path stayed untouched.

    The control differs with the mode because the two arms have different knobs available:
      kd -> --kd-resident 4 moves the sampling window (proven to shift weights, max|delta| 0.18)
      ce -> there is no window in CE, so the control perturbs the SEED. Its job is unchanged: show that
            this comparison can detect a weight difference at all, so that 'IDENTICAL' is a measurement
            and not the only thing the comparison is capable of printing.
    """
    if mode == "kd":
        return ["--arm", "kd", "--alpha", "0.0"], ["--kd-resident", "4"], "kd-resident 4"
    return ["--arm", "ce"], None, "seed+1"


def base_args(out, data, logits, steps, seed, ckdir):
    """arm4's launch arguments, shortened. Everything that could differ between the runs is fixed here in
    ONE place, so the three invocations cannot drift into being three experiments.

    ckdir is PER-RUN and that is load-bearing, not tidiness. Without it all three runs inherit the trainer's
    DEFAULT resume-checkpoint path and write to the same file. I looked at that once and judged it harmless
    because nothing READS it -- wrong on the write side: two runs racing on the same path die with
    'WinError 32: the file is in use by another process' mid-save, and the gate then concludes nothing from
    a crashed run. Isolation also removes any chance a stale resume from a previous invocation is picked up.
    """
    return ["--tag", "s0", "--recall", "off", "--stages", "C",
            "--steps", str(steps), "--seq", "512", "--batch", "8", "--accum", "1",
            "--warmup", "200", "--max-nonfinite", "50",
            "--xproj-rank", "26", "--chunk-steps", "0",
            "--logits-dir", logits, "--data-dir", data,
            "--ckpt-dir", ckdir, "--resume-ckpt", os.path.join(ckdir, "resume.pt"),
            "--out", out, "--seed", str(seed), "--allow-cpu", "--device", "cpu"]


def run(script, args, tag):
    print(f"  [{tag}] {os.path.basename(script)} {' '.join(a for a in args if a.startswith('--kd-resident') or a == '2' or a == '4')}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(SUBPROC_PATH + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run([sys.executable, script] + args, cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit(f"  [{tag}] FAILED (exit {r.returncode}) -- the gate cannot conclude from a crashed run.")
    return r.stdout


def filesha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def weights_sha(p):
    """sha256 over the WEIGHTS, in sorted key order, including dtype and shape.

    NOT sha256 of the checkpoint file, which was the instruction and is the wrong instrument here: the
    checkpoint also stores the argparse cfg, and the change under test adds --min-host-ram-gib to it. A
    file hash would therefore come out DIFFERENT for a reason that has nothing to do with behaviour, and
    a gate that fails for a benign reason gets argued with instead of believed. This answers the same
    question in bytes -- the trajectory is deterministic, so identical weights means identical run -- and
    it answers it about the thing the run produced rather than about the flags it was invoked with."""
    import torch
    sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def bpbs(out):
    """Every BPB the run printed, in order, as TEXT.

    WHY THE WEIGHT COMPARISON IS NOT ENOUGH (MM, 2026-08-01). This gate's verdict is bit-identity of the
    saved weights, and the whole evaluation path runs under torch.no_grad(): bpb_on never touches a
    parameter. A defect confined to it therefore leaves the gate GREEN while corrupting the reported BPB
    -- including the P62 [DECIDING] line, which is the metric the entire rung is decided on. Exercised is
    not certified, and four of the changed _i64 sites live exactly there. It is the stage-1 family: every
    guard passed and the one number that mattered was the broken one.

    Compared as strings, so no float parsing can round two different values together. RESOLUTION IS 1e-4,
    the trainer's own print precision -- this is NOT bit-identity and must not be described as such. It is
    50x below the sigma_seed adoption bar of 0.005 and 100x below the 2-sigma claim bar, which is what
    makes it adequate for the question it answers, and stating the limit is part of the answer.

    TIMING IS STRIPPED, and the first version of this function did not strip it. The trainer prints the
    stage summary as "val BPB 4.4199 -> 2.5641 (-1.8557) | 242 tok/s | 56.5 min", so taking the whole line
    made the comparison fail on WALL CLOCK -- 242 vs 248 tok/s, 56.5 vs 55.0 min -- while every BPB on it
    was identical. A gate that fails on nondeterministic noise gets overridden, and an overridden gate is
    worse than no gate: it teaches everyone that red means nothing. Fourth time this rung that the defect
    was in the repair rather than in the thing repaired."""
    return [_TIMING.sub("", ln).strip() for ln in out.splitlines() if "BPB" in ln]


# Timing fields on a BPB line: nondeterministic, and never part of a claim about numbers.
_TIMING = re.compile(r"\|\s*[\d.]+\s*(?:tok/s|min)\s*")
# The TRAINING rate, taken only from the stage-summary line.
_STAGE_TPS = re.compile(r"stage \w+ done:.*?\|\s*([\d.]+)\s*tok/s")


def tok_s(out):
    """The TRAINING throughput, from the stage-summary line only.

    The first version matched every "([\\d.]+) tok/s" in the output and averaged them, which mixes the
    training rate with the evaluation rate -- bpb_on is a batched no-grad forward and runs about 9x
    faster. The average came out ~2170 tok/s while the real training rate was 242, and that wrong number
    was used to tell the Capo the gate would take 25 minutes. It took 203. A measurement that is never
    checked against the thing it describes will happily be a different quantity with a plausible value.

    The gate reports old vs new because 3860 tok/s on 2xT4 is what makes the 145-hour calendar fit, and
    the calendar is what makes the account rotation fit."""
    return [float(x) for x in _STAGE_TPS.findall(out)]


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
    ap.add_argument("--mode", choices=["ce", "kd"], default="ce",
                    help="which arm carries the edit under test. Default ce: the conditional-KD-load edit "
                         "changes behaviour only where the apparatus is NOT needed, so KD mode would "
                         "exercise the unmodified branch.")
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
    tmp = tempfile.mkdtemp(prefix="eps_")          # created FIRST: the extraction target lives inside it
    old = os.path.join(tmp, OLD_NAME)
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
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"VOID: {TRAIN_REL} at {resolved[:12]} is byte-identical to the working copy. There is no\n"
                 f"  edit under test -- the gate would compare the trainer to itself and print PASS.")
    print(f"epsilon-identity gate   steps={a.steps} seed={a.seed}\n"
          f"  old = {resolved} : {TRAIN_REL}\n"
          f"        (immutable sha, and verified byte-DIFFERENT from the working copy)\n")
    try:
        pa = os.path.join(tmp, "A_old.pt"); pb = os.path.join(tmp, "B_new_r2.pt"); pc = os.path.join(tmp, "C_new_r4.pt")
        ca, cb, cc = (os.path.join(tmp, d) for d in ("ck_A", "ck_B", "ck_C"))
        for d in (ca, cb, cc): os.makedirs(d, exist_ok=True)
        arm, ctrl, ctrl_name = mode_args(a.mode)
        # BOTH ENDS ARE HASHED BEFORE AND AFTER. Pinning the reference is not enough: the object under
        # test can move instead. It did -- the trainer was edited while a run of this gate was in flight,
        # so run A had already launched against the commit while B and C would have read a different file
        # than A's counterpart. That is the 2026-07-29 law from the side it did not cover. Re-hashing at
        # exit makes it VOID by construction rather than merely unlikely.
        ends0 = {p: filesha(p) for p in (old, new)}
        out_a = run(old, base_args(pa, data, logits, a.steps, a.seed, ca) + arm, f"A old [{a.mode}]")
        out_b = run(new, base_args(pb, data, logits, a.steps, a.seed, cb) + arm, f"B new [{a.mode}]")
        # The control perturbs one knob and nothing else: the window in KD mode, the seed in CE mode.
        c_args = (base_args(pc, data, logits, a.steps, a.seed, cc) + arm + ctrl) if ctrl else \
                 (base_args(pc, data, logits, a.steps, a.seed + 1, cc) + arm)
        run(new, c_args, f"C new [{a.mode}, {ctrl_name}]")

        moved = [p for p, h in ends0.items() if filesha(p) != h]
        if moved:
            print("\n  VERDICT: VOID -- a comparison end changed WHILE the gate was running:\n    "
                  + "\n    ".join(os.path.basename(p) for p in moved)
                  + "\n  The three runs did not all see the same pair of files, so nothing they produced\n"
                    "  can be attributed. Settle the tree and re-run. This is not a failure of the edit.")
            sys.exit(2)
        print("  both ends unchanged across all three runs (re-hashed at exit)")
        # PRINTED so a later check can verify the SHIPPED trainer against the CERTIFIED one without
        # anybody having to remember which file this was. A gate that does not name what it certified
        # leaves the next step to trust rather than to comparison.
        print(f"  CERTIFIED FILE  {TRAIN_REL}\n"
              f"                  sha256 {ends0[new]}")

        same_ab, why_ab, d_ab = compare(pa, pb)
        same_ac, why_ac, d_ac = compare(pa, pc)
        print(f"\n  CLAIM    A(old) vs B(new): {'IDENTICAL' if same_ab else 'DIFFERS'}  -- {why_ab}"
              + (f"  max|delta| {d_ab:.3e}" if not same_ab else ""))
        print(f"  CONTROL  A(old) vs C(new,perturbed): {'IDENTICAL' if same_ac else 'DIFFERS'}  -- {why_ac}"
              + (f"  max|delta| {d_ac:.3e}" if not same_ac else ""))

        # The same claim in bytes. torch.equal above answers tensor by tensor and says WHERE it broke;
        # this says it in one value that can be quoted in a brief without re-running anything.
        ha, hb = weights_sha(pa), weights_sha(pb)
        print(f"\n  weights sha256   A(old) {ha[:32]}...\n"
              f"                   B(new) {hb[:32]}...   {'MATCH' if ha == hb else 'DIFFER'}")
        if (ha == hb) != same_ab:
            print("  WARNING: the hash and the tensor comparison disagree. Trust neither until that is "
                  "explained -- two instruments cannot both be right here.")

        # THE EVALUATION PATH, which the weight comparison structurally cannot reach.
        # THE EMPTY CASE IS VOID, NOT FAIL, AND NEVER PASS. Three outcomes, because there are three
        # states and collapsing any two of them is the failure this whole gate exists to prevent:
        #   ba == bb, non-empty  -> certified to print precision
        #   ba != bb             -> FAIL, the edit changed a reported number
        #   ba empty             -> VOID, the check did not run. NOT a failure of the edit.
        # Written as an early exit rather than a flag, because a flag is what lets "nothing was checked"
        # flow into a branch that prints "0 line(s) match" -- true, useless, and indistinguishable from
        # success at a glance.
        ba, bb = bpbs(out_a), bpbs(out_b)
        if not ba:
            print("\n  VERDICT: VOID -- the runs printed no BPB line, so the evaluation path was not\n"
                  "  compared at all. bpb_on runs under no_grad and the weight comparison is structurally\n"
                  "  blind to it, so without these lines the DECIDING metric's code path is uncertified.\n"
                  "  This is not a failure of the edit: it is an unrun check. Give the gate enough steps\n"
                  "  to reach a stage exit, or point it at data carrying the P62 stream, and re-run.")
            sys.exit(2)
        same_bpb = (ba == bb)
        print(f"\n  reported BPB lines   A(old) {len(ba)}   B(new) {len(bb)}   "
              + ("IDENTICAL to 1e-4 (print precision)" if same_bpb else "DIFFER"))
        for ln in (ba if same_bpb else []):
            print(f"      {ln}")
        if not same_bpb:
            for x, y in zip(ba, bb):
                if x != y:
                    print(f"      A: {x}\n      B: {y}")

        # THROUGHPUT. Absolute numbers here are CPU-local and are NOT the 3860 tok/s T4 figure the
        # calendar is built on; the RATIO is the measurement, and it is device-independent because both
        # runs are the same device with the same recipe.
        ta, tb = tok_s(out_a), tok_s(out_b)
        if ta and tb:
            ma, mb = sum(ta)/len(ta), sum(tb)/len(tb)
            dpc = 100.0 * (mb - ma) / ma
            verdict = ("REGRESSION -- declare it" if dpc < -2.0
                       else "within +/-2%, no schedule impact" if dpc <= 2.0 else "faster")
            print(f"\n  throughput (CPU-local, ratio is the measurement)\n"
                  f"      A(old) {ma:8.1f} tok/s   B(new) {mb:8.1f} tok/s   {dpc:+.1f}%  -- {verdict}")
            if dpc < -2.0:
                print("      The 145-hour calendar is priced on 3860 tok/s on 2xT4; a regression here is a\n"
                      "      schedule failure that no correctness check would have caught.")
        else:
            print("\n  throughput: no 'tok/s' lines in the output -- not measured, and NOT to be reported as "
                  "'no regression'.")

        # The evaluation path is part of the VERDICT, not a note beside it. A green gate over a corrupted
        # [DECIDING] number is worse than a red one.
        if same_ab and not same_bpb:
            print("\n  VERDICT: FAIL -- the weights are bit-identical but the REPORTED BPB is not.\n"
                  "  Everything that decides this rung is read out of the evaluation path, and the weight\n"
                  "  comparison cannot see it: bpb_on runs under no_grad and touches no parameter. A run\n"
                  "  built on this trainer would train correctly and report the wrong number.")
            rc = 1
        elif same_ab and not same_ac:
            # Worded from the ACTUAL mode and control, not from the edit this gate was first written for.
            # The first CE run printed the stage-4 sentence -- "the flag is behaviour-preserving at its
            # default ... may be compared against arm4" -- which named the wrong edit and the wrong control
            # on a log meant to be committed and cited. The measurements were right and the label was not,
            # which is this project's whole failure mode in miniature.
            print(f"\n  VERDICT: PASS -- in {a.mode.upper()} mode the edit is behaviour-preserving (bit-identical\n"
                  f"  to {resolved[:12]}), and the comparison is demonstrably able to detect a difference:\n"
                  f"  perturbing {ctrl_name} moved the weights. 'IDENTICAL' here is a measurement, not the only\n"
                  f"  thing this comparison could have printed.\n"
                  f"  The evaluation path is certified separately and to 1e-4, not bit-exact: {len(ba)} BPB\n"
                  f"  line(s) match, including the P62 [DECIDING] value. bpb_on runs under no_grad, so the\n"
                  f"  weight comparison above is structurally blind to it -- exercised is not certified.")
            rc = 0
        elif not same_ab:
            print(f"\n  VERDICT: FAIL -- in {a.mode.upper()} mode the edit is NOT neutral: it changes the weights\n"
                  f"  against {resolved[:12]}. Anything measured with it carries a second, undeclared variable\n"
                  f"  and cannot be read against runs from the old generation. Fix the trainer, do not relabel\n"
                  f"  the experiment.")
            rc = 1
        else:
            print("\n  VERDICT: VOID -- the planted control did NOT fire: r=4 produced the same weights as the\n"
                  "  old trainer, so this comparison cannot see the window and its 'identical' proves nothing.\n"
                  "  Diagnose the harness before trusting ANY result from it.")
            rc = 2
        print("\nSTOP. No commit.")
        return rc
    finally:
        # ONE cleanup path: the scratch tree holds the extracted trainer, so removing it removes the
        # stale copy too. The previous two-step version left the trainer behind whenever the process was
        # killed mid-run -- which happened, and put a stale trainer inside mve/.
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
