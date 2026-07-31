#!/usr/bin/env python3
"""WS3 -- assemble the Kaggle packages for the rung-1 screening chain (2-1-2).

CHAIN (adjudicated). The three comparisons are sequential; two of them are two arms wide.
    stage 1   arms 1+2   V2048 vs V4096                       parallel, two accounts
    stage 2   arm  3     x_proj r26, from the vocab winner     alone
    stage 3   arms 4+5   alpha 0.5 and 0.25, shared control    parallel; these consume the logits
Critical path is three arm-durations, not five. The third account is relay: it carries BOTH stage-1
datasets, because a relay that can only cover one of two equiprobable failures is half a relay.

THE VALIDATION SPLIT IS CUT IN BYTES, NOT TOKENS, and this is the one thing here that is not
bookkeeping. The two stage-1 arms tokenize the same corpus differently, so "the last 2% of tokens" is a
DIFFERENT byte range in each -- the arms would compute BPB over different text and the comparison would
be meaningless while looking perfectly healthy. The split is therefore a document boundary in byte
space, converted to a token index per vocabulary, so both arms evaluate on byte-identical text.

EVERY PACKAGE ASSERTS ITS OWN IDENTITY before training. The failure class is a healthy run measuring
the wrong thing -- the logistics version of a CE control silently receiving KD loss -- and the only
defence is that the package declares and verifies the full arm configuration tuple: vocab sha, ids sha,
corpus manifest, arm id, and for stages 2-3 the parent checkpoint sha and alpha.

Run: python benchmarks/phase64/data/pack_kaggle.py --stage 1
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
import assert_package        # noqa: E402  -- the one source of the generation-pin function
DATA = os.path.join(ROOT, "results", "phase64", "rung1")
P62 = os.path.join(ROOT, "results", "phase62", "code_val.txt")
CODE_FILES = ["benchmarks/phase64/mve/mve_train.py", "benchmarks/phase64/mve/mve_model.py",
              "benchmarks/phase55/phase55_ssm.py", "benchmarks/phase57/phase57_sparse.py",
              "benchmarks/phase57/phase57_ternary.py", "benchmarks/phase57/phase59_moe.py",
              # phase58_predict is pulled in by phase59_moe. The list is the TRANSITIVE closure of local
              # imports from the trainer, computed rather than guessed -- the first hand-written version
              # was missing exactly this one and would have failed on Kaggle, not here.
              "benchmarks/phase57/phase58_predict.py",
              "benchmarks/phase62/cartography.py"]

# ---------------------------------------------------------------------------------------------------
# WHITELIST-COMPLETENESS ASSERT over every packaged source directory (Architect, 2026-07-30).
#
# WHAT IT IS FOR. A stale copy of the trainer (`_eps_old_mve_train.py`, left by an epsilon-identity gate
# whose process was SIGKILLed -- `finally` does not run, which is also how arm7 died at resident=16) sat in
# mve/ for a day. It never reached a package, and the ONLY reason is that CODE_FILES is an explicit
# enumeration rather than a directory walk: a property of this packager that nobody had asserted. That is
# precisely the failure class this project keeps paying for, so the implicit property becomes a check.
#
# WHY NOT A BLACKLIST (`_eps_*`): a blacklist catches what you thought of. This catches what you did not.
#
# WHY CLASSIFICATION AND NOT MEMBERSHIP. mve/ holds 15 .py files and only 2 ship; the other 13 are local
# apparatus (acceptance harnesses, MQAR, benches) that must NOT travel. "Refuse anything not in CODE_FILES"
# would therefore refuse 13 healthy files. The rule is COMPLETENESS: every .py here is either SHIPPED or
# declared LOCAL-ONLY, and anything in neither list stops the build until a human says which it is.
#
# THE SIDE EFFECT IS THE POINT: a new module in mve/ blocks packaging until someone declares whether it
# travels. That is a deliberate cost, paid once per module, against silently shipping something.
#
# Layered on purpose, and none of the layers is redundant: .gitignore protects the REPO; the gate's
# `finally` protects the HAPPY PATH; extracting to mkdtemp() removes the PRODUCER; only this assert
# protects what is UPLOADED, and only this one survives a SIGKILL and an unknown future producer.
#
# SCOPE: every directory the whitelist draws from, not just mve/. The axis for extending
# was corrected: the Builder had judged the other three "lower risk" BY INSPECTION -- the exact move that
# missed four defects this week. The MM's datum settles it without theory: copies of sources already exist
# inside benchmark trees (phase56/outputs_run*/SiliconLLM/*.py), so "no automatic producer writes there" is
# a statement about today, not a property of the system. Cost is genuinely nil: packagers are not in the
# bundle, so no CODE_SHA moves and no gate is owed; and all eleven additions are LOCAL-ONLY, so what ships
# cannot change. A zero-risk extension that closes a class is not deferred.
LOCAL_ONLY = {
    "benchmarks/phase64/mve": [
        "data_smoke.py", "kd_information.py", "mve_data.py", "mve_logits.py",
        "ws1_acceptance.py", "ws1_ddp_stop.py", "ws2_acceptance.py", "ws4_branch_check.py",
        "ws4_mqar.py", "ws4_seqbench.py", "ws5_optbench.py", "ws6_collapse.py", "ws6_probe.py",
    ],
    "benchmarks/phase55": ["phase55_export.py", "phase55_export_5m.py", "phase55_interactive.py"],
    "benchmarks/phase57": ["phase58_reg.py", "phase61_projquant.py", "r1_calibration.py"],
    "benchmarks/phase62": ["corpus.py", "e5_0_acceptance.py", "e5_1_mtp.py", "e5_2_carve.py",
                           "task3_catA.py"],
}


def assert_whitelist_complete():
    """Refuse to build if any source directory holds a .py that is neither shipped nor declared local-only.

    Call this at the HEAD of every build path. pack_code_bundle had it and pack_vocab did not -- and
    pack_vocab is the bigger exposure, copying CODE_FILES into every data package (2.6 GB per account).
    Nothing foreign passed only because of the same implicit property this assert exists to convert into a
    check: circular, and on the largest path. It also loses the forcing function -- whoever adds a module
    and runs pack_kaggle instead of pack_code_bundle would build in silence.
    """
    bad = []
    for d, local in LOCAL_ONLY.items():
        dpath = os.path.join(ROOT, *d.split("/"))
        shipped = {os.path.basename(f) for f in CODE_FILES if f.startswith(d + "/")}
        found = {f for f in os.listdir(dpath) if f.endswith(".py")}
        unknown = sorted(found - (shipped | set(local)))
        if unknown:
            bad += [f"{d}/{u}" for u in unknown]
        stale = sorted((shipped | set(local)) - found)
        if stale:
            print(f"  NOTE: declared but absent from {d}: {', '.join(stale)} (stale list, not a blocker)")
    if bad:
        sys.exit(
            "PACKAGING REFUSED -- unclassified module(s) in a packaged source directory:\n  "
            + "\n  ".join(bad)
            + "\n\n  Every .py in these directories must be declared: add it to CODE_FILES if it must travel\n"
              "  inside the Kaggle package, or to LOCAL_ONLY if it stays here. This is not a formality -- a\n"
              "  stale trainer copy once sat in mve/ for a day, and the only thing that kept it out of a\n"
              "  2.6 GB upload was that CODE_FILES happens to be an explicit list. If the file above is\n"
              "  leftover scratch, DELETE it; do not add it to a list to silence this.")
BPEDIR = os.path.join(ROOT, "data", "phase64", "corpus")
OUT = os.path.join(ROOT, "kaggle_rung1")
# ONE tag, naming the corpus build the package was cut from: s0 for the 0.5 GB screening slice, m0 for
# the 5.5 GB main run. Set per stage by STAGE_SRC, overridable with --src-tag.
#
# IT WAS TWO FOR ABOUT AN HOUR, and the split was wrong. The argument was that the package-facing name is
# an INTERFACE the trainer opens (ts_{TAG}.u16) while the source name is mere provenance -- so a main-run
# package would have shipped a file called ts_s0.u16 containing the m0 corpus. The damage is not a
# confusing filename: the trainer's RUN: line and the checkpoint's cfg would both have recorded --tag s0
# while the run consumed m0, and section 19 settled that a record is annotated, never falsified. There we
# kept cfg['out']=w16 precisely BECAUSE it was the record of what had been invoked; here the record would
# have stated something that did not happen. The floor, if this ever costs more than a constant, is the
# startup assert now in mve_train: the ids file size must match the token count declared for the tag.
TAG = "s0"
VAL_FRAC = 0.02

# One arm per screening comparison, keyed by stage. Each spec: (arm id, vocab, x_proj rank, output subdir).
# Stage 1 (vocab) ran and adopted V2048 by the sealed sigma_seed rule. Stage 2 (x_proj r=26) is therefore
# a SINGLE new arm on V2048, fresh seed-paired init, same recipe -- its chained control is stage 1's
# already-computed arm1 (P62 1.1377), so no dense V2048 arm is re-run. The data is byte-identical to
# arm1_V2048, so DATA_SHA will match; only the cell (x_proj rank, arm id) differs.
# Each spec: (arm id, vocab, x_proj rank, data subdir, alpha). alpha=None => CE arm (stages 1-2); a float
# => KD arm on the stage-3 curve, where all three arms share ONE data subdir (stage3_data) and read the
# shared logits bundle -- alpha is the only delta, in the cell.
SPECS = {
    1: [("arm1_V2048", 2048, 0, "account_1/arm1_V2048", None),
        ("arm2_V4096", 4096, 0, "account_2/arm2_V4096", None)],
    2: [("arm3_xproj26_V2048", 2048, 26, "stage2/arm3_xproj26_V2048", None)],
    3: [("arm4_a000_V2048", 2048, 26, "stage3_data", 0.0),
        ("arm5_a025_V2048", 2048, 26, "stage3_data", 0.25),
        ("arm6_a050_V2048", 2048, 26, "stage3_data", 0.5)],
    # STAGE 4 -- ORDER PROBE, RECORD-ONLY, GATES NOTHING. Runs beside the main run, which occupies one
    # account at a time, so it is free on the calendar.
    #
    # WHAT IT ASKS. Stage 3 measured CE-on-window (arm4, resident=2) at 1.1715 against CE-i.i.d. over the
    # SAME token pool (arm3, 1.1377): +0.0338 = 6.8 sigma_seed, roughly five times the entire alpha span the
    # stage existed to measure. Coverage is excluded as the cause -- the slice is 117808/117808 windows and
    # the residence gate passes -- so what is left is the ORDER in which those tokens arrive. This arm makes
    # the window WIDER and nothing else.
    #
    # WHY DOSE-RESPONSE AND NOT A BINARY CONTRAST (Architect). A single wide-vs-narrow pair could only say
    # "something about the window matters". Three points -- resident=2 (1.1715), resident=16, i.i.d. (1.1377)
    # -- can say whether BPB slides monotonically toward the i.i.d. value as the window widens. If it does,
    # the mechanism IS width, and the curve also tells us how much width is enough to buy back the deficit,
    # which is the number rung-2 actually needs. A binary result would leave us knowing neither.
    #
    # AS RUN, not as planned: w16 was attempted and DIED -- host-RAM OOM (SIGKILL at ~24 min, after the
    # guards had passed; a GPU OOM would have raised a CUDA error instead). 16 chunks resident is 2.39 GiB
    # per process and 4.77 GiB under DDP x2, and the session did not have it. Relaunched clean at 8 (2.39
    # GiB total), fresh, not resumed from the w16 partial. The spec carries 8 because that is the point that
    # exists; w16 is UNMEASURED and must never be quoted as if it were. The deviation happens to strengthen
    # the reading: a narrower window should recover LESS of the domain deficit, so crossing the
    # pre-registered threshold at 8 is an a-fortiori result.
    4: [("arm7_w8_V2048", 2048, 26, "stage3_data", 0.0)],
    # STAGE 5 -- THE MAIN RUN. One arm, CE, on the 5.5 GB corpus (src tag m0) rather than the 0.5 GB
    # screening slice. See RECIPE for the curriculum and the accelerator assert; see KD_APPARATUS for why
    # this package does not carry anchors/t2s/decomp.
    5: [("main_V2048_r26_CE", 2048, 26, "main_run", None)],
}
# Which corpus build each stage's packages are cut FROM. Stages 1-4 screened on s0 (0.5 GB); the main run
# uses m0 (5.5 GB, 0.70 expected passes at 1.5 B tokens). Overridable with --src-tag, defaulted per stage
# so the operator cannot pack the main run against the screening slice by omission.
STAGE_SRC = {5: "m0"}
# Sampling-window width per arm; absent = the trainer's default of 2 (what stages 1-3 ran). Kept OUT of the
# SPECS tuple so the existing five-field call sites are untouched -- one probe does not justify reshaping a
# structure four other code paths unpack.
KD_RESIDENT = {"arm7_w8_V2048": 8}
# Does this arm's package carry the KD apparatus (anchors/t2s/decomp)? Same side-table shape as
# KD_RESIDENT, and for the same reason: four code paths unpack the five-field spec tuple.
#
# WHY IT IS DECLARED AND NOT INFERRED FROM alpha. Stages 1 and 2 are CE arms whose packages nevertheless
# SHIPPED the apparatus -- they were built before the trainer's load became conditional, and their
# DATA_SHA is recorded in the brief and in every cell that ran. Deriving the flag from `alpha is None`
# would silently change the bytes of an already-adjudicated package and make a re-pack disagree with the
# record. The label gets corrected; the record does not. Absent here = True = what was actually shipped.
KD_APPARATUS = {"main_V2048_r26_CE": False}

# The recipe an arm's cell runs. SCREEN is what stages 1-4 ran and is the default, so nothing about the
# already-adjudicated cells changes by this table existing.
#
# `extra` is spliced into the arg list as a Python expression, which is why it is a string of source and
# not a list: the cell assembles its args at runtime from NG (GPU count), so the template is source text
# all the way down.
#
# VERIFIED, not assumed: a stage-3 cell regenerated after this change was diffed against the one that
# actually ran. The recipe arguments are identical -- '--recall', 'off', '--stages', 'C', and an empty
# extras list. The cell does differ in three places, all of them separately adjudicated work rather than
# side effects of this table: CODE_SHA (the trainer changed), the --expect-slice-sha pin (Architect,
# decision 4), and the --kd-resident splice (the order probe). Saying "byte-identical" would have been
# the easier sentence and it would have been false.
SCREEN = dict(recall="off", stages="C", extra="[]",
              title="rung-1 SCREENING  --  STAGE {STAGE}",
              steps_note="stage-C steps for ONE screening arm (15% of the stage-C token budget)")
RECIPE = {
    # THE MAIN RUN. Not a screening arm: the full curriculum on the winners the screening adopted --
    # V=2048 (stage 1, sealed sigma_seed rule), x_proj r=26 (stage 2), CE-primary (stage 3 fell to the
    # tie-breaker; span-KD remains CHALLENGER, not baseline).
    #
    # --stages CDEF, so --steps is the TOTAL and the trainer splits it by STAGE_SPLIT (C .55 / D .20 /
    # E .20 / F .05). Stage E's seq 512 -> 2048 trade is internal to the trainer and needs no flag here.
    # --qat-alpha 3600 is the ramp at <=10% of stage D, which the MVE priced when the QAT gate passed with
    # a +0.32 shock.
    #
    # --expect-gpus 2 is deliberate and it KILLS the single-GPU fallback below. That is the point: the
    # 183,105-step budget is arithmetic on 8192 tok/step = batch 8 x seq 512 x accum 1 x 2 GPUs. On one
    # T4 the cell would quietly train a run half the size of the one that was pre-registered, and the
    # deviation would be invisible in the log. Refusing is cheaper than discovering it at the end.
    "main_V2048_r26_CE": dict(
        recall="on", stages="CDEF",
        extra="['--sparse-moe', '--qat-alpha', '3600', "
              "'--expect-gpus', '2', '--expect-gpu-name', 'T4', "
              "'--min-host-ram-gib', '20']",
        title="rung-1 MAIN RUN  --  1.5 B tokens, curriculum C->F",
        steps_note="TOTAL steps across C/D/E/F; the trainer splits by STAGE_SPLIT"),
}
# arm3 (stage 2, CE-on-whole-corpus, P62 1.1377) is the RECORD-ONLY cross-check, NOT the curve's alpha=0
# point: the branch-point control showed CE samples the whole corpus while the KD harness samples the
# resident-logit window, so arm4 (KD alpha=0) is the true alpha=0. arm3 is not re-run and not in SPECS[3].
STAGE3_XCHECK = "arm3_xproj26_V2048 (P62 1.1377, CE-on-whole-corpus): record-only cross-check, not the alpha=0 anchor"


NOTEBOOK = '''# =============================================================================
# {TITLE}  --  {ARM}   (V={V})
# -----------------------------------------------------------------------------
# Kaggle settings (right sidebar), all three matter:
#   Accelerator : GPU T4 x2      -- Kaggle bills SESSION-hours, so the 2nd T4 is free quota-wise
#   Persistence : "Files only"   -- REQUIRED: keeps /kaggle/working so --resume works across sessions
#   Internet    : OFF is fine    -- nothing is downloaded; a run with no LOGITS_SHA needs no teacher
#
# TWO datasets are attached: the arm's data package, and the shared code bundle. They are separate
# because the data is gigabytes and the code is kilobytes, and a trainer fix must not cost a re-upload
# of the corpus. The code that runs is the bundle's, pinned by CODE_SHA below.
#
# Run this one cell. It trains until the session budget, then stops cleanly. Re-run the SAME cell
# next session; it resumes. Repeat until it reports done -- STAGE-DONE for a single-stage screening
# arm, MVE-DONE from the trainer when a multi-stage curriculum finishes its last stage.
#
# THE DECIDING METRIC IS THE P62 code-val BPB printed at stage exit, NOT the internal tail val.
# The tail val is apparatus (curves, divergence, liveness) and must never be quoted.
# =============================================================================
import glob, hashlib, json, os, shutil, subprocess, sys, torch

# ---- the fields the Architect fills at launch, from the prereg ---------------------------------
STEPS = {STEPS}   # {STEPS_NOTE}
CODE_SHA = '{CODE_SHA}'   # pins the trainer generation (checked against the bundle)
DATA_SHA = '{DATA_SHA}'   # pins the DATA generation (checked against the package's file set)
LOGITS_SHA = '{LOGITS_SHA}'   # pins the teacher-logit generation ('' when the stage needs no logits)
SLICE_SHA  = '{SLICE_SHA}'   # pins the SAMPLING DOMAIN (which windows carry teacher signal); '' when no slice
# ------------------------------------------------------------------------------------------------
assert STEPS, ("STEPS is not filled. It is the pre-registered per-arm budget and must come from the "
               "prereg, not from whoever runs the cell. Refusing to start an unbudgeted arm.")

PKG = next((os.path.dirname(p) for p in
            glob.glob('/kaggle/input/**/PACKAGE_MANIFEST.json', recursive=True)), None)
assert PKG, 'data package not found: attach the dataset containing PACKAGE_MANIFEST.json'
BUNDLE = next((os.path.dirname(p) for p in
               glob.glob('/kaggle/input/**/CODE_MANIFEST.json', recursive=True)), None)
assert BUNDLE, 'code bundle not found: attach the dataset containing CODE_MANIFEST.json'
print('data package:', PKG); print('code bundle :', BUNDLE)

# IDENTITY FIRST, and fail closed, on EVERY axis. A swapped dataset trains cleanly and answers a question
# nobody asked; a stale trainer produces a gate number from code nobody registered; wrong logits give the
# alpha arms different teacher signal so the curve stops being one variable. None shows up in the loss
# curve. assert_package is imported from the BUNDLE, not the data package -- the copy inside the package
# predates these checks. expect_arm is None for the stage-3 shared data bundle (the arms are cell-level).
sys.path.insert(0, BUNDLE)
import assert_package
assert_package.check(expect_arm={EXPECT_ARM}, expect_vocab={V}, expect_data_sha=DATA_SHA, root=PKG)
assert_package.check_code(BUNDLE, CODE_SHA)
LOGITS = ''
if LOGITS_SHA:
    LOGITS = next((os.path.dirname(p) for p in
                   glob.glob('/kaggle/input/**/LOGITS_MANIFEST.json', recursive=True)), None)
    assert LOGITS, 'logits bundle not found: attach the dataset containing LOGITS_MANIFEST.json'
    assert_package.check_logits(LOGITS, LOGITS_SHA)      # the third pin: byte-identical teacher signal
    print('logits bundle:', LOGITS)

# STAGE the code onto the writable disk before running it. /kaggle/input is a READ-ONLY mount, and
# several of these modules create their results directory at IMPORT time -- so importing them from the
# mount raises OSError before a single step runs. That is exactly how the first launch attempt died. The
# import-time writes are now guarded at the source as well; this copy is the belt to that pair of braces,
# and it costs 167 KB. It is re-made every session so the staged tree can never drift from the bundle.
CODE = '/kaggle/working/code'
if os.path.isdir(CODE): shutil.rmtree(CODE)
shutil.copytree(BUNDLE + '/code', CODE)
TRAIN = CODE + '/benchmarks/phase64/mve/mve_train.py'

# The file that RUNS is the file that is checked, by explicit path -- and after staging, that is the
# STAGED copy, so this also catches a bad copy. check_code() above validated the bundle as a tree; this
# validates the one path handed to the interpreter. The reason not to trust path precedence is sitting in
# /kaggle/input: the data packages still contain their own stale mve_train.py, so "which copy wins" is a
# real question with a real wrong answer. It is asserted, not assumed.
_want = json.load(open(BUNDLE + '/CODE_MANIFEST.json'))['files']['benchmarks/phase64/mve/mve_train.py']
_got = hashlib.sha256(open(TRAIN, 'rb').read()).hexdigest()
assert _got == _want, ('the trainer about to run is not the one the bundle declares:\\n'
                       '  path %s\\n  sha  %s\\n  want %s' % (TRAIN, _got, _want))
print('trainer   :', TRAIN, '\\n  sha', _got[:16], 'VERIFIED (staged from the read-only bundle)')

# ---- carry the run across Kaggle sessions -------------------------------------------------------
# A batch ("Save Version") run ends at the session limit and /kaggle/working does NOT survive into the
# next one -- it comes back only if the previous version's OUTPUT is attached as an INPUT. This block
# rehydrates from that input. Two rules, both deliberate:
#   1. a file already in /kaggle/working ALWAYS wins -- it is this session's own progress, and copying
#      an older input over it would silently rewind the run;
#   2. nothing is renamed or reinterpreted. The trainer itself refuses a resume file whose stages,
#      steps, arm or VOCABULARY do not match the command, so a wrong carry stops before training
#      instead of producing a clean curve for the wrong arm.
def rehydrate():
    got = []
    for pat, dst in [('resume_{ARM}.pt', '/kaggle/working/resume_{ARM}.pt'),
                     ('{ARM}.pt', '/kaggle/working/{ARM}.pt'),
                     ('{ARM}.pt.done', '/kaggle/working/{ARM}.pt.done')]:
        if os.path.exists(dst):
            got.append(pat + ' (already in working -- kept)'); continue
        hit = [p for p in glob.glob('/kaggle/input/**/' + pat, recursive=True)]
        if hit:
            shutil.copy(hit[0], dst); got.append(pat + ' <- ' + hit[0])
    for d in glob.glob('/kaggle/input/**/stages_{ARM}', recursive=True):
        if not os.path.isdir('/kaggle/working/stages_{ARM}'):
            shutil.copytree(d, '/kaggle/working/stages_{ARM}'); got.append('stages_{ARM}/ <- ' + d)
        break
    print('resume state:', ('\\n  ' + '\\n  '.join(got)) if got else 'nothing carried -- starting fresh')
rehydrate()

if os.path.exists('/kaggle/working/{ARM}.pt.done'):
    print('\\n==== {ARM} ALREADY COMPLETE -- nothing to do. Do not burn a session. ====')
    raise SystemExit(0)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
NG = torch.cuda.device_count()
print('GPUs:', NG, [torch.cuda.get_device_name(i) for i in range(NG)])

args = ['--tag', '{TAG}', {ARM_ARGS} '--recall', '{RECALL}', '--stages', '{STAGES}',
        '--steps', str(STEPS), '--seq', '512', '--batch', '8',
        '--accum', '1' if NG >= 2 else '2',          # effective batch 16 on 1 or 2 GPUs
        '--fp16', '--warmup', '200', '--max-nonfinite', '50', '--require-p62',
        '--xproj-rank', '{XPROJ}',                    # 0 = dense (stage 1); 26 = x_proj (stages 2-3)
        '--chunk-steps', '0',                         # 0 = DERIVE the sweep cadence from the budget; never hand-set
        ] + (['--logits-dir', LOGITS] if LOGITS else []) + (
        # FOURTH PIN. Stage 3 ran with this unset: the slice fingerprint was printed and never verified, so
        # domain identity across the three arms held only because one shared bundle made it hold physically.
        # That is a fortunate packaging property, not a check -- and a property nobody asserted is one a
        # future repackaging can withdraw silently. Passing it converts the observation into a refusal.
        ['--expect-slice-sha', SLICE_SHA] if SLICE_SHA else []) + (
        ['--kd-resident', '{KD_RESIDENT}'] if '{KD_RESIDENT}' else []) + (
        # Whatever this run needs beyond the screening recipe. Empty for stages 1-4, which is why their
        # cells are unchanged by this field existing; the main run puts --sparse-moe, --qat-alpha and the
        # accelerator asserts here. Parameterised rather than forked into a second template: two copies of
        # a 130-line cell drift, and a cell that has drifted is indistinguishable from one that has not.
        {EXTRA_ARGS}) + [
        '--data-dir', PKG + '/data', '--ckpt-dir', '/kaggle/working',
        '--out', '/kaggle/working/{ARM}.pt', '--resume-ckpt', '/kaggle/working/resume_{ARM}.pt',
        '--save-stage-ckpt', '/kaggle/working/stages_{ARM}',
        '--resume', '--time-budget-min', '660', '--ckpt-min', '20']
cmd = ([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(NG),
        '--master_port', '29555', TRAIN] if NG >= 2 else [sys.executable, TRAIN]) + args
print('RUN:', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=False)

done = os.path.exists('/kaggle/working/{ARM}.pt.done')
print(os.linesep + '==== STAGE-DONE for {ARM}? ' + str(done) + ' ====')
print('Send the Builder: the P62 code-val BPB line, the full cell output, and '
      '/kaggle/working/{ARM}.pt (plus stages_{ARM}/ -- stage 2 branches from it).'
      if done else
      'NOT finished. Save Version again, and in the NEW version add this run\\'s OUTPUT as an input\\n'
      '(Add Input -> Notebook Output -> this notebook). rehydrate() picks the resume file up by name;\\n'
      'without that input attached the next session starts from zero and the hours are gone.')
'''


def sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()


def split_point(meta, docbound):
    """A document boundary near (1-VAL_FRAC) of the corpus. Returned in BYTES: it is the same cut for
    every vocabulary, which is the whole point."""
    target = int(meta["bytes"] * (1.0 - VAL_FRAC))
    i = int(np.searchsorted(docbound, target))
    return int(docbound[min(i, len(docbound) - 1)])


def token_index_at(V, byte_off):
    """First student token index whose bytes start at or after byte_off, plus the residual so a cut that
    does not land on a token boundary is visible rather than silently absorbed."""
    from cartography import Bpe
    bpe = Bpe.load(os.path.join(BPEDIR, f"bpe{V}_code.bin"))
    # CHUNKED, for the same reason build_vocab is: the full prefix sum is an int64 array as long as the
    # corpus in tokens -- 18 GiB at m0 scale, and np.cumsum materializes a second one on top of the gather.
    # This function needs exactly one index out of that array and its final element. Walking it in chunks
    # gives both, with a 128 MiB temporary, and the result is identical: `ends` is strictly increasing, so
    # the first chunk whose running total reaches byte_off contains the same token searchsorted would find.
    ids = np.fromfile(os.path.join(DATA, f"ids_V{V}_{TAG}.u16"), dtype=np.uint16)
    exp = np.array(bpe.exp_len, dtype=np.int64)
    CH, base, ntr, resid, found = 1 << 24, 0, None, 0, False
    for i in range(0, len(ids), CH):
        seg = np.cumsum(exp[ids[i:i + CH]]) + base
        if not found and int(seg[-1]) >= byte_off:
            j = int(np.searchsorted(seg, byte_off, "left"))
            # seg[j] is the byte AFTER token i+j. The cut is exact when some token ends precisely on it --
            # the normal case here, because documents are joined on a trailing newline and that newline is
            # its own token. Then that token belongs to TRAINING, so ntr = index+1. Reporting
            # ends[k-1]-byte_off instead measured the end of the PREVIOUS token and printed -1 on a cut
            # that was in fact exact.
            exact = int(seg[j]) == byte_off
            ntr = i + j + 1 if exact else i + j
            resid = 0 if exact else int(seg[j] - byte_off)
            found = True
        base = int(seg[-1])
    if not found:
        raise AssertionError(f"val split byte {byte_off} lies past the end of the token stream ({base})")
    return ntr, resid, len(ids), base


def _materialize_data(V, dest, kd=True):
    """Copy the V-vocabulary data files + write meta into dest/data, returning the file table and the
    values a manifest needs. Shared by the per-arm packages (pack_vocab) and the stage-3 shared data
    bundle (pack_data_bundle), so the two can never disagree on the bytes -- one source, no drift.

    `kd` MUST mirror the trainer's own `need_win` (arm == "kd" or --restrict-to-slice). It is passed from
    the spec's alpha rather than defaulted per call site, so the package and the code that reads it cannot
    drift apart: if the trainer would not open anchors/t2s/decomp, the package does not carry them, and if
    it would, a missing file is a hard open() failure rather than a silent degradation. That coupling is
    the point -- a CE package that shipped the apparatus anyway would be 14 GB instead of 4.3, and one
    that omitted it while the trainer wanted it would fail at rehydrate, not at step 4000."""
    meta = json.load(open(os.path.join(DATA, f"meta_{TAG}.json")))
    docbound = np.fromfile(os.path.join(DATA, f"docbound_{TAG}.i64"), dtype=np.int64)
    b_off = split_point(meta, docbound)
    ntr, resid, ntok, nbytes = token_index_at(V, b_off)
    row = next(r for r in meta["vocabs"] if r["V"] == V)
    if kd and not row.get("kd_apparatus", True):
        sys.exit(f"PACKAGING REFUSED: a KD package was requested for V={V}, but the source build "
                 f"'{TAG}' was made without --kd-apparatus, so anchors/t2s/decomp do not exist.\n"
                 f"  REMEDY: rebuild the corpus with --kd-apparatus. Do NOT pack it as a CE package to "
                 f"get past this -- that would silently change which arm the package can run.")

    # File names the trainer already expects, so no trainer change is needed to read a rung-1 package.
    pairs = [(os.path.join(DATA, f"ids_V{V}_{TAG}.u16"),     f"ts_{TAG}.u16"),
             (os.path.join(BPEDIR, f"bpe{V}_code.bin"),          f"bpe{V}_ts.bin")]
    if kd:
        pairs[1:1] = [(os.path.join(DATA, f"anchors_V{V}_{TAG}.i32"), f"anchors_{TAG}.i32"),
                      (os.path.join(DATA, f"t2s_V{V}_{TAG}.i32"),     f"t2s_{TAG}.i32"),
                      (os.path.join(DATA, f"decomp_V{V}_{TAG}.npz"),  f"decomp_{TAG}.npz")]
    # P62 code-val, encoded with THIS package's tokenizer. It is the DECIDING metric for the screening
    # (prereg v7): an external, fixed, temporally held-out byte set -- byte-identical across arms by
    # construction and byte-normalised across vocabularies. The internal tail val is apparatus only.
    p62_raw = open(P62, "rb").read()
    from cartography import Bpe
    p62_ids = np.array(Bpe.load(os.path.join(BPEDIR, f"bpe{V}_code.bin")).encode(p62_raw), dtype=np.uint16)
    ddir = os.path.join(dest, "data"); os.makedirs(ddir, exist_ok=True)
    pdst = os.path.join(ddir, f"p62_{TAG}.u16")
    p62_ids.tofile(pdst)
    files = {}
    for src, name in pairs + [(pdst, f"p62_{TAG}.u16")]:
        dst = os.path.join(ddir, name)
        # p62 is generated straight into the package, so src == dst for that entry. Relying on the size
        # guard below to skip it would work by accident; say it.
        if os.path.abspath(src) != os.path.abspath(dst):
            if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst)
        files[name] = dict(sha256=sha(dst), bytes=os.path.getsize(dst))

    # The KD-derived fields stay in the schema as nulls rather than disappearing: a reader that finds no
    # 'anchors' key cannot tell "CE package" from "older packer". None says it, and says it explicitly.
    tmeta = dict(corpus=meta["corpus"], bytes=meta["bytes"], teacher=meta.get("teacher") if kd else None,
                 V_student=V, n_student_tok=ntok,
                 n_teacher_tok=meta.get("n_teacher_tok") if kd else None,
                 bytes_per_student_tok=row["bytes_per_student_tok"],
                 bytes_per_teacher_tok=meta.get("bytes_per_teacher_tok") if kd else None,
                 anchors=row["anchors"] if kd else None,
                 anchor_frac=row["anchor_frac"] if kd else None,
                 t2s_mapped=row["t2s_mapped"] if kd else None,
                 t2s_vocab=row["t2s_vocab"] if kd else None,
                 kd_apparatus=bool(kd),
                 kd_apparatus_note=("anchors/t2s/decomp are shipped; this package can run --arm kd" if kd
                                    else "CE package: no teacher-derived arrays, and no sampling-window "
                                         "restriction, so there is no slice to pin"),
                 n_train_tok=ntr, n_val_tok=ntok - ntr, tag=TAG,
                 p62_bytes=len(p62_raw), p62_tok=int(len(p62_ids)),
                 p62_note="the DECIDING metric (prereg v7); the internal tail val is record-only apparatus",
                 val_split_byte=b_off, val_split_residual_bytes=resid,
                 val_split_note=("cut at a DOCUMENT boundary in byte space so every vocabulary arm "
                                 "evaluates on byte-identical text"),
                 seg_bytes=(meta["bytes"] / max(row["anchors"], 1)) if kd else None,
                 seg_student_tok=row["seg_student_tok_mean"] if kd else None,
                 seg_teacher_tok=row["seg_teacher_tok_mean"] if kd else None)
    mp = os.path.join(ddir, f"meta_{TAG}.json")
    json.dump(tmeta, open(mp, "w"), indent=1)
    files[f"meta_{TAG}.json"] = dict(sha256=sha(mp), bytes=os.path.getsize(mp))
    return files, meta, row, b_off, resid, ntok


def _base_manifest(V, files, b_off, meta):
    """The vocabulary/corpus identity fields common to every package and bundle."""
    cman = json.load(open(os.path.join(BPEDIR, "corpus_manifest.json")))
    return dict(V_student=V, tag=TAG,
                corpus_sha256=cman["corpus_sha256"], raw_sha256=meta["raw_sha256"],
                bpe_sha256=files[f"bpe{V}_ts.bin"]["sha256"], ids_sha256=files[f"ts_{TAG}.u16"]["sha256"],
                data_sha256=assert_package.data_generation_sha(files),   # the generation pin
                corpus_bytes_per_tok=cman["tokenizers"][str(V)].get("corpus_bytes_per_tok"),
                val_split_byte=b_off, files=files)


def pack_data_bundle(V, xproj, dest):
    """The SHARED data bundle for the stage-3 alpha curve: the identical V-data every alpha arm reads, as
    ONE dataset. No arm id (the arms are cell-level, differing only in alpha), no code (the code bundle is
    separate). One physical artefact makes the data byte-identical across the three arms by construction."""
    if os.path.isdir(dest): shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    files, meta, row, b_off, resid, ntok = _materialize_data(V, dest)
    man = dict(stage=3, xproj_rank=xproj, shared=True, **_base_manifest(V, files, b_off, meta))
    json.dump(man, open(os.path.join(dest, "PACKAGE_MANIFEST.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(HERE, "assert_package.py"), os.path.join(dest, "assert_package.py"))
    return man, sum(f["bytes"] for f in files.values())


def pack_vocab(V, arm, stage, dest, xproj=0, extras=None):
    os.makedirs(dest, exist_ok=True)
    files, meta, row, b_off, resid, ntok = _materialize_data(V, dest, kd=KD_APPARATUS.get(arm, True))
    man_code = {}
    man = dict(arm=arm, stage=stage, xproj_rank=xproj, code=man_code,
               **_base_manifest(V, files, b_off, meta), **(extras or {}))
    # Code rides WITH the data. It is ~200 KB against a 1 GB payload, and a package that carries its own
    # trainer cannot be run against a stale copy left in /kaggle/working by an earlier session.
    for rel in CODE_FILES:
        cd = os.path.join(dest, "code", os.path.dirname(rel))
        os.makedirs(cd, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(dest, "code", rel))
        man_code[rel] = sha(os.path.join(dest, "code", rel))
    json.dump(man, open(os.path.join(dest, "PACKAGE_MANIFEST.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(HERE, "assert_package.py"), os.path.join(dest, "assert_package.py"))
    # The notebook cell is written centrally by write_cells (which reads this manifest for DATA_SHA), not
    # here: a cell needs STEPS and CODE_SHA, neither known at build time, and one authoritative copy that
    # the operator pastes from beats a per-package copy that can drift from it.
    tot = sum(f["bytes"] for f in files.values())
    return man, tot, resid


def write_cells(stage, steps, code_sha):
    """Regenerate the notebook cells, the single authoritative copy the operator pastes from -- written
    OUTSIDE the package directories so it can never be confused with a shipped-and-stale one. DATA_SHA is
    read from each package's freshly built manifest, so the cell pins the exact generation it was written
    against. Stage 3 additionally pins LOGITS_SHA from the shared logits bundle and sets --arm kd --alpha."""
    logits_sha = slice_sha = ""
    if any(alpha is not None for *_, alpha in SPECS[stage]):
        lm = os.path.join(OUT, "logits_bundle", "LOGITS_MANIFEST.json")
        if not os.path.isfile(lm):
            sys.exit("no logits bundle -- run pack_logits_bundle.py first. The alpha cells pin a sha they "
                     "cannot invent.")
        lman = json.load(open(lm))
        logits_sha = lman["logits_sha256"]
        # The slice fingerprint rides in the LOGITS manifest, not the data one -- which chunk positions carry
        # teacher signal is a property of the logit generation. Absent = a bundle built before slicing; the
        # cell then leaves the pin empty rather than inventing one, and the trainer prints-but-does-not-verify
        # exactly as stage 3 did. Refusing here would break replay of older bundles for no safety gain.
        slice_sha = lman.get("slice_sha256", "")
        if not slice_sha:
            print("  NOTE: logits manifest carries no slice_sha256 -- cells ship with the domain pin EMPTY.")
    out = []
    for arm, V, xproj, subdir, alpha in SPECS[stage]:
        rcp = RECIPE.get(arm, SCREEN)
        mp = os.path.join(OUT, subdir, "PACKAGE_MANIFEST.json")
        if not os.path.isfile(mp):
            sys.exit(f"no package at {subdir} -- build stage {stage} first (pack_kaggle.py --stage {stage}).")
        man = json.load(open(mp))
        # Recompute if absent: the stage-1 packages were built before the field existed, and the pin is a
        # pure function of the file set either way -- so the cell gets the true generation, not a KeyError.
        data_sha = man.get("data_sha256") or assert_package.data_generation_sha(man["files"])
        # KD arm (alpha set) vs CE arm: the arm flag, the alpha, whether logits are pinned, and whether the
        # data package carries its own arm id (the shared stage-3 bundle does not, so expect_arm is None).
        if alpha is None:
            arm_args = "'--arm', 'ce',"; expect_arm = f"'{arm}'"; lsha = ""
        else:
            arm_args = f"'--arm', 'kd', '--alpha', '{alpha}',"; expect_arm = "None"; lsha = logits_sha
        p = os.path.join(OUT, f"NOTEBOOK_{arm}.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(NOTEBOOK.replace("{TITLE}", rcp["title"].replace("{STAGE}", str(stage)))
                            .replace("{STEPS_NOTE}", rcp["steps_note"]).replace("{TAG}", TAG)
                            .replace("{ARM}", arm).replace("{V}", str(V)).replace("{STAGE}", str(stage))
                            .replace("{XPROJ}", str(xproj)).replace("{STEPS}", str(steps))
                            .replace("{CODE_SHA}", code_sha).replace("{DATA_SHA}", data_sha)
                            .replace("{LOGITS_SHA}", lsha).replace("{EXPECT_ARM}", expect_arm)
                            .replace("{SLICE_SHA}", slice_sha if alpha is not None else "")
                            .replace("{KD_RESIDENT}", str(KD_RESIDENT.get(arm, "")))
                            .replace("{RECALL}", rcp["recall"]).replace("{STAGES}", rcp["stages"])
                            .replace("{EXTRA_ARGS}", rcp["extra"])
                            .replace("{ARM_ARGS}", arm_args))
        out.append((p, data_sha))
    return out


def main():
    # HEAD OF THE BUILD PATH. pack_vocab() copies CODE_FILES into every data package and never checked
    # what else was sitting in those directories; the check lived only in pack_code_bundle, which is the
    # SMALLER path. Placed here it covers every invocation, including --cells-only, and restores the
    # forcing function for whoever reaches for this script rather than the other one.
    assert_whitelist_complete()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--cells-only", action="store_true",
                    help="regenerate the notebook cells against the current code bundle, without "
                         "rebuilding the data packages")
    ap.add_argument("--steps", type=int, default=0, help="pre-registered per-arm stage-C step budget")
    ap.add_argument("--src-tag", default="",
                    help="which corpus build under results/phase64/rung1 to cut the package from. "
                         "Defaults per stage via STAGE_SRC (s0 for the screening, m0 for the main run); "
                         "pass it only to override deliberately.")
    a = ap.parse_args()
    global TAG
    TAG = a.src_tag or STAGE_SRC.get(a.stage, TAG)
    if a.stage not in SPECS:
        sys.exit(f"stage {a.stage} is not defined. Stages 2-3 are packed only once the previous winner is "
                 f"known -- their parent must exist.")
    if a.cells_only:
        cm = os.path.join(OUT, "code_bundle", "CODE_MANIFEST.json")
        if not os.path.isfile(cm):
            sys.exit("no code bundle: run pack_code_bundle.py first. The cell pins a sha it cannot invent.")
        bman = json.load(open(cm))
        cs = bman["code_sha256"]
        # REFUSE to pin a sha from a bundle that no longer matches the repo. This exact mistake was made
        # once already: a bundled file was edited after the bundle was built, the cells were generated
        # from the stale manifest, and the cell would have refused the very bundle it shipped with. The
        # anchor is only worth having if it cannot be generated out of date.
        drift = [rel for rel, want in bman["files"].items()
                 if sha(os.path.join(ROOT, rel) if rel != "assert_package.py"
                        else os.path.join(HERE, "assert_package.py")) != want]
        if drift:
            sys.exit("the code bundle is STALE relative to the repo -- re-run pack_code_bundle.py first:\n  "
                     + "\n  ".join(drift))
        if not a.steps:
            sys.exit("--steps is required: the cell must carry the pre-registered budget, and a cell that "
                     "silently defaults to a number nobody registered is the failure this field exists for.")
        for p, ds in write_cells(a.stage, a.steps, cs):
            print(f"  wrote {p}   DATA_SHA {ds[:16]}...")
        print(f"\n  stage {a.stage}   STEPS = {a.steps}   CODE_SHA = {cs[:16]}...")
        print("\nSTOP. Cells regenerated. No commit.")
        return

    label = {1: "stage 1 (vocab, arms 1+2 parallel)",
             2: "stage 2 (x_proj r=26 on V2048, one arm; chained control = stage-1 arm1)",
             3: "stage 3 (alpha curve {0,0.25,0.5}, ONE shared data bundle + shared logits)",
             4: "stage 4 (order probe, record-only: window width 2 -> 8, everything else = arm4)",
             5: "stage 5 (MAIN RUN: V2048 + x_proj r=26 + CE, curriculum C->F on the 5.5 GB corpus)"}[a.stage]
    print(f"WS3 Kaggle packaging   {label}   tag={TAG}\n")

    if a.stage == 4:
        # The probe REUSES stage 3's bundles untouched -- that is the whole point. Rebuilding the data would
        # risk a fresh generation sha and turn "same inputs, wider window" into "two things changed", which
        # is the one reading the probe cannot afford.
        subdir = SPECS[4][0][3]
        if not os.path.isfile(os.path.join(OUT, subdir, "PACKAGE_MANIFEST.json")):
            sys.exit(f"stage 4 reuses stage 3's data bundle at {subdir}, which is not built. Pack stage 3 first.")
        print(f"  reusing stage-3 bundles unchanged (data {subdir}, code, logits) -- nothing rebuilt.")
        print(f"  probe arm: {SPECS[4][0][0]}   window resident={KD_RESIDENT[SPECS[4][0][0]]} "
              f"(vs 2 for arm4)   alpha=0.0, V2048, r=26 -- identical to arm4 otherwise")
        print(f"\n  Next: pack_kaggle.py --cells-only --stage 4 --steps <N>")
        print("\nSTOP. No bundle rebuilt, nothing launched. No commit.")
        return

    if a.stage == 3:
        # ONE shared data bundle for all three alpha arms -- built once, not per arm. The logits are their
        # own shared bundle (pack_logits_bundle.py); the arms differ only in alpha, set in the cell.
        subdir = SPECS[3][0][3]                                  # all three point at the same data subdir
        V, xproj = SPECS[3][0][1], SPECS[3][0][2]
        man, tot = pack_data_bundle(V, xproj, os.path.join(OUT, subdir))
        print(f"  shared data bundle -> {subdir}   V{V} r={xproj}   {tot/2**20:.0f} MiB   "
              f"DATA_SHA {man['data_sha256'][:16]}...")
        lm = os.path.join(OUT, "logits_bundle", "LOGITS_MANIFEST.json")
        if os.path.isfile(lm):
            print(f"  shared logits bundle: LOGITS_SHA {json.load(open(lm))['logits_sha256'][:16]}... "
                  f"(build/upload separately: pack_logits_bundle.py)")
        else:
            print("  NOTE: logits bundle not built yet -- run pack_logits_bundle.py before --cells-only.")
        print(f"  arms (cell-level, alpha only): {', '.join(f'{arm}(a={al})' for arm,_,_,_,al in SPECS[3])}")
        print(f"  cross-check: {STAGE3_XCHECK}")
        print(f"\n  Next: pack_kaggle.py --cells-only --stage 3 --steps <N>  (fills STEPS+CODE_SHA+DATA_SHA+LOGITS_SHA)")
        print("\nSTOP. Shared bundle built, nothing launched. No commit.")
        return

    print(f"  {'arm':20s} {'V':>6s} {'xproj':>6s} {'payload':>10s} {'val resid':>10s}  data sha")
    rows = []
    for arm, V, xproj, subdir, _alpha in SPECS[a.stage]:
        dest = os.path.join(OUT, subdir)
        man, tot, resid = pack_vocab(V, arm, a.stage, dest, xproj=xproj)
        rows.append((arm, V, tot, resid, man))
        print(f"  {arm:20s} {V:6d} {xproj:6d} {tot/2**20:9.0f}M {resid:10d}  {man['data_sha256'][:16]}...")

    if a.stage == 1:
        # relay carries BOTH: the failure it covers is equiprobable across the two stage-1 arms
        rel = os.path.join(OUT, "account_3_relay")
        for arm, V, _, _, _ in rows:
            pack_vocab(V, arm, 1, os.path.join(rel, arm))
        print(f"  {'account_3 relay':20s} {'both':>6s} {'':>6s} "
              f"{sum(t for _, _, t, _, _ in rows)/2**20:9.0f}M           -  carries arm1 and arm2")

    print(f"\n  val split: byte {rows[0][4]['val_split_byte']} (document boundary), identical across arms")
    print(f"  packages under {OUT}")
    print("\n  REMINDERS: Kaggle datasets PRIVATE (corpus is never redistributed); the cell asserts the data\n"
          "  generation (DATA_SHA) and requires the p62 decider (--require-p62) before training. Stage-1/2\n"
          "  packages carry NO logits (only stage 3 needs them).")
    print(f"\n  Next: pack_kaggle.py --cells-only --stage {a.stage} --steps <N>   (fills STEPS + CODE_SHA + DATA_SHA)")
    print("\nSTOP. Packages built, nothing launched. No commit.")


if __name__ == "__main__":
    main()
