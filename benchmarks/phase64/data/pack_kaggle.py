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
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "phase62"))
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
BPEDIR = os.path.join(ROOT, "data", "phase64", "corpus")
OUT = os.path.join(ROOT, "kaggle_rung1")
TAG = "s0"
VAL_FRAC = 0.02


NOTEBOOK = '''# =============================================================================
# rung-1 SCREENING  --  STAGE 1  --  {ARM}   (V={V})
# -----------------------------------------------------------------------------
# Kaggle settings (right sidebar), all three matter:
#   Accelerator : GPU T4 x2      -- Kaggle bills SESSION-hours, so the 2nd T4 is free quota-wise
#   Persistence : "Files only"   -- REQUIRED: keeps /kaggle/working so --resume works across sessions
#   Internet    : OFF is fine    -- nothing is downloaded; stage 1 needs no teacher logits
#
# Run this one cell. It trains until the session budget, then stops cleanly. Re-run the SAME cell
# next session; it resumes. Repeat until it prints STAGE1-DONE.
#
# THE DECIDING METRIC IS THE P62 code-val BPB printed at stage exit, NOT the internal tail val.
# The tail val is apparatus (curves, divergence, liveness) and must never be quoted.
# =============================================================================
import glob, os, subprocess, sys, torch

# ---- the one field the Architect fills at launch, from the prereg -----------------------------
STEPS = None      # <-- stage-C steps for ONE screening arm (15% of the stage-C token budget)
# ------------------------------------------------------------------------------------------------
assert STEPS, ("STEPS is not filled. It is the pre-registered per-arm budget and must come from the "
               "prereg, not from whoever runs the cell. Refusing to start an unbudgeted arm.")

PKG = next((os.path.dirname(p) for p in
            glob.glob('/kaggle/input/**/PACKAGE_MANIFEST.json', recursive=True)), None)
assert PKG, 'package not found: attach the dataset containing PACKAGE_MANIFEST.json'
print('package:', PKG)

# IDENTITY FIRST, and fail closed. The two stage-1 arms differ ONLY in which vocabulary they carry,
# so a swapped dataset trains cleanly and answers a question nobody asked. Nothing in the loss curve
# would look wrong.
sys.path.insert(0, PKG)
import assert_package
assert_package.check(expect_arm='{ARM}', expect_vocab={V}, root=PKG)

CODE = PKG + '/code'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
NG = torch.cuda.device_count()
print('GPUs:', NG, [torch.cuda.get_device_name(i) for i in range(NG)])

TRAIN = CODE + '/benchmarks/phase64/mve/mve_train.py'
args = ['--tag', 's0', '--arm', 'ce', '--recall', 'off', '--stages', 'C',
        '--steps', str(STEPS), '--seq', '512', '--batch', '8',
        '--accum', '1' if NG >= 2 else '2',          # effective batch 16 on 1 or 2 GPUs
        '--fp16', '--warmup', '200', '--max-nonfinite', '50',
        '--data-dir', PKG + '/data', '--ckpt-dir', '/kaggle/working',
        '--out', '/kaggle/working/{ARM}.pt', '--resume-ckpt', '/kaggle/working/resume_{ARM}.pt',
        '--save-stage-ckpt', '/kaggle/working/stages_{ARM}',
        '--resume', '--time-budget-min', '660', '--ckpt-min', '20']
cmd = ([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(NG),
        '--master_port', '29555', TRAIN] if NG >= 2 else [sys.executable, TRAIN]) + args
print('RUN:', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=False)

done = os.path.exists('/kaggle/working/{ARM}.pt.done')
print(os.linesep + '==== STAGE1-DONE for {ARM}? ' + str(done) + ' ====')
print('Send the Builder: the P62 code-val BPB line, the full cell output, and '
      '/kaggle/working/{ARM}.pt (plus stages_{ARM}/ -- stage 2 branches from it).'
      if done else 'NOT finished -> re-run THIS CELL next session (it resumes).')
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
    ids = np.fromfile(os.path.join(DATA, f"ids_V{V}_{TAG}.u16"), dtype=np.uint16).astype(np.int64)
    ends = np.cumsum(np.array(bpe.exp_len, dtype=np.int64)[ids])
    k = int(np.searchsorted(ends, byte_off, "left"))
    # ends[k] is the byte AFTER token k. The cut is exact when some token ends precisely on it -- which is
    # the normal case here, because documents are joined on a trailing newline and that newline is its own
    # token. Then token k belongs to TRAINING, so ntr = k+1. Reporting ends[k-1]-byte_off instead measured
    # the end of the PREVIOUS token and printed -1 on a cut that was in fact exact.
    exact = k < len(ends) and int(ends[k]) == byte_off
    ntr = k + 1 if exact else k
    resid = 0 if exact else int(ends[k] - byte_off)
    return ntr, resid, len(ids), int(ends[-1])


def pack_vocab(V, arm, stage, dest, extras=None):
    os.makedirs(dest, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, f"meta_{TAG}.json")))
    docbound = np.fromfile(os.path.join(DATA, f"docbound_{TAG}.i64"), dtype=np.int64)
    b_off = split_point(meta, docbound)
    ntr, resid, ntok, nbytes = token_index_at(V, b_off)
    row = next(r for r in meta["vocabs"] if r["V"] == V)

    # File names the trainer already expects, so no trainer change is needed to read a rung-1 package.
    pairs = [(os.path.join(DATA, f"ids_V{V}_{TAG}.u16"),     f"ts_{TAG}.u16"),
             (os.path.join(DATA, f"anchors_V{V}_{TAG}.i32"), f"anchors_{TAG}.i32"),
             (os.path.join(DATA, f"t2s_V{V}_{TAG}.i32"),     f"t2s_{TAG}.i32"),
             (os.path.join(DATA, f"decomp_V{V}_{TAG}.npz"),  f"decomp_{TAG}.npz"),
             (os.path.join(BPEDIR, f"bpe{V}_code.bin"),      f"bpe{V}_ts.bin")]
    # P62 code-val, encoded with THIS package's tokenizer. It is the DECIDING metric for the screening
    # (prereg v7): an external, fixed, temporally held-out byte set -- byte-identical across arms by
    # construction and byte-normalised across vocabularies. The internal tail val is apparatus only.
    p62_raw = open(P62, "rb").read()
    from cartography import Bpe
    p62_ids = np.array(Bpe.load(os.path.join(BPEDIR, f"bpe{V}_code.bin")).encode(p62_raw), dtype=np.uint16)
    pdst = os.path.join(dest, "data", f"p62_{TAG}.u16")
    os.makedirs(os.path.dirname(pdst), exist_ok=True)
    p62_ids.tofile(pdst)
    files = {}
    ddir = os.path.join(dest, "data"); os.makedirs(ddir, exist_ok=True)
    pairs = pairs + [(pdst, f"p62_{TAG}.u16")]
    for src, name in pairs:
        dst = os.path.join(ddir, name)
        # p62 is generated straight into the package, so src == dst for that entry. Relying on the size
        # guard below to skip it would work by accident; say it.
        if os.path.abspath(src) != os.path.abspath(dst):
            if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst)
        files[name] = dict(sha256=sha(dst), bytes=os.path.getsize(dst))

    tmeta = dict(corpus=meta["corpus"], bytes=meta["bytes"], teacher=meta["teacher"],
                 V_student=V, n_student_tok=ntok, n_teacher_tok=meta["n_teacher_tok"],
                 bytes_per_student_tok=row["bytes_per_student_tok"],
                 bytes_per_teacher_tok=meta["bytes_per_teacher_tok"],
                 anchors=row["anchors"], anchor_frac=row["anchor_frac"],
                 t2s_mapped=row["t2s_mapped"], t2s_vocab=row["t2s_vocab"],
                 n_train_tok=ntr, n_val_tok=ntok - ntr, tag=TAG,
                 p62_bytes=len(p62_raw), p62_tok=int(len(p62_ids)),
                 p62_note="the DECIDING metric (prereg v7); the internal tail val is record-only apparatus",
                 val_split_byte=b_off, val_split_residual_bytes=resid,
                 val_split_note=("cut at a DOCUMENT boundary in byte space so every vocabulary arm "
                                 "evaluates on byte-identical text"),
                 seg_bytes=meta["bytes"] / max(row["anchors"], 1),
                 seg_student_tok=row["seg_student_tok_mean"], seg_teacher_tok=row["seg_teacher_tok_mean"])
    mp = os.path.join(ddir, f"meta_{TAG}.json")
    json.dump(tmeta, open(mp, "w"), indent=1)
    files[f"meta_{TAG}.json"] = dict(sha256=sha(mp), bytes=os.path.getsize(mp))

    man_code = {}
    cman = json.load(open(os.path.join(BPEDIR, "corpus_manifest.json")))
    man = dict(arm=arm, stage=stage, V_student=V, tag=TAG,
               corpus_sha256=cman["corpus_sha256"],
               raw_sha256=meta["raw_sha256"],
               bpe_sha256=files[f"bpe{V}_ts.bin"]["sha256"],
               ids_sha256=files[f"ts_{TAG}.u16"]["sha256"],
               corpus_bytes_per_tok=cman["tokenizers"][str(V)].get("corpus_bytes_per_tok"),
               val_split_byte=b_off, files=files, code=man_code, **(extras or {}))
    # Code rides WITH the data. It is ~200 KB against a 1 GB payload, and a package that carries its own
    # trainer cannot be run against a stale copy left in /kaggle/working by an earlier session.
    for rel in CODE_FILES:
        cd = os.path.join(dest, "code", os.path.dirname(rel))
        os.makedirs(cd, exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(dest, "code", rel))
        man_code[rel] = sha(os.path.join(dest, "code", rel))
    json.dump(man, open(os.path.join(dest, "PACKAGE_MANIFEST.json"), "w"), indent=1)
    shutil.copyfile(os.path.join(HERE, "assert_package.py"), os.path.join(dest, "assert_package.py"))
    with open(os.path.join(dest, "NOTEBOOK_cell.py"), "w", encoding="utf-8") as f:
        f.write(NOTEBOOK.replace("{ARM}", arm).replace("{V}", str(V)))
    tot = sum(f["bytes"] for f in files.values())
    return man, tot, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    if a.stage != 1:
        sys.exit("stages 2 and 3 depend on the winner of the previous stage; they are packed when it is "
                 "known. Building them now would require choosing a parent that does not exist yet.")

    print(f"WS3 Kaggle packaging   stage 1 (arms 1+2, parallel)   tag={TAG}\n")
    print(f"  {'account':10s} {'arm':6s} {'V':>6s} {'payload':>10s} {'val resid':>10s}  ids sha")
    rows = []
    for acct, (arm, V) in enumerate([("arm1_V2048", 2048), ("arm2_V4096", 4096)], start=1):
        dest = os.path.join(OUT, f"account_{acct}", arm)
        man, tot, resid = pack_vocab(V, arm, 1, dest)
        rows.append((acct, arm, V, tot, resid, man))
        print(f"  {'account_'+str(acct):10s} {arm.split('_')[0]:6s} {V:6d} {tot/2**20:9.0f}M "
              f"{resid:10d}  {man['ids_sha256'][:16]}...")

    # relay carries BOTH: the failure it covers is equiprobable across the two stage-1 arms
    rel = os.path.join(OUT, "account_3_relay")
    for _, arm, V, _, _, _ in rows:
        pack_vocab(V, arm, 1, os.path.join(rel, arm))
    rtot = sum(t for _, _, _, t, _, _ in rows)
    print(f"  {'account_3':10s} {'relay':6s} {'both':>6s} {rtot/2**20:9.0f}M          -  carries arm1 and arm2")

    print(f"\n  val split: byte {rows[0][5]['val_split_byte']} (document boundary), identical for both arms;"
          f" residual {rows[0][4]} / {rows[1][4]} bytes")
    print(f"  packages under {OUT}")
    print("\n  REMINDERS: Kaggle datasets PRIVATE (corpus is never redistributed); every notebook runs\n"
          "  assert_package.py before training; stage-1 packages carry NO logits (only stage 3 needs them).")
    print("\nSTOP. Packages built, nothing launched. No commit.")


if __name__ == "__main__":
    main()
