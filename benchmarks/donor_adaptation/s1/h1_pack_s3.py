#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1 session 3 (addendum M) -- re-point the existing bundle at s2 and re-hash it.

This is NOT a re-pack.  `h1_pack.py` built the bundle behind a chain of preconditions that
have already fired and whose inputs have not changed; re-running it would rebuild artefacts
that are byte-identical and would need the whole gate chain live again.  What changes between
session 2 and session 3 is exactly three things:

    * the resume artefact:  h1_trained_s1.npz  ->  h1_trained_s2.npz
    * the trainer:          addendum M's steps_completed / stop_reason / seconds_per_step
    * the command:          --max-hours 2.8 -> 11.0, seed 3141, --out h1_trained_s3.npz

and this script does those three and re-hashes EVERY file, so the manifest still describes
what is actually in the directory rather than what was there in September.

It REFUSES to ship if:
  0. this packer is not the exact committed Git blob;
  1. the patched trainer does not carry addendum M's fields (the bundle would silently run
     the old one and the owed fields would go missing for a third session);
  2. the resume artefact is not the file the PUBLISHED session-2 reading was taken from
     -- `h1_eval_h1_s2_cum390.json` records `trained` as a path, and that path must be the
     one being shipped.  Addendum N pins the exact bytes now present at that published path;
  3. the starting bundle differs byte-for-byte from session 2's manifest or RUN.md;
  4. the final manifest does not describe the final files after RUN.md is changed.
"""
import argparse, hashlib, io, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "_h1_bundle")
SRC_S2 = r"D:\_ktmp\h1_s2_final\h1_trained_s2.npz"
TRAINER = os.path.join(HERE, "h1_qat.py")

SEED3 = 3141
MAX_HOURS = 11.0
S2_BYTES = 1334711974
S2_SHA256 = "4030d2af63aed924d7e17559bc3dc84ba8db519056c389cdb068b38b58a7e3b3"
TRAINER_SHA256 = "7876d6032c3fd46f4d836d88aaf7ebae087f0437dc743f310f0a3a4652bd30ad"
BASE_MANIFEST_SHA256 = "a4843e04af47e9840f851045e2dd30b93033a00bb9c0aba788e99a77fa10dfb8"
BASE_RUN_SHA256 = "e8e3cb1df78c1df63efc20834a83c2b52dd3562f83b0c89b4096c163f6195e92"
S2_BPB = 0.962592908257438


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def refuse(msg):
    raise SystemExit("REFUSING TO SHIP: " + msg)


def git(*args):
    p = subprocess.run(["git"] + list(args), cwd=os.path.join(HERE, "..", "..", ".."),
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        refuse("git %s failed: %s" % (" ".join(args), p.stderr.strip()))
    return p.stdout.strip()


def committed_provenance():
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    rel = os.path.relpath(__file__, repo).replace(os.sep, "/")
    head = git("rev-parse", "HEAD")
    head_blob = git("rev-parse", "HEAD:" + rel)
    work_blob = git("hash-object", os.path.abspath(__file__))
    if work_blob != head_blob:
        refuse("packer is not the committed HEAD blob (worktree %s, HEAD %s)" %
               (work_blob, head_blob))
    return {"head_commit": head, "packer_git_blob": head_blob,
            "packer_sha256": sha256(__file__), "packer_path": rel}


def validate_starting_bundle():
    if not os.path.isdir(BUNDLE):
        refuse("no bundle dir at %s" % BUNDLE)

    mp = os.path.join(BUNDLE, "MANIFEST.json")
    rp = os.path.join(BUNDLE, "RUN.md")
    if sha256(mp) != BASE_MANIFEST_SHA256:
        refuse("starting MANIFEST.json is not the pinned session-2 manifest")
    if sha256(rp) != BASE_RUN_SHA256:
        refuse("starting RUN.md is not the pinned session-2 handover")
    man = json.load(io.open(mp, encoding="utf-8"))
    if man.get("stage") != "H1" or man.get("sessions") != 2:
        refuse("starting manifest is not the H1 two-session bundle")
    for name, meta in man.get("files", {}).items():
        path = os.path.join(BUNDLE, name)
        if not os.path.isfile(path):
            refuse("starting bundle is missing %s" % name)
        if os.path.getsize(path) != meta["bytes"] or sha256(path) != meta["sha256"]:
            refuse("starting bundle file %s differs from its session-2 manifest" % name)
    if not os.path.isfile(os.path.join(BUNDLE, "h1_trained_s1.npz")):
        refuse("starting bundle has no session-1 resume artefact")
    if os.path.exists(os.path.join(BUNDLE, "h1_trained_s2.npz")):
        refuse("starting bundle already contains session-2 artefact; refusing a partial rebuild")

    src = io.open(TRAINER, encoding="utf-8").read()
    for needle in ('"steps_completed": step', '"stop_reason": stop_reason',
                   '"seconds_per_step"', 'stop_reason = "time-cap"'):
        if needle not in src:
            refuse("h1_qat.py does not carry addendum M (%r missing)" % needle)
    if sha256(TRAINER) != TRAINER_SHA256:
        refuse("h1_qat.py is not addendum N's pinned trainer")

    if not os.path.isfile(SRC_S2):
        refuse("no s2 artefact at %s" % SRC_S2)
    if os.path.getsize(SRC_S2) != S2_BYTES or sha256(SRC_S2) != S2_SHA256:
        refuse("session-2 artefact differs from addendum N's pinned size/hash")

    evp = os.path.join(HERE, "results", "h1", "h1_eval_h1_s2_cum390.json")
    if not os.path.exists(evp):
        refuse("no published s2 eval at %s -- session 3 would resume from an unscored point"
               % evp)
    ev = json.load(io.open(evp, encoding="utf-8"))
    named = os.path.normcase(os.path.abspath(ev.get("trained", "")))
    if named != os.path.normcase(os.path.abspath(SRC_S2)):
        refuse("published s2 eval read %r, but this script ships %r" %
               (ev.get("trained"), SRC_S2))
    if abs(ev["bpb"]["trained-8L"] - S2_BPB) > 1e-15:
        refuse("published s2 BPB changed from the pinned %.15f" % S2_BPB)
    return man, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate every pinned input without changing the bundle")
    a = ap.parse_args()
    t0 = time.time()
    provenance = committed_provenance()
    man, ev = validate_starting_bundle()
    print("  packer: committed HEAD blob %s" % provenance["packer_git_blob"])
    print("  resume: %d B  %s  BPB %.15f" %
          (S2_BYTES, S2_SHA256, ev["bpb"]["trained-8L"]))
    print("  starting bundle: exact session-2 manifest and files")
    if a.check:
        print("CHECK ONLY: all pins passed; bundle untouched")
        return 0

    # ---- 1. the trainer must be addendum M's ---------------------------------------------
    shutil.copy2(TRAINER, os.path.join(BUNDLE, "h1_qat.py"))
    print("  trainer: addendum M fields present, copied")

    # ---- 2. swap the resume artefact ------------------------------------------------------
    print("  resume artefact: named by the published eval (BPB %.6f), %d B"
          % (ev["bpb"]["trained-8L"], os.path.getsize(SRC_S2)))
    dst_s2 = os.path.join(BUNDLE, "h1_trained_s2.npz")
    old_s1 = os.path.join(BUNDLE, "h1_trained_s1.npz")
    print("  copying %s (%.2f GB) ..." %
          (os.path.basename(SRC_S2), os.path.getsize(SRC_S2) / 1073741824.0))
    tc = time.time()
    shutil.copy2(SRC_S2, dst_s2)
    if os.path.getsize(dst_s2) != S2_BYTES or sha256(dst_s2) != S2_SHA256:
        refuse("copied session-2 artefact failed its exact size/hash check")
    print("    copied and byte-verified in %.0f s" % (time.time() - tc))
    if os.path.exists(old_s1):
        os.remove(old_s1)
        print("  removed h1_trained_s1.npz (session 3 resumes from s2, and the upload is "
              "1.33 GB lighter without it)")

    # ---- 3. write the command BEFORE hashing it ---------------------------------------------
    rp = os.path.join(BUNDLE, "RUN.md")
    run = io.open(rp, encoding="utf-8").read()
    run += u"""

---

## Session 3 (addendum M) -- ONE long session, not four short ones

`--max-hours 2.8` was a default in `h1_qat.py`, not a platform limit: Kaggle allows **12 h**.
Session 3 runs **11 h** -- the trainer's own cap must fire BEFORE the platform's, because the
1.33 GB `np.savez` happens only after the clean stop.

At the measured **51.7 s/step** that is **~766 steps in ONE session**, against ~390 across
both previous sessions, with **one new** Adam restart rather than four prospective short-run
restarts.

```
python3 h1_qat.py \\
    --factors  h0_trained3.npz \\
    --resume   h1_trained_s2.npz \\
    --labels   labels_E256.npz \\
    --stats    h1_actstats.npz \\
    --train    h1_train.npz \\
    --heldout  h1_heldout.npz \\
    --calib    h1_calib.npz \\
    --out      /kaggle/working/h1_trained_s3.npz \\
    --router-lr 0.0003 --aux 0.01 --steps 4000 --bs 2 --accum 8 --lr 2e-4 \\
    --every 250 --seed %d --max-hours %.1f
```

**What to watch, and it is new this session.** The periodic checkpoint at `--every 250` has
**never executed**: sessions 1 and 2 both stopped at step 195. The first `save()` at step 250
is therefore the first exercise of that path -- if it fails, the session is lost, so the step
250 log line is the one that matters most.

**Coming back:** `h1_trained_s3.npz` plus its `.json`. The gate is **not** decided on the T4 --
run `h1_eval.py` on CPU fp32, as for sessions 1 and 2.
""" % (SEED3, MAX_HOURS)
    io.open(rp, "w", encoding="utf-8").write(run)
    print("  RUN.md: session 3 command appended")

    # ---- 4. hash the FINAL files, including the already-updated RUN.md ----------------------
    files = sorted(f for f in os.listdir(BUNDLE)
                   if os.path.isfile(os.path.join(BUNDLE, f))
                   and f not in ("MANIFEST.json", "dataset-metadata.json"))
    hashes = {}
    for f in files:
        p = os.path.join(BUNDLE, f)
        hashes[f] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        print("  %-24s %12d B  %s" % (f, hashes[f]["bytes"], hashes[f]["sha256"][:16]))

    # ---- 5. manifest and post-write verification -------------------------------------------
    mp = os.path.join(BUNDLE, "MANIFEST.json")
    man["sessions"] = 3
    man["budget_gpu_hours"] = 5.6 + MAX_HOURS
    man["addendum"] = "M+N -- one 11 h session replaces four short ones and avoids three " \
                      "prospective Adam restarts; exact resume and bundle identities pinned"
    man["command_session3"] = {
        "resume": "h1_trained_s2.npz", "factors": "h0_trained3.npz",
        "out": "h1_trained_s3.npz", "steps": 4000, "bs": 2, "accum": 8,
        "lr": "2e-4", "router_lr": "0.0003", "aux": "0.01", "every": 250,
        "seed": SEED3, "max_hours": MAX_HOURS,
        "expected_steps_at_51.7_s_per_step": int(MAX_HOURS * 3600 / 51.7),
    }
    man["session3_provenance"] = provenance
    man["session3_resume_source"] = {
        "published_eval": "results/h1/h1_eval_h1_s2_cum390.json",
        "path": SRC_S2, "bytes": S2_BYTES, "sha256": S2_SHA256,
        "published_bpb": S2_BPB,
    }
    man["files"] = hashes
    man["repacked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(man, io.open(mp, "w", encoding="utf-8"), indent=1)
    for f, meta in hashes.items():
        p = os.path.join(BUNDLE, f)
        if os.path.getsize(p) != meta["bytes"] or sha256(p) != meta["sha256"]:
            refuse("final manifest verification failed for %s" % f)
    print("  MANIFEST.json rewritten and verified: %d final files, sessions=3" % len(hashes))

    total = sum(v["bytes"] for v in hashes.values())
    print("\nbundle ready: %d files, %.2f GB, in %.0f s"
          % (len(hashes), total / 1073741824.0, time.time() - t0))
    print("upload:  python scripts/kaggle_ops.py raw acct1 -- datasets version "
          "-p benchmarks/donor_adaptation/s1/_h1_bundle -m \"session 3: resume from s2, "
          "addendum M\" --dir-mode zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
