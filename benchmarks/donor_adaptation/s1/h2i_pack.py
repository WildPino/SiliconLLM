#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the hash-closed H2I Phase B Kaggle bundle.

Run --check first.  The builder refuses unless it is the exact committed HEAD blob, validates
all scientific source identities, writes RUN.md before hashing, then copies and re-hashes every
payload.  An existing destination is never overwritten.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
H1_BUNDLE = os.path.join(HERE, "_h1_bundle")
DEST = os.path.join(HERE, "_h2i_bundle")

SCIENTIFIC = {
    "h0_trained3.npz": (os.path.join(H1_BUNDLE, "h0_trained3.npz"), 352967686,
                         "dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8"),
    "labels_E256.npz": (os.path.join(H1_BUNDLE, "labels_E256.npz"), 859537,
                         "c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c"),
    "h1_actstats.npz": (os.path.join(H1_BUNDLE, "h1_actstats.npz"), 1189630,
                         "49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe"),
    "e37_routers_E256.npz": ("D:/_ktmp/e37/e37_routers_E256.npz", 44046858,
                              "42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a"),
    "h2i_applied_8L.json": (os.path.join(HERE, "results", "h2i", "h2i_applied_8L.json"),
                             12239,
                             "c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc"),
    "h1_train.npz": (os.path.join(H1_BUNDLE, "h1_train.npz"), 64000260,
                       "0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9"),
    "h1_heldout.npz": (os.path.join(H1_BUNDLE, "h1_heldout.npz"), 98564,
                         "110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35"),
    "h1_calib.npz": (os.path.join(H1_BUNDLE, "h1_calib.npz"), 131332,
                       "b8d4184db0988f4d86ae3089e99ba167b8e05e6dcdbbc504f7679729fe7aa019"),
}

APPARATUS = {
    "h2i_qat.py": os.path.join(HERE, "h2i_qat.py"),
    "h2i_applied.py": os.path.join(HERE, "h2i_applied.py"),
    "h1_qat.py": os.path.join(HERE, "h1_qat.py"),
    "common.py": os.path.join(HERE, "..", "density", "common.py"),
    "t2_rules.py": os.path.join(HERE, "..", "ternary", "t2_rules.py"),
    "e6_generate.py": os.path.join(HERE, "..", "engine", "e6_generate.py"),
}

RUN_TEXT = r"""# H2I Phase B -- one continuous T4 session

Scientific protocol: `BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md`, addendum B. This bundle is
weight-only R8. Do not enable activation int8/LUT, change k/layers, resume H1, or choose a
checkpoint by its progress BPB.

Upload this directory as one Kaggle dataset. Use a T4 GPU, copy the dataset contents into the
working directory if Kaggle mounted them read-only, then run from the directory containing
`MANIFEST.json`:

```bash
python3 h2i_qat.py \
  --manifest MANIFEST.json \
  --factors h0_trained3.npz \
  --labels labels_E256.npz \
  --stats h1_actstats.npz \
  --routers e37_routers_E256.npz \
  --phase-a h2i_applied_8L.json \
  --train h1_train.npz \
  --heldout h1_heldout.npz \
  --calib h1_calib.npz \
  --out /kaggle/working/h2i_trained_s1.npz \
  --steps 4000 --bs 2 --accum 8 \
  --lr 2e-4 --router-lr 3e-4 --aux 0.01 \
  --every 250 --seed 4242 --max-hours 11.0
```

Expected startup:

- manifest and all input hashes pass before model loading;
- R8 parity, STE-gradient, hard/soft k=E, real k16=560 and live-carve controls fire;
- E37 routers are installed exactly;
- first applied update moves `L03.gate`, `L24.down`, and `L03.router` within 40 attempts.

Stop and return the log if any control refuses, any non-finite microbatch appears, or no update
is applied within 40 attempts. Otherwise let the trainer stop itself at 11 hours. Return:

- `h2i_trained_s1.npz` and `h2i_trained_s1.json`;
- `h2i_trained_s1.checkpoint.npz` and `.json` if present;
- the complete stdout/stderr log.

The GPU BPBs are progress diagnostics. Do not select the best checkpoint. The final artifact,
whatever its curve says, is adjudicated once by `h2i_eval.py` on CPU fp32.
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def committed_self():
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        rel = os.path.relpath(os.path.abspath(__file__), root).replace("\\", "/")
        work = subprocess.check_output(
            ["git", "hash-object", os.path.abspath(__file__)], cwd=root, text=True
        ).strip()
        blob = subprocess.check_output(
            ["git", "rev-parse", "HEAD:" + rel], cwd=root, text=True,
            stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("cannot establish committed packer provenance: %s" % exc)
    if work != blob:
        raise SystemExit("PACKER IS NOT THE COMMITTED HEAD BLOB: %s != %s -- STOP"
                         % (work, blob))
    return {"head_commit": head, "packer_path": rel, "packer_git_blob": blob,
            "packer_sha256": sha256_file(os.path.abspath(__file__))}


def validate_sources():
    records = {}
    for name, (path, expected_size, expected_sha) in SCIENTIFIC.items():
        if not os.path.isfile(path):
            raise SystemExit("missing H2I scientific source %s: %s" % (name, path))
        size, digest = os.path.getsize(path), sha256_file(path)
        if size != expected_size or digest != expected_sha:
            raise SystemExit("H2I source mismatch %s: %d %s, expected %d %s"
                             % (name, size, digest, expected_size, expected_sha))
        records[name] = {"source": os.path.abspath(path), "bytes": size, "sha256": digest,
                         "role": "scientific_input"}
    for name, path in APPARATUS.items():
        if not os.path.isfile(path):
            raise SystemExit("missing H2I apparatus source %s: %s" % (name, path))
        records[name] = {"source": os.path.abspath(path), "bytes": os.path.getsize(path),
                         "sha256": sha256_file(path), "role": "apparatus"}
    return records


def verify_bundle(dest, manifest):
    files = manifest.get("files", {})
    expected_names = set(SCIENTIFIC) | set(APPARATUS) | {"RUN.md"}
    if set(files) != expected_names:
        raise SystemExit("bundle manifest names differ: got %s expected %s"
                         % (sorted(files), sorted(expected_names)))
    for name, expected in files.items():
        path = os.path.join(dest, name)
        if not os.path.isfile(path):
            raise SystemExit("bundle missing %s" % path)
        size, digest = os.path.getsize(path), sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise SystemExit("bundle verification failed for %s" % name)
    return True


def build(dest, provenance, sources):
    if os.path.exists(dest):
        raise SystemExit("destination already exists; refusing overwrite: %s" % dest)
    os.makedirs(dest)
    for name, record in sources.items():
        shutil.copyfile(record["source"], os.path.join(dest, name))
    run_path = os.path.join(dest, "RUN.md")
    with open(run_path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(RUN_TEXT)

    files = {}
    for name in list(SCIENTIFIC) + list(APPARATUS) + ["RUN.md"]:
        path = os.path.join(dest, name)
        files[name] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path),
                       "role": (sources[name]["role"] if name in sources else "instructions")}
    manifest = {
        "brief": "BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md addendum B",
        "stage": "H2I_PHASE_B", "status": "READY_NOT_RUN",
        "model": "Qwen/Qwen2.5-1.5B",
        "revision": "8faed761d45a263340a0528343f099c05c9a4323",
        "layers": [3, 6, 9, 12, 15, 18, 21, 24], "E": 256, "k": 16,
        "command": {"steps": 4000, "bs": 2, "accum": 8, "lr": 0.0002,
                    "router_lr": 0.0003, "aux": 0.01, "every": 250,
                    "seed": 4242, "max_hours": 11.0, "resume": None},
        "gates": {"score_bpb_lt": 1.0190764652622473,
                  "rank_free_gt": 9, "rank_teacher_forced_gt": 99,
                  "rank_mean_lt": 3.675},
        "router_init": {"source": "E37", "sha256": SCIENTIFIC["e37_routers_E256.npz"][2]},
        "packer_provenance": provenance, "files": files,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = os.path.join(dest, "MANIFEST.json")
    with open(manifest_path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    with open(manifest_path, encoding="utf-8") as fh:
        reread = json.load(fh)
    verify_bundle(dest, reread)
    return manifest_path, reread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dest", default=DEST)
    args = parser.parse_args()
    provenance = committed_self()
    sources = validate_sources()
    print("H2I source check PASS: %d payloads; packer HEAD %s"
          % (len(sources), provenance["head_commit"]), flush=True)
    if args.check:
        if os.path.isdir(args.dest):
            manifest_path = os.path.join(args.dest, "MANIFEST.json")
            if not os.path.isfile(manifest_path):
                raise SystemExit("existing bundle has no MANIFEST.json: %s" % args.dest)
            with open(manifest_path, encoding="utf-8") as fh:
                verify_bundle(args.dest, json.load(fh))
            print("existing bundle verification PASS: %s" % args.dest, flush=True)
        else:
            print("destination absent; ready to build once: %s" % args.dest, flush=True)
        return 0
    manifest_path, manifest = build(args.dest, provenance, sources)
    total = sum(item["bytes"] for item in manifest["files"].values())
    print("H2I bundle BUILT_AND_VERIFIED: %s" % args.dest, flush=True)
    print("payload bytes %d; manifest %s sha256 %s" %
          (total, manifest_path, sha256_file(manifest_path)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
