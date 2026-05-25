"""
Phase 30: Regression Harness

Reads data/baselines/phase29a_baseline.json (internal, primary) and
data/baselines/phase29c_baseline.json (external, stress test) and verifies
that the current see.exe produces BPB within tolerance of each frozen value.

Also runs encode→decode SHA roundtrip on 3 representative corpora.

Exit code 0 = all pass.
Exit code 1 = one or more failures (details printed to stdout).

Usage:
    python scripts/regression_test.py
    python scripts/regression_test.py --baseline-only    (skip roundtrip)
    python scripts/regression_test.py --roundtrip-only   (skip BPB check)
"""

import os
import re
import sys
import json
import hashlib
import argparse
import subprocess
import tempfile
import shutil

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
DATA    = os.path.join(ROOT, "data")
BASEDIR = os.path.join(DATA, "baselines")
EXT_DIR = os.path.join(DATA, "external")

BASELINE_PRIMARY  = os.path.join(BASEDIR, "phase29a_baseline.json")
BASELINE_EXTERNAL = os.path.join(BASEDIR, "phase29c_baseline.json")

# Map baseline corpus keys (short names) to actual filenames in DATA
CORPUS_FILENAME_MAP = {
    "natural_text":  "natural_text.txt",
    "markdown_docs": "markdown_docs.md",
    "c_code":        "c_code.c",
    "shuffled":      "shuffled.bin",
    "json_synth":    "json_synth.json",
    "c_header":      "c_header_synth.h",
    "notes_it":      "project_notes_it.txt",
    "md_mixed":      "repo_markdown_mixed.md",
    "log_synth":     "log_synth.log",
}

ROUNDTRIP_FILES = [
    (os.path.join(DATA, "natural_text.txt"),    "general"),
    (os.path.join(DATA, "c_code.c"),            "general"),
    (os.path.join(EXT_DIR, "prose_real.txt"),   "prose"),
]

PROFILE_ARGS = {
    "general":  ["--expert-profile", "general"],
    "prose":    ["--expert-profile", "prose"],
    "span-pfx": ["--expert-profile", "general", "--span-pfx"],
    "no-token": ["--expert-profile", "experimental", "--lz-key", "6"],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_audit(fpath, profile):
    args = PROFILE_ARGS.get(profile, ["--expert-profile", profile])
    cmd  = [SEE, "audit", fpath, "--weights", WEIGHTS, "--blend", "moe"] + args
    res  = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out  = res.stdout + res.stderr
    m = re.search(r"Quantized BPB:\s+([\d.]+)", out)
    return float(m.group(1)) if m else None


def check_baselines(baseline_path, label, tolerance):
    if not os.path.exists(baseline_path):
        print(f"  [SKIP] {label}: baseline not found at {baseline_path}")
        return 0, 0

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    corpora = baseline.get("corpora", {})
    tol = baseline.get("tolerance_bpb", tolerance)

    passes = failures = 0
    for corpus_name, corpus_data in corpora.items():
        # Resolve file path (try mapped name, then raw name, then external dir)
        mapped = CORPUS_FILENAME_MAP.get(corpus_name, corpus_name)
        candidate_paths = [
            os.path.join(DATA, mapped),
            os.path.join(DATA, corpus_name),
            os.path.join(EXT_DIR, mapped),
            os.path.join(EXT_DIR, corpus_name),
        ]
        fpath = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not fpath:
            print(f"  [SKIP] {corpus_name}: file not found (external corpus — run to re-download)")
            continue

        # Filter only profile keys (not domain/size/etc metadata)
        profiles = {k: v for k, v in corpus_data.items()
                    if isinstance(v, dict) and "quant_bpb" in v}

        for profile, ref in profiles.items():
            ref_bpb = ref.get("quant_bpb")
            if ref_bpb is None:
                continue

            print(f"  {corpus_name:<28} [{profile:<9}]", end="", flush=True)
            cur_bpb = run_audit(fpath, profile)

            if cur_bpb is None:
                print(f"  ERROR (audit failed)")
                failures += 1
                continue

            delta = cur_bpb - ref_bpb
            status = "PASS" if abs(delta) <= tol else ("REGR" if delta > 0 else "IMPR")
            print(f"  base={ref_bpb:.4f}  cur={cur_bpb:.4f}  d={delta:+.4f}  [{status}]")

            if status == "REGR":
                failures += 1
            else:
                passes += 1

    return passes, failures


def check_roundtrips():
    passes = failures = 0
    tmpdir = tempfile.mkdtemp(prefix="see_rt_")
    try:
        for fpath, profile in ROUNDTRIP_FILES:
            if not os.path.exists(fpath):
                print(f"  [SKIP] {os.path.basename(fpath)}: not found")
                continue

            name     = os.path.basename(fpath)
            enc_path = os.path.join(tmpdir, name + ".see")
            dec_path = os.path.join(tmpdir, name + ".dec")

            args = PROFILE_ARGS.get(profile, ["--expert-profile", profile])
            print(f"  {name:<28} [{profile:<9}]", end="", flush=True)

            enc = subprocess.run(
                [SEE, "encode", fpath, enc_path, "--weights", WEIGHTS, "--blend", "moe"] + args,
                capture_output=True, text=True, timeout=300)
            if enc.returncode != 0:
                print(f"  ENCODE FAILED")
                print(f"    {enc.stderr.strip()[:120]}")
                failures += 1
                continue

            dec = subprocess.run(
                [SEE, "decode", enc_path, dec_path, "--weights", WEIGHTS],
                capture_output=True, text=True, timeout=300)
            if dec.returncode != 0:
                print(f"  DECODE FAILED")
                print(f"    {dec.stderr.strip()[:120]}")
                failures += 1
                continue

            sha_orig = sha256_file(fpath)
            sha_dec  = sha256_file(dec_path)
            if sha_orig == sha_dec:
                sz = os.path.getsize(enc_path)
                print(f"  SHA OK  ({sz//1024}KB encoded)")
                passes += 1
            else:
                print(f"  SHA MISMATCH")
                print(f"    original: {sha_orig}")
                print(f"    decoded:  {sha_dec}")
                failures += 1

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return passes, failures


def main():
    parser = argparse.ArgumentParser(description="SEE regression harness")
    parser.add_argument("--baseline-only",  action="store_true")
    parser.add_argument("--roundtrip-only", action="store_true")
    parser.add_argument("--tolerance",      type=float, default=0.005)
    args = parser.parse_args()

    total_pass = total_fail = 0
    sep = "=" * 75

    if not args.roundtrip_only:
        print(sep)
        print("BPB REGRESSION — Phase 29A baseline (internal, primary)")
        print(sep)
        p, f = check_baselines(BASELINE_PRIMARY, "29A", args.tolerance)
        total_pass += p; total_fail += f

        print()
        print(sep)
        print("BPB REGRESSION — Phase 29C baseline (external, stress test)")
        print(sep)
        p, f = check_baselines(BASELINE_EXTERNAL, "29C", args.tolerance)
        total_pass += p; total_fail += f

    if not args.baseline_only:
        print()
        print(sep)
        print("ENCODE/DECODE ROUNDTRIP — SHA-256 integrity")
        print(sep)
        p, f = check_roundtrips()
        total_pass += p; total_fail += f

    print()
    print(sep)
    print(f"RESULT: {total_pass} passed, {total_fail} failed")
    print(sep)

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
