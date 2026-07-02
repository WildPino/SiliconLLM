"""
Phase 30: Header & Integrity Tests

Tests:
  1. bad_magic        — archive with wrong magic bytes is rejected at decode
  2. wrong_weights    — archive decoded with wrong weights file is rejected
  3. truncated_header — archive with truncated header is rejected
  4. profile_roundtrip — encode with each named profile, decode, verify SHA
  5. weights_absent   — decode with missing weights path is rejected

Exit 0 = all pass, 1 = any failure.
"""

import os
import sys
import struct
import shutil
import hashlib
import tempfile
import subprocess

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE     = os.path.join(ROOT, "see.exe")
WEIGHTS = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
SAMPLE  = os.path.join(ROOT, "data", "natural_text.txt")

PROFILES = [
    ("general",  ["--expert-profile", "general"]),
    ("prose",    ["--expert-profile", "prose"]),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(src, dest, extra=None):
    cmd = [SEE, "encode", src, dest, "--weights", WEIGHTS, "--blend", "moe",
           "--expert-profile", "general"] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode


def decode(src, dest, weights=None):
    w = weights or WEIGHTS
    cmd = [SEE, "decode", src, dest, "--weights", w]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode, r.stderr.decode("utf-8", errors="replace")


results = []

def run(name, fn):
    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, str(e)
    status = "PASS" if ok else "FAIL"
    results.append((name, ok, msg))
    print(f"  [{status}] {name}")
    if not ok:
        print(f"         {msg}")


tmpdir = tempfile.mkdtemp(prefix="see_hdr_")
try:
    good_enc = os.path.join(tmpdir, "good.see")
    assert encode(SAMPLE, good_enc) == 0, "baseline encode failed"

    # ── 1. bad magic ──────────────────────────────────────────────────────────
    def test_bad_magic():
        bad = os.path.join(tmpdir, "bad_magic.see")
        with open(good_enc, "rb") as f:
            data = bytearray(f.read())
        # Overwrite first 4 bytes (magic)
        struct.pack_into("<I", data, 0, 0xDEADBEEF)
        with open(bad, "wb") as f:
            f.write(data)
        rc, err = decode(bad, os.path.join(tmpdir, "bad_magic.dec"))
        if rc != 0 and "bad magic" in err.lower():
            return True, "correctly rejected with 'bad magic'"
        return False, f"expected rejection, got rc={rc}, stderr={err[:80]}"

    run("bad_magic", test_bad_magic)

    # ── 2. wrong weights ──────────────────────────────────────────────────────
    def test_wrong_weights():
        fake_w = os.path.join(tmpdir, "fake.bin")
        with open(WEIGHTS, "rb") as f:
            data = bytearray(f.read())
        data[8] ^= 0xFF  # corrupt one byte inside the weights header body
        with open(fake_w, "wb") as f:
            f.write(data)
        rc, err = decode(good_enc, os.path.join(tmpdir, "wrong_w.dec"), weights=fake_w)
        # Either weights mismatch OR invalid weights magic (the byte we flipped may break parsing)
        if rc != 0:
            return True, f"correctly rejected: {err.strip()[:80]}"
        return False, f"expected rejection, got rc=0"

    run("wrong_weights", test_wrong_weights)

    # ── 3. truncated header ───────────────────────────────────────────────────
    def test_truncated_header():
        trunc = os.path.join(tmpdir, "truncated.see")
        with open(good_enc, "rb") as f:
            data = f.read()
        # Write only first 20 bytes (header is 92 bytes)
        with open(trunc, "wb") as f:
            f.write(data[:20])
        rc, err = decode(trunc, os.path.join(tmpdir, "trunc.dec"))
        if rc != 0:
            return True, f"correctly rejected: {err.strip()[:80]}"
        return False, f"expected rejection, got rc=0"

    run("truncated_header", test_truncated_header)

    # ── 4. profile roundtrip ──────────────────────────────────────────────────
    sha_orig = sha256_file(SAMPLE)

    def test_profile_rt(profile_name, profile_args):
        enc = os.path.join(tmpdir, f"{profile_name}.see")
        dec = os.path.join(tmpdir, f"{profile_name}.dec")
        cmd = [SEE, "encode", SAMPLE, enc, "--weights", WEIGHTS, "--blend", "moe"] + profile_args
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0:
            return False, f"encode failed: {r.stderr.decode()[:80]}"
        rc, err = decode(enc, dec)
        if rc != 0:
            return False, f"decode failed: {err[:80]}"
        sha_dec = sha256_file(dec)
        if sha_dec == sha_orig:
            return True, f"SHA match  ({os.path.getsize(enc)//1024}KB)"
        return False, f"SHA mismatch\n  orig={sha_orig}\n  dec ={sha_dec}"

    for pname, pargs in PROFILES:
        run(f"profile_roundtrip:{pname}", lambda p=pname, a=pargs: test_profile_rt(p, a))

    # ── 5. missing weights ────────────────────────────────────────────────────
    def test_missing_weights():
        rc, err = decode(good_enc, os.path.join(tmpdir, "nomatch.dec"),
                         weights="/nonexistent/weights.bin")
        if rc != 0:
            return True, f"correctly rejected: {err.strip()[:80]}"
        return False, "expected rejection, got rc=0"

    run("missing_weights", test_missing_weights)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULT: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
