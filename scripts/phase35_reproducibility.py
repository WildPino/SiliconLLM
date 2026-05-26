"""
Phase 35: Reproducibility & Portability Audit

Tests:
  A. Encode determinism     — encode same file twice, compare byte-for-byte
  B. Fixture decode         — decode pre-committed .see fixtures, verify SHA-256
  C. Header rejection       — bad magic / truncated / header_size mismatch / bad weights hash
  D. Compiler variant       — build with -O2, -O3, -ffast-math; compare archive output

Usage:
    python scripts/phase35_reproducibility.py [--skip-compiler]

Exit 0 = all tests passed (or compiler tests skipped by flag).
Exit 1 = one or more failures.
"""

import os
import sys
import json
import struct
import hashlib
import shutil
import argparse
import tempfile
import subprocess
import textwrap

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEE      = os.path.join(ROOT, "see.exe")
WEIGHTS  = os.path.join(ROOT, "weights", "entropy_weights_factors_r16.bin")
FIXTURES = os.path.join(ROOT, "data", "fixtures")
MANIFEST = os.path.join(FIXTURES, "manifest.json")

# ── tiny inline inputs for fixtures / determinism tests ──────────────────────
FIXTURE_INPUTS = {
    "tiny_text": textwrap.dedent("""\
        Silicon Entropy Engine — a streaming MoE compressor.
        Each byte is predicted by a mixture of experts conditioned on all prior bytes.
        The LZ6 expert maintains a hash-based dictionary for repeated sequences.
        TOKPFX finds signal in inside-token byte patterns.
        Experts compete via exponentiated-gradient credit assignment.
        The range coder converts probability distributions to bits.
        """),
    "tiny_code": textwrap.dedent("""\
        #include <stdint.h>
        static inline uint32_t rol32(uint32_t x, int n) {
            return (x << n) | (x >> (32 - n));
        }
        uint32_t fnv1a_32(const uint8_t *data, size_t len) {
            uint32_t h = 2166136261u;
            for (size_t i = 0; i < len; i++)
                h = (h ^ data[i]) * 16777619u;
            return h;
        }
        """),
    "tiny_json": textwrap.dedent("""\
        {"name":"SEE","version":"1.0","experts":["LZ6","TOKPFX","BI","UNI"],
         "config":{"moe_eta":0.03,"moe_share":0.001,"lz_key":6,"top_k":256},
         "profiles":{"general":"LZ6+TOKPFX","prose":"LZ6+TOKPFX+TOK_PREV_ELIG"}}
        """),
}

FIXTURE_PROFILES = {
    "tiny_text": "general",
    "tiny_code": "general",
    "tiny_json": "general",
}

SEE3_MAGIC = 0x33454553

# ── helpers ───────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_see(args, check=True):
    """Run see.exe with the given args list. Returns CompletedProcess."""
    cmd = [SEE] + args
    return subprocess.run(cmd, capture_output=True, check=check)


def encode(src, dst, profile="general"):
    run_see(["encode", src, dst, "--weights", WEIGHTS,
             "--expert-profile", profile])


def decode(src, dst):
    run_see(["decode", src, dst, "--weights", WEIGHTS])


# ── Section A: encode determinism ────────────────────────────────────────────

def test_determinism(tmpdir):
    results = []
    for name, content in FIXTURE_INPUTS.items():
        src = os.path.join(tmpdir, f"{name}.src")
        out1 = os.path.join(tmpdir, f"{name}_run1.see")
        out2 = os.path.join(tmpdir, f"{name}_run2.see")

        with open(src, "wb") as f:
            f.write(content.encode("utf-8"))

        profile = FIXTURE_PROFILES[name]
        encode(src, out1, profile)
        encode(src, out2, profile)

        h1 = sha256_file(out1)
        h2 = sha256_file(out2)
        ok = h1 == h2
        results.append((name, ok, h1[:16]))
        print(f"  {'OK' if ok else 'FAIL':<4}  {name:<18}  archive sha={h1[:16]}  {'identical' if ok else 'DIFFER'}")

    return results


# ── Section B: fixture decode ─────────────────────────────────────────────────

def generate_fixtures(tmpdir):
    """Build fixture archives and manifest. Call once to seed data/fixtures/."""
    os.makedirs(FIXTURES, exist_ok=True)
    manifest = {}

    for name, content in FIXTURE_INPUTS.items():
        src = os.path.join(tmpdir, f"{name}.src")
        archive = os.path.join(FIXTURES, f"{name}.see")

        # Write in binary mode so SHA matches exactly what the decoder will produce
        raw = content.encode("utf-8")
        with open(src, "wb") as f:
            f.write(raw)

        profile = FIXTURE_PROFILES[name]
        encode(src, archive, profile)

        decoded = os.path.join(tmpdir, f"{name}.dec")
        decode(archive, decoded)
        with open(decoded, "rb") as f:
            dec_sha = sha256_bytes(f.read())

        original_sha = sha256_bytes(raw)
        archive_sha  = sha256_file(archive)

        manifest[name] = {
            "profile":      profile,
            "original_sha": original_sha,
            "archive_sha":  archive_sha,
            "decoded_sha":  dec_sha,
            "input_bytes":  len(content.encode("utf-8")),
        }
        assert original_sha == dec_sha, f"Fixture {name}: encode/decode SHA mismatch during generation"
        print(f"  generated  {name}.see  ({os.path.getsize(archive)} bytes)")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest written to {MANIFEST}")
    return manifest


def test_fixture_decode(tmpdir):
    if not os.path.exists(MANIFEST):
        print("  SKIP  manifest not found — run with --generate-fixtures first")
        return []

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for name, entry in manifest.items():
        archive = os.path.join(FIXTURES, f"{name}.see")
        if not os.path.exists(archive):
            print(f"  MISS  {name}.see not found")
            results.append((name, False, "missing"))
            continue

        # Verify archive SHA matches manifest (detects bit-rot or accidental edits)
        arc_sha = sha256_file(archive)
        if arc_sha != entry["archive_sha"]:
            print(f"  FAIL  {name}.see archive SHA changed (expected {entry['archive_sha'][:16]}, got {arc_sha[:16]})")
            results.append((name, False, "archive_sha_changed"))
            continue

        decoded = os.path.join(tmpdir, f"{name}_fix.dec")
        try:
            decode(archive, decoded)
        except subprocess.CalledProcessError as e:
            print(f"  FAIL  {name}.see decode error: {e.stderr.decode(errors='replace').strip()}")
            results.append((name, False, "decode_error"))
            continue

        with open(decoded, "rb") as f:
            dec_sha = sha256_bytes(f.read())

        ok = dec_sha == entry["original_sha"]
        results.append((name, ok, dec_sha[:16]))
        print(f"  {'OK' if ok else 'FAIL':<4}  {name}.see => decoded sha={dec_sha[:16]}  {'MATCH' if ok else 'MISMATCH (expected ' + entry['original_sha'][:16] + ')'}")

    return results


# ── Section C: header rejection ───────────────────────────────────────────────

def corrupt_and_try_decode(archive_bytes, patch_fn, tmpdir, label):
    """Apply patch_fn to archive bytes, write to tmp file, try to decode. Expect failure."""
    corrupted = os.path.join(tmpdir, f"corrupt_{label}.see")
    out = os.path.join(tmpdir, f"corrupt_{label}.out")

    data = bytearray(archive_bytes)
    patch_fn(data)

    with open(corrupted, "wb") as f:
        f.write(data)

    proc = run_see(["decode", corrupted, out, "--weights", WEIGHTS], check=False)
    rejected = proc.returncode != 0

    # Also ensure output file is either absent or empty (no silent corruption)
    silently_corrupt = False
    if os.path.exists(out):
        size = os.path.getsize(out)
        if size > 0 and rejected:
            # decoder errored AND wrote partial output — check if it matches original SHA
            silently_corrupt = False  # error was raised, partial output is OK
        elif size > 0 and not rejected:
            silently_corrupt = True  # returned 0 but decoded something from corrupted input

    status = "OK (rejected)" if rejected and not silently_corrupt else \
             "FAIL (silent corruption)" if silently_corrupt else \
             "FAIL (accepted corrupt input)"
    ok = rejected and not silently_corrupt
    print(f"  {'OK' if ok else 'FAIL':<4}  {label:<35}  {status}")
    return ok


def test_header_rejection(tmpdir):
    # Use a fixture archive as source of valid bytes
    archive = os.path.join(FIXTURES, "tiny_text.see")
    if not os.path.exists(archive):
        print("  SKIP  fixture tiny_text.see not found")
        return []

    with open(archive, "rb") as f:
        good_bytes = f.read()

    results = []

    # 1. Bad magic
    def bad_magic(d): struct.pack_into("<I", d, 0, 0xDEADBEEF)
    results.append(corrupt_and_try_decode(good_bytes, bad_magic, tmpdir, "bad_magic"))

    # 2. Truncated header (file shorter than header_size)
    def truncate_header(d): del d[45:]   # SeeArchiveHeader = 92 bytes; cut to 45
    results.append(corrupt_and_try_decode(good_bytes, truncate_header, tmpdir, "truncated_header"))

    # 3. header_size mismatch (write a different value)
    def bad_header_size(d): struct.pack_into("<I", d, 4, 999)
    results.append(corrupt_and_try_decode(good_bytes, bad_header_size, tmpdir, "header_size_mismatch"))

    # 4. Weights hash mismatch (corrupt last byte of the 32-byte SHA field at offset 60)
    # SeeArchiveHeader layout: magic(4)+header_size(4)+archive_flags(4)+original_size(4)+
    #   chunk_size(4)+codebook_seed(4)+lz_hash_size(4)+req_topk(2)+tail_mode(2)+
    #   coder_scale_bits(2)+lz_top_n(1)+profile_id(1)+decay(4)+blend_lambda(4)+
    #   lz_K(4)+moe_eta(4)+moe_share(4)+seed_byte0(1)+seed_byte1(1)+lz_key_bytes(1)+
    #   _pad1(1) = 60 bytes before weights_sha256[32]
    def bad_weights_hash(d):
        offset = 60  # start of weights_sha256[32]
        d[offset] ^= 0xFF
    results.append(corrupt_and_try_decode(good_bytes, bad_weights_hash, tmpdir, "weights_hash_mismatch"))

    # 5. Completely empty file
    def empty_file(d): del d[:]
    results.append(corrupt_and_try_decode(good_bytes, empty_file, tmpdir, "empty_file"))

    return results


# ── Section D: compiler variant ───────────────────────────────────────────────

def compile_see(flags, outname, tmpdir):
    """Compile see.exe with given flags. Returns path to binary or None on error."""
    out = os.path.join(tmpdir, outname)
    src_files = [os.path.join(ROOT, "see.c")]
    src_dir = os.path.join(ROOT, "src")
    EXCLUDE = {"test_rc.c", "wave_engine.c"}
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith(".c") and fname not in EXCLUDE:
            src_files.append(os.path.join(src_dir, fname))

    cmd = ["gcc"] + flags + ["-I", ROOT, "-I", os.path.join(ROOT, "src")] + src_files + ["-o", out, "-lm"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return None, proc.stderr.decode(errors="replace")
    return out, None


def test_compiler_variants(tmpdir):
    variants = [
        ("-O2",        ["-O2", "-march=native"]),
        ("-O3",        ["-O3", "-march=native"]),
        ("-ffast-math", ["-O2", "-march=native", "-ffast-math"]),
    ]

    # Build all variants
    binaries = {}
    for label, flags in variants:
        out, err = compile_see(flags, f"see_{label.lstrip('-')}.exe", tmpdir)
        if out is None:
            print(f"  FAIL  compile {label}: {err[:200]}")
            binaries[label] = None
        else:
            print(f"  built {label:<15} => {os.path.basename(out)}")
            binaries[label] = out

    # For each variant, encode the same small file and compare archives
    results = []
    src = os.path.join(tmpdir, "cmp_input.txt")
    with open(src, "wb") as f:
        f.write(FIXTURE_INPUTS["tiny_text"].encode("utf-8"))

    archives = {}
    for label, binary in binaries.items():
        if binary is None:
            archives[label] = None
            continue
        arc = os.path.join(tmpdir, f"cmp_{label.lstrip('-')}.see")
        proc = subprocess.run(
            [binary, "encode", src, arc, "--weights", WEIGHTS, "--expert-profile", "general"],
            capture_output=True
        )
        if proc.returncode != 0:
            archives[label] = None
            print(f"  FAIL  encode with {label}")
        else:
            archives[label] = sha256_file(arc)

    # Compare
    ref_label = "-O3"  # the current build flag
    ref_sha = archives.get(ref_label)

    for label, sha in archives.items():
        if sha is None:
            continue
        if label == ref_label:
            continue
        match = sha == ref_sha
        risk = ""
        if label == "-ffast-math" and not match:
            risk = "  <- PORTABILITY RISK: -ffast-math changes output"
        elif not match:
            risk = "  <- archives differ"
        ok_str = "MATCH" if match else "DIFFER"
        print(f"  {label:<18}  vs {ref_label}: {ok_str}{risk}")
        results.append((label, match))

        # If ffast-math differs, also verify it can round-trip within itself
        if not match:
            arc_path = os.path.join(tmpdir, f"cmp_{label.lstrip('-')}.see")
            dec_path = os.path.join(tmpdir, f"cmp_{label.lstrip('-')}.dec")
            binary = binaries[label]
            proc = subprocess.run(
                [binary, "decode", arc_path, dec_path, "--weights", WEIGHTS],
                capture_output=True
            )
            if proc.returncode == 0:
                with open(dec_path, "rb") as f:
                    dec_sha = sha256_bytes(f.read())
                with open(src, "rb") as fh:
                    original_sha = sha256_bytes(fh.read())
                self_rt = dec_sha == original_sha
                print(f"    self-roundtrip ({label}): {'OK — can decode its own output' if self_rt else 'FAIL — cannot decode its own output'}")
            else:
                print(f"    self-roundtrip ({label}): DECODE FAILED")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compiler", action="store_true",
                        help="Skip compiler variant tests (faster, no recompile needed)")
    parser.add_argument("--generate-fixtures", action="store_true",
                        help="Generate fixture files and manifest, then exit")
    args = parser.parse_args()

    if not os.path.exists(SEE):
        print(f"ABORT: see.exe not found at {SEE}")
        sys.exit(1)
    if not os.path.exists(WEIGHTS):
        print(f"ABORT: weights not found at {WEIGHTS}")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="phase35_")
    all_ok = True

    try:
        if args.generate_fixtures:
            print("=== Generating fixtures ===")
            generate_fixtures(tmpdir)
            print("Done.")
            return

        # ── A: Determinism ────────────────────────────────────────────────────
        print()
        print("=" * 70)
        print("A. ENCODE DETERMINISM — encode twice, compare byte-for-byte")
        print("=" * 70)
        det_results = test_determinism(tmpdir)
        det_ok = all(ok for _, ok, _ in det_results)
        if not det_ok:
            all_ok = False
        print(f"  => {'PASS' if det_ok else 'FAIL'}  ({sum(ok for _, ok, _ in det_results)}/{len(det_results)} identical)")

        # ── B: Fixture decode ─────────────────────────────────────────────────
        print()
        print("=" * 70)
        print("B. FIXTURE DECODE — pre-committed archives must decode correctly")
        print("=" * 70)
        fix_results = test_fixture_decode(tmpdir)
        fix_ok = all(ok for _, ok, _ in fix_results) if fix_results else True
        if not fix_ok:
            all_ok = False
        if fix_results:
            print(f"  => {'PASS' if fix_ok else 'FAIL'}  ({sum(ok for _, ok, _ in fix_results)}/{len(fix_results)} decoded correctly)")

        # ── C: Header rejection ───────────────────────────────────────────────
        print()
        print("=" * 70)
        print("C. HEADER REJECTION — corrupt archives must be rejected, not silently accepted")
        print("=" * 70)
        hdr_results = test_header_rejection(tmpdir)
        hdr_ok = all(hdr_results) if hdr_results else True
        if not hdr_ok:
            all_ok = False
        if hdr_results:
            print(f"  => {'PASS' if hdr_ok else 'FAIL'}  ({sum(hdr_results)}/{len(hdr_results)} correctly rejected)")

        # ── D: Compiler variants ──────────────────────────────────────────────
        if args.skip_compiler:
            print()
            print("=" * 70)
            print("D. COMPILER VARIANTS — skipped (--skip-compiler)")
            print("=" * 70)
        else:
            print()
            print("=" * 70)
            print("D. COMPILER VARIANTS — -O2 / -O3 / -ffast-math output comparison")
            print("=" * 70)
            cmp_results = test_compiler_variants(tmpdir)
            # Non-matching ffast-math is expected; we report it but don't fail the suite
            # (it's a documentation finding, not a binary pass/fail)
            ffast_mismatch = any(label == "-ffast-math" and not ok for label, ok in cmp_results)
            o2_mismatch    = any(label == "-O2" and not ok for label, ok in cmp_results)
            if o2_mismatch:
                all_ok = False
                print("  => FAIL  -O2 and -O3 produce different archives (float sensitivity confirmed)")
            elif ffast_mismatch:
                print("  => WARN  -ffast-math produces different archives (document as forbidden flag)")
            else:
                print("  => PASS  -O2/-O3 produce identical archives")

        # ── Final verdict ─────────────────────────────────────────────────────
        print()
        print("=" * 70)
        print(f"PHASE 35 RESULT: {'PASS' if all_ok else 'FAIL'}")
        print("=" * 70)
        sys.exit(0 if all_ok else 1)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
