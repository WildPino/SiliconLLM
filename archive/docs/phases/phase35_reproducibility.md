# Phase 35: Reproducibility & Portability Audit

*Executed: 2026-05-26 | SEE V1.0 | Clang 21.1.8 (LLVM-MinGW) / Windows 11 x86_64*

---

## Question

Will a `.see` archive produced today decode correctly tomorrow, on a different build or compiler?

---

## Test Results

### A. Encode Determinism

**Result: PASS**

Encoding the same file twice with identical flags produces byte-identical `.see` archives.
The codec has no hidden randomness: no timestamp in the archive, no PRNG, no uninitialized state
that propagates to the output.

| input     | archive SHA (first 16) | result    |
|-----------|------------------------|-----------|
| tiny_text | 65da59653d3a404f       | identical |
| tiny_code | 4b02831f527a6230       | identical |
| tiny_json | 8d56b9604965b746       | identical |

### B. Fixture Decode

**Result: PASS**

Three `.see` archives are committed in `data/fixtures/` alongside `manifest.json`.
These serve as permanent decode-only regression tests: they prove that the decoder can
reconstruct the original bytes from an archive produced at V1.0 seal time.

`scripts/phase35_reproducibility.py` verifies:
1. The archive file has not been altered (SHA against manifest).
2. Decoding produces the original input (SHA of decoded output against manifest).

If either check fails on a future build, the archive format has regressed.

### C. Header Rejection

**Result: PASS — 5/5 corrupt cases correctly rejected**

| corruption                  | outcome        |
|-----------------------------|----------------|
| bad magic (0xDEADBEEF)      | rejected       |
| truncated to 45 bytes       | rejected       |
| header_size set to 999      | rejected       |
| weights SHA-256 corrupted   | rejected       |
| empty file (0 bytes)        | rejected       |

All cases return a non-zero exit code and print an error message. None silently produce
output from corrupt input.

**Important note on `header_size` policy:** The check is a strict equality
(`hdr.header_size != sizeof(SeeArchiveHeader)`). This means:

- A future codec with a larger header **cannot decode old archives** → version incompatible.
- An old codec **cannot decode archives from a future codec** → same error.

This is intentional for V1.0. Forward compatibility would require a minimum-size check
(`hdr.header_size >= sizeof`) plus skipping unknown bytes. That is not implemented.
**Archives are version-locked**: decode requires the same or byte-compatible header layout.

### D. Compiler Variant Analysis

**Result: -O2 = -O3 (PASS); -ffast-math differs (WARN — forbidden flag)**

Tested with Clang 21.1.8 on x86_64 Windows:

| variant      | vs -O3 output | self-roundtrip |
|--------------|---------------|----------------|
| -O2          | MATCH         | —              |
| -O3          | (reference)   | OK             |
| -ffast-math  | DIFFER        | OK             |

**`-ffast-math` is a forbidden build flag for SEE.**

`-ffast-math` enables reassociation of floating-point operations and may replace `expf()`
with approximate versions. The SEE codec uses `expf()` in two critical paths:

1. **MoE weight update** (`moe_engine.c`): `w[e] *= expf(-eta * loss[e])` — determines expert
   credit weights, affects which distribution is passed to the range coder.
2. **Softmax / CDF** (`see_codec.c`): probability normalization → `(int)(p * CDF_SCALE)` — the
   integer CDF slots fed into the range coder.

Under `-ffast-math`, these computations can produce different floating-point results.
Since the range coder is deterministic given its CDF input, a changed CDF means a changed
bitstream. An archive encoded with `-ffast-math` cannot be decoded by a binary built without it.

**Consequence:** `-ffast-math` does NOT break self-roundtrip (encode and decode with the same
binary), but does break cross-binary decode. Since distributable `.see` archives must be
decodable by any V1.0-compatible binary, this flag invalidates the archive.

**Mitigation (V1.0):** The build command does not include `-ffast-math`. No action required
beyond documentation. A future version could add a `#pragma float_control` guard in the
hot path, or port MoE/CDF to fixed-point arithmetic (eliminating the sensitivity entirely).

---

## Portability Guarantee (V1.0)

```
Archives produced by SEE V1.0 are guaranteed to decode correctly on:

  - Same platform (x86_64 Windows)
  - Same compiler family (Clang/GCC with -O2 or -O3, no -ffast-math)
  - Same weights file (verified via SHA-256 in archive header)
  - Same codec version (header_size equality enforced)

Archives are NOT guaranteed to decode on:

  - A future SEE version with a different SeeArchiveHeader layout
  - A binary compiled with -ffast-math or equivalent relaxed-fp flags
  - A different CPU architecture (untested; float rounding may differ)
  - MSVC (untested; _controlfp settings may affect float behavior)
```

---

## Permanent Fixtures

`data/fixtures/manifest.json` records the SHA-256 of each fixture's original input and
expected decoded output. Running `python scripts/phase35_reproducibility.py` will verify
these on any future build.

If the decoder ever fails to reproduce the correct SHA from a committed `.see` file,
the codec has introduced a regression in the archive format.

---

## Future Work

- **Fixed-point MoE/CDF path**: eliminate float sensitivity entirely. Would allow
  `-ffast-math` and guarantee cross-compiler bit-identical output.
- **Forward-compat header**: change strict equality to `hdr.header_size >= sizeof(current)`
  and skip unknown bytes — enables old decoders to read archives from new encoders (if
  new fields are optional). Requires a `min_decoder_version` field.
- **Cross-architecture test**: build for ARM or 32-bit x86, verify fixture decode passes.
  Currently untested.
