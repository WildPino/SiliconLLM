# E66 — one byte at seven billion

**Status:** IN PROGRESS — planted controls passed; no treatment cell has been exported or read.

**Brief:** `briefs/BRIEF_E66_ONE_BYTE_AT_SEVEN_BILLION.md` (+ addenda A/B)
**Runner:** `benchmarks/donor_adaptation/engine/e66_one_byte_at_7b.py`
**Control result:** `benchmarks/donor_adaptation/engine/results/e66/e66_controls.json`
**Log:** `benchmarks/donor_adaptation/engine/e66_run.log`

**Scope:** Qwen2.5-Coder-7B, quality and rank only. No rate; `G-E63d` remains `VOID` and OWED.

## 1. Controls-first checkpoint

Addendum B changed the execution order so two expensive 7 B exports are not built before the
existing instrument paths prove they are alive. The runner itself was committed before launch
and refused dirty-source execution.

| gate | measured | reference | \|d\| | threshold | status |
|---|---:|---:|---:|---:|---|
| `G-E66a` — 1.5 B int8 path | 0.7688588323885 | 0.7688588381537 | 5.765e-09 | 1e-08 | **FIRES** |
| `G-E66b` — 7 B ternary path | 4.0172324738248 | 4.0172325980000 | 1.242e-07 | 1e-04 | **FIRES** |
| `G-E66c` — configuration | `attn=avx4`, expected `quant` on both cells | exact | — | exact | **FIRES** |

The preflight took 2,625 seconds. This is not a rate measurement. It read only the two existing
control artifacts; `D:/_ktmp/e66/coder7b_i8_foldlayers.bin` and
`coder7b_i8_nofold.bin` did not exist when the checkpoint closed.

## 2. Provenance

| object | value |
|---|---|
| runner HEAD | `04216cc391daf481e2a9898a7918082d797f2319` |
| runner Git blob | `34ded51fa5b54f8a2a1641a6a6746590038ff15e` |
| runner sha256 | `8fe0460265a11afc42b30e09d9880f65ae9a0e94fc03fde012e76ca0e90b56a8` |
| engine sha256 | `56272fdbe615d61739094605cb026aa308fd74604ba5598fab501c09188ae687` |
| controls JSON sha256 | `da0dbe3ad667ad017290805647745584f0c709787832650f5177a92fa04cac9d` |

## 3. What is and is not licensed now

Both planted controls fired, so the two registered int8 exports and treatment cells are
licensed. There is still **no E66 answer**: `G-E66d/e/f/g` have not been measured, no score or
rank claim may be made, and the catalog must continue to say the treatment is pending.
