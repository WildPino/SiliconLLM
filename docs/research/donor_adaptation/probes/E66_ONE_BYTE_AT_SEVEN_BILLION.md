# E66 — one byte at seven billion

**Status:** COMPLETE — score survives; the registered rank bar does not.

**Brief:** `briefs/BRIEF_E66_ONE_BYTE_AT_SEVEN_BILLION.md` (+ addenda A–D)
**Runner:** `benchmarks/donor_adaptation/engine/e66_one_byte_at_7b.py`
**Control result:** `benchmarks/donor_adaptation/engine/results/e66/e66_controls.json`
**Canonical combined result:** `benchmarks/donor_adaptation/engine/results/e66/e66_one_byte_at_7b_run2.json`
**Audit:** `benchmarks/donor_adaptation/engine/results/e66/e66_run2_audit.json`
**Log:** `benchmarks/donor_adaptation/engine/e66_run.log`

**Scope:** Qwen2.5-Coder-7B, quality and rank only. No rate; `G-E63d` remains `VOID` and OWED.

## 0. Verdict

> **At 7 B, the one-byte layers plus ternary head are effectively score-neutral: A2 is only
> +0.000378 BPB from fp32. Four of five 32-token trajectories are exactly identical to the fp32
> donor. The fifth diverges, leaving 137/160 exact-position agreement: substantially alive, but
> below the preregistered 150/160 rank bar.**

This is the first post-hoc compressed 7 B artifact in the programme that is plainly functional
on `engine.c`; it is not a 50 tok/s result. Its 9.258 GB file is a fidelity rung, not the final
traffic budget.

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

## 3. Treatment score checkpoint

Both registered int8 artifacts were exported and their BPB cells asserted `attn=avx4`,
`quant=int8`, the expected sidecars and `N_PREDICTED = 12264`.

| arm | fold | BPB | delta vs fp32 | delta vs chance |
|---|---|---:|---:|---:|
| A1 | layers | 0.674442168983 | +0.000415613983 | -3.395664052292 |
| A2 | none | **0.674404586411** | **+0.000378031411** | **-3.395701634864** |

Thus `G-E66d = BELOW-CHANCE` and `G-E66e = ONE-BYTE-IS-NEARLY-FREE`. The signed fold effect
`A1-A2` is **+0.000037582572 BPB**: the registered `<0.01` prediction holds, and folding is
slightly harmful rather than helpful at one byte.

## 4. Post-run defect: rank run 1 is void

The first complete continuation observed A1 131/160 and A2 137/160, but its generation helper
discarded rather than asserted the engine's `CONFIG` line. The commands passed `--attn avx4`, yet
`G-E66c` explicitly requires reading that state back. Addendum D therefore voids only rank run 1.
These counters are retained as raw observations and are not the E66 rank answer.

## 5. Repaired rank run 2

Addendum D pinned the score result by sha256, discarded its rank member, and re-ran only the ten
engine generations plus fp32 reference. The repaired runner was committed before launch. Every
generation now stores an engine-reported `CONFIG attn=avx4 ... quant=int8`; all ten assertions
passed. Run-2 token IDs are identical to the void run-1 observations.

| arm | prompt 0 | prompt 1 | prompt 2 | prompt 3 | prompt 4 | total | `G-E66f` |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 fold=layers | 3/32 | 32/32 | 32/32 | 32/32 | 32/32 | **131/160** | **RANK-DOES-NOT-SURVIVE** |
| A2 fold=none | 9/32 | 32/32 | 32/32 | 32/32 | 32/32 | **137/160** | **RANK-DOES-NOT-SURVIVE** |

The miss is not diffuse token noise: both arms reproduce four complete trajectories and diverge
autoregressively on one. The formal unit remains 160 positions because that is what §5 registered,
but the per-prompt split is mandatory context: effective trajectory count is five, not 160
independent trials.

## 6. Predictions scored

| registered prediction | outcome |
|---|---|
| both planted controls fire | **TAKEN** |
| A2 below chance | **TAKEN**, by 3.395702 BPB |
| A2 within 0.05 BPB of fp32 | **TAKEN**, +0.000378 |
| A2 rank at least 150/160 | **MISSED**, 137/160 |
| fold worth less than 0.01 BPB | **TAKEN**, 0.0000376; direction slightly harmful |

Four of five predictions are taken. The only miss is the gate the brief explicitly said carried
the experiment, so the overall verdict remains mixed rather than being promoted from score alone.

## 7. What is and is not licensed now

1. The 7 B one-byte fidelity question is closed for this donor and construction. Do not repeat it
   unless donor, quantizer, head format, slice or estimand changes.
2. This does not license a scale law: E62's non-monotonicity remains, now with a fourth point.
3. The result is post-hoc. It neither proves nor requires a healing run at this fidelity rung.
4. No rate was measured. The dense one-byte traffic remains far outside the 50 tok/s budget;
   `G-E63d` is still the separate clean 10 B carved-int8 rate debt.
5. Rank run 1 remains void. Controls/BPB come from the pinned score run; rank comes only from run 2.
