# Free engine speedup: `--skip off` is 1.18× faster than the deployed default, at bit-identical output

**Status:** MEASURED and PARITY-CHECKED, 2026-08-21. Builder. No commit, no push.
**Origin:** side-finding of `CONTROLLER_MLP_AUDIT.md`; verified independently here.
**Scope:** the 8.3M sandbox engine (`benchmarks/phase60/engine.c`, E1M1 dense) on the 3600X.
**Nothing was modified.** This is a flag flip on the existing binary.

---

## The finding

The engine's deployed dense config is **E3.5 = `--mlp lut --skip on --exp fast`**. The `--skip on`
path (E3) is the *dReLU row-skip* optimisation: it skips the `up` projection for rows whose `gate`
output is ≤ 0, and tile-skips the `down` matvec. It was adopted as an optimisation.

**It is slower than not doing it.**

| config (t6, pinned, 3 runs) | µs/token | tok/s | LUT-MLP organ |
|---|---|---|---|
| `--skip on` (**deployed default**) | 476.3, 495.5, 478.0 → **483.3** | 2069.9 | 161.9, 170.6, 163.2 → **165.2 µs** |
| `--skip off` | 405.2, 408.8, 411.0 → **408.3** | 2449.1 | 86.8, 87.3, 89.2 → **87.8 µs** |
| **gain from `--skip off`** | **1.18× engine-wide** | **+379 tok/s** | **1.88× on the organ** |

## Parity — checked first, per the P60 law

Project law (`project_phase60_engine`): *kernel-bit-exact does NOT compose to correctness-system →
end-to-end parity always.* So parity was established **before** the speed claim, not after:

```
# logits dump, 400 tokens, identical seq/offset
bin/engine_64_1b_check.exe --weights results/phase60/e1_model.bin --mlp lut --skip on  \
    --exp exact --threads 6 --dumplogits lg_on.bin  --ntok 400 --seq 512
bin/engine_64_1b_check.exe --weights results/phase60/e1_model.bin --mlp lut --skip off \
    --exp exact --threads 6 --dumplogits lg_off.bin --ntok 400 --seq 512
cmp lg_on.bin lg_off.bin     ->  BIT-IDENTICAL (1,638,400 bytes each)

# and the quality gate, independently
--bpb --eval-tok 20000, skip on  ->  BPB 0.867424
--bpb --eval-tok 20000, skip off ->  BPB 0.867424      (identical to 6 decimals)
```

**Bit-identical logits and identical BPB.** E2 and E3 are documented in `engine.c:612` as producing
the same output — that is confirmed here, so the 1.18× is genuinely free.

## Why the optimisation loses

Measured, not assumed: at the sandbox shape the dReLU gate leaves **~25% of rows active**
(`gemv_donor_bench mlpctl` measures 2280/8960 = 25.4% at donor shape). The skip path replaces a
**SIMD tile-major** `matvec_lut_full` over all rows with a **scalar row-major** loop over the active
ones. Skipping 75% of the rows does not pay for losing `pshufb` vectorisation on the remaining 25%.

This is the same lesson as Phase 61 and the `matvec_blk4` revert, in a third setting: **a kernel-level
win (fewer FLOPs) does not compose to an engine-level win when it costs the vector path.**

*Flagged as the mechanism hypothesis, consistent with the measurements but not isolated by an
experiment that varies sparsity alone.* The engine-wide 1.18× and the parity are measured facts; the
"why" is not.

## Caveats that must travel with the number

1. **The engine-wide figure is config-dependent.** With `--exp exact` (E3, not the deployed E3.5)
   the total barely moves — 1198.7 vs 1194.0 µs — because `scan-recur` (~750 µs) dominates and
   swamps a 78 µs organ difference. The 1.18× is specific to the **deployed** `--exp fast` config,
   where scan collapses to ~40 µs and the MLP difference becomes visible in the total. The **1.88×
   on the organ itself is config-independent.**
2. **Sandbox scale only.** 8.3M, `D=256`, `MLP_HID=1024`, one machine (3600X). At donor dimensions
   the same flip measures 27.3 ms (`skip=off`) vs 62.6–72.3 ms (`skip=on`) — a **2.3–2.6× organ
   gain**, so the direction holds and widens, but that is a synthetic-weight harness
   (`DONOR_PROJ_RATE.md` §10.2), not the engine.
3. **The trained-model activation sparsity is what decides it.** probe-2 measured 92/79% sparsity on
   the trained model; this run's gate leaves ~25% active. **If a future model is much sparser, the
   skip path could win again** — the flag should be chosen by measurement per model, not fixed.
4. Not tested on the MoE (E4) path, which uses a different kernel.

## Recommendation

Flip the deployed default to `--skip off` for the dense path **after** re-running the parity ladder
(`make gates`) on the full artefact set, and record the flag in the config rather than hard-coding it —
caveat 3 means this is a per-model measurement, not a permanent truth. The `skip` code should stay:
it is correct, it is bit-exact, and a sparser model may need it.

**Reproduce:** the six commands above, plus
`for s in on off; do bin/engine_64_1b_check.exe --weights results/phase60/e1_model.bin --timing --mlp lut --skip $s --exp fast --threads 6; done`
with `OMP_PROC_BIND=close OMP_PLACES=cores` on a quiet machine.
