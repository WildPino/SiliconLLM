# METH-126: shared parent A removes 396 MB without changing the E1280 factors

**Decision.** Adopt `M126FB01` as an exact storage representation of the
quality-gated centered E1280 factor bank. Every parent has ten byte-identical
child A matrices in all 24 layers; the native reader uses the single stored
parent A and the original selected child B. This cuts the bank from
893,141,032 to **496,779,304 bytes (44.4%)**. It changes neither numerical
quality nor selected-factor bytes per token. It is not a compact LUT code or
an end-to-end model result.

The [protocol](METH_126_SHARED_A_FACTOR_BANK_PROTOCOL_20260928.md) was
committed at `937729f` before the complete export and CPU outcomes. The
[exporter](../../../benchmarks/native_expert_scaling/meth126_export_shared_a.py)
binds the METH-124 source SHA-256
`8e4ef1f8abea39d9562e8b617f460e89591fa2b2d7fa7478fac0901804064580`.
It compared 27,648 sibling A byte arrays, then reread every router, A and B
byte in the compact file. The [ledger](meth126_shared_a_factor_export.json)
records the compact SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`,
12.28 seconds and 1.411 GB peak process RSS observed at layer boundaries,
within the 15-minute/4-GB stop.

The modified [C fixture checker](../../../benchmarks/native_expert_scaling/meth124_centered_factor_cpu.c)
accepts both bank versions. On the original and compact files its 96 case
lines are identical: 96/96 pass, zero maximum parent/child route, gate and
residual error, and the same deterministic checksum `85889676.352122`.
The compact bank with the preexisting one-byte negative fixture exits 1
with one failed case. Logs:
[original](meth126_original_fixture_raw.log),
[compact](meth126_shared_a_fixture_raw.log),
[negative](meth126_shared_a_negative_raw.log).

The [varied native checker](../../../benchmarks/native_expert_scaling/meth125_varied_centered_factor_cpu.c)
used the same 256 actual E1280 pre-MLP positions and five single-thread
repeats for both versions. The two [logs](meth126_original_varied_raw.log)
and [compact log](meth126_shared_a_varied_raw.log) have identical per-layer
coverage (252–402 selected children) and checksum `17922336.276280403`.

| 24-layer measure | Original E1280 | Shared-A E1280 |
|---|---:|---:|
| Bank bytes | 893,141,032 | 496,779,304 |
| Process RSS | 930,713,600 | 533,766,144 |
| Distinct selected A+B rows touched, 256 positions | 16,290 | 10,573 |
| Distinct selected factor bytes | 233,533,440 | 151,574,528 |
| Nominal selected factor bytes/token | 2,752,512 | 2,752,512 |
| Route median, ms/token-equivalent | 1.679 | 1.539 |
| Factor median, ms/token-equivalent | 1.167 | 1.068 |

These timing values are one sequential pair, not a speed claim. The original
METH-125 run reported 1.480/1.031 ms on the same 256 positions, showing
run-to-run variation. The exact format saves RAM and lowers the distinct
factor working set over this finite varied set; each token still addresses
four A and four B rank-8 matrices per layer. The source BF16 factors and FP32
router are unchanged, so METH-121/123 quality evidence applies to this
representation. A quality-valid LUT compression, low-traffic donor core,
full native integration, accepted-token ≥50/s, and 10B/100B donor transfer
remain open.

Reproduce the export and C checks with:

```powershell
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth126_export_shared_a.py --source benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth124_centered_factor_bank.bin --out benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_shared_a_factor_bank.bin --report docs/research/NATIVE_EXPERT_SCALING_20260925/meth126_shared_a_factor_export.json
clang -O3 -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth124_centered_factor_cpu.c -o benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_factor_checker.exe -lpsapi -lm
clang -O3 -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth125_varied_centered_factor_cpu.c -o benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth126_varied_checker.exe -lpsapi -lm
```
