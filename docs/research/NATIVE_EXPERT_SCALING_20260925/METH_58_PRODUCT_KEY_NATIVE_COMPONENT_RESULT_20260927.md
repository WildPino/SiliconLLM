# METH-58: learned E128 product-key native component parity

**Decision:** the versioned export and C component pass the frozen E128
route/gate/residual parity gate on 96 inputs across all 24 layers. This
validates the learned adapter component; the Qwen core has not been integrated
into `benchmarks/phase60/engine.c`, so neither full-model logit parity nor
accepted batch-1 tokens/s has been measured.

## Bound inputs and artifact

The [protocol](METH_58_PRODUCT_KEY_NATIVE_COMPONENT_PROTOCOL_20260927.md)
precommitted the gate before export. The input METH-56 update-512 checkpoint
SHA-256 is
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`;
the METH-57 source manifest SHA-256 is
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`.
The [exporter](../../../benchmarks/native_expert_scaling/meth58_export_product_key.py)
reads the actual checkpoint, checks those hashes, writes all 24 layers of
rank-64 projection and 8/16 product keys as FP32, and stores all 128 expert
rank-8 A/B factors per layer at their BF16 forward values. It then checks
every exported byte against the source tensors on readback. Every layer has
128 distinct A/B factor pairs by content hash, and METH-57 measured all 128
B slots changed from initialization per layer.

The local generated bank is
`benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_bank.bin`,
93,732,900 bytes, SHA-256
`d725bab99a3bba2235654810375d82e654144277104682defb01dd9a6d29600e`.
It is generated from the bound local checkpoint and is not committed to Git.
The 519,588-byte [frozen fixture](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_fixtures.bin)
has SHA-256
`34702296e1a2c552c4dc0ab16ceaf542b55d7bccc70c81762b4882d1c933d958`.
The [export ledger](meth58_product_key_export_result.json) records all 96
activation hashes, prompt source and positions, selected IDs and resource use.
It used 3.719 seconds after GPU start, 1,238,106,624 peak allocated GPU bytes
and 2,977,644,544 end RSS on the local RTX 3060; no T4.

## C comparison

The [C reader/kernel](../../../benchmarks/native_expert_scaling/meth58_product_key_native.c)
checks the bank/fixture header and exact length. It evaluates the 16
combinations from the best four keys on each axis, takes top four, applies
softmax and BF16 rounding, then reads only the selected experts and evaluates
their rank-8 SiLU residual. The 96 fixture inputs include the first and last
actual hidden state of the first METH-57 code prompt in each layer, plus two
fixed-seed BF16 synthetic inputs per layer. Across them, 124 different expert
IDs are selected and 86,010 of 86,016 expected residual elements are nonzero;
the maximum absolute expected residual is 0.5078125.

The [raw per-case log](meth58_product_key_native_raw.log), SHA-256
`b28dc2bb7f5be258d4118190dad9b578eeb28bb81de2960a99488adb3ea5536`,
records 96/96 exact unordered top-four sets, zero gate mismatches over the
1e-5 bound and zero residual mismatches over the 0.03 bound. The observed
maximum absolute gate and residual errors are both **0** at stored BF16
precision. A negative control increased one expected residual by 1.0 in a
temporary fixture; the C checker exited 1 with exactly one residual failure.
That temporary file was removed after the check.

```powershell
.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth58_export_product_key.py --bank benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_bank.bin --fixtures benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_fixtures.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth58_product_key_export_result.json
gcc -O2 -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth58_product_key_native.c -lm -o benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_native.exe
benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_native.exe benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_bank.bin benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth58_product_key_fixtures.bin
```

Local `gcc` resolves to clang 21.1.8. The exact C result is a component
readback for E128 with BF16 factors. It does not test a compact LUT factor
code, the donor core, a native tokenizer, large-E trained routing or full
generation. The next step is to select a native Qwen-core bridge, integrate
this learned adapter into `engine.c`, and measure route/logit parity and
accepted-token throughput on the same bundle. That design must account for
the dense donor core's active traffic before claiming a viable 50 tok/s model.
