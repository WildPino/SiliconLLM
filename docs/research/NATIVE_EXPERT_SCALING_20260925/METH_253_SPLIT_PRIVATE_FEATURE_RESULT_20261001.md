# METH-253: identical private nonlinear functions pass native cost

Frozen fd5d88a, session17039 exits0. All361 physical segments/header/EOF
read back exactly. All344,064 check values are **bitwise equal** to252;
256-state aggregate checksum79.074600755 is unchanged. The two-workshare
feature implementation preserves original BF16 private row arithmetic,
LUT and complete readout. Prior scoped numeric/source-fidelity gates carry.

Native passes9.456050/9.317569/9.315366ms, median **9.317569<=10**.
Every frozen gate passes. No retiming or selection change. Original source
pooled function SSE/energy .000026084689113638605 (81.03% below old source
fixture) is conserved; no new source/model quality was scored in253.

[Raw result](meth253_split_private_feature_native_result.json) SHA256
`9262f98cd87021d32697bb79badc1f07d7480a558def3cce6c9e48fb7466611a`.
Check SHA256
`e8f2db8290f416ddc589ef2b48d09968b1af781705a759c8e78fd9e20bd5f666`;
old/new output payload SHA256
`52bb99669a38fd4726be8ed88e7152aed09b3470bb641a4f00dcfe5a16aa29fe`.
Actual329,334,308byte source fixture unchanged. Runtime8.110s,
RSS1,255,931,904bytes,native peak345,509,888bytes,six threads/localCPU,
no GPU computation/T4/download. Split execution reads no row map; it
computes selected shared rows redundantly before private overwrites.

This qualifies the private32 nonlinear source operator only. Separately
freeze matched E16/E160 unit selection using actual saved source weights,
original learned parent readouts/fit captures/keys. Do not replace source
fixtures with claims of learned routing/large-n/DRAM or full model quality/
>=50 accepted tok/s. Full same-artifact composition remains required.
