# METH-60: locate R8 core ranking damage under the active-byte budget

**Uncertainty and decision.** The stored Instruct R8 core plus fixed
METH-56 product-key experts passes METH-59's reused-document BPB gate but
fails its prompt top-1 gate: 94.230% versus ≥95%. The R8 donor alone is
94.255%, pointing to the core. Determine whether head, attention or FFN
rounding causes the lost rankings, and whether restoring one organ at BF16
fits the 560 MB/token active-payload design allotment. A passing arm is
only a diagnostic candidate; it needs stored mixed-precision export and
a new source-disjoint quality audit before native promotion.

**Bound inputs.** Original Qwen2.5-0.5B-Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-59 packed R8 core SHA-256
`5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd`,
METH-56 update-512 checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-57 manifest SHA-256
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`.
Use the exact METH-57 24 prompt IDs already viewed by METH-59; this is
mechanism diagnosis, not independent validation.

**Frozen arms.** Score original BF16 donor and adapter as controls;
then score the same fixed adapter with (1) all core matrices R8,
(2) BF16 tied head with R8 attention+FFN, (3) BF16 attention with R8
head+FFN, and (4) BF16 FFN with R8 head+attention. Restore BF16 tensors
directly from the bound source, not by requantization. Keep every
normalization/bias control at its original value and preserve tied
head/embedding aliasing. Report top-1 versus the original BF16 donor
overall and by code/prose/technical; also report gain over all-R8 and
parameter/byte charge for each restored organ. Retain per-prompt counts.

**Interpretation.** A candidate must achieve ≥95% pooled top-1 and
≥90% in every category **and** ≤560 million ideal addressed bytes per
token for core codes/scales/controls, product-key projection/keys and
selected BF16 expert factors. The byte sum is arithmetic, not a DRAM
trace. If none meets both, identify the organ whose restore gives the
largest top-1 gain per added byte and design a narrower precision map;
do not silently promote an over-budget arm. If BF16 original-adapter
control fails to reproduce METH-57's 3,997/4,125 top-1 matches, mark
the apparatus invalid and stop.

**Cost and stop.** Local RTX 3060 only, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS, ≤10 minutes. No T4. Stop on budget overrun or failed
identity controls. A top-1 screen cannot establish semantic quality,
generation, PIQA or accepted native token rate.
