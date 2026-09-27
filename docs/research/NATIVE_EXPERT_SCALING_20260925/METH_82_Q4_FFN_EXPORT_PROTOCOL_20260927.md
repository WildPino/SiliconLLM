# METH-82: packed grouped-Q4 FFN export with exact readback

**Motive and source.** METH-81 verifies 535,739,904 ideal addressed
bytes for a Qwen2.5-0.5B-Instruct core with BF16 tied head+attention,
grouped-Q4 FFN, FP32 controls, FP32 product keys and selected BF16
factors. Export the **same bound BF16 donor** to a physically stored
core before any quality claim. Bind the successful METH-56 E128
checkpoint SHA to metadata for later composition; do not alter it.

**Exact transform.** Each two-dimensional FFN weight is partitioned
per output row into contiguous groups of 64 input weights. Set a
group's scale to its maximum absolute weight divided by 7, rounded
to FP16; for an all-zero group use scale 1.0. Quantize against the
stored FP16 scale with round-to-nearest-even, clamp to signed
integer [−7,7], then add 8 and pack two codes per uint8 byte,
even input index in the low nibble. Reconstruct as
`(nibble−8) × stored_scale`, cast to BF16. Preserve the tied head and
all attention matrices byte-exact BF16 and controls FP32. Do not
run data calibration or adaptation. Store format/version, source
model/revision/hash, group width, code convention and METH-56
checkpoint hash in safetensors metadata.

**Readback and decision.** From the saved file, reload every code,
scale, BF16 and FP32 tensor and compare byte-exactly with the export
buffers. Independently unpack all 72 FFN matrices from stored codes
and scales and compare their BF16 reconstructions with the exporter
reference. Require every input width divisible by 64, all codes in
[−7,7], 290 original tensors mapped to one tied head, 72 FFN,
96 attention and 121 controls, and physical file ≤528,000,000 bytes.
Record per-organ reconstruction error, physical payload/header bytes,
export time and peak resources. A pass permits a fresh development
quality test for the composed METH-56 E128 model; it does not prove
quality, real DRAM traffic, a CPU LUT, full `engine.c` execution or
accepted-token speed.

Local RTX 3060 only; ≤10.5 GiB allocated GPU, ≤20 GiB RSS and
≤15 minutes including file hash/readback. Stop and retain failure
evidence on OOM, nonfinite weight/scale or budget failure. No T4.
