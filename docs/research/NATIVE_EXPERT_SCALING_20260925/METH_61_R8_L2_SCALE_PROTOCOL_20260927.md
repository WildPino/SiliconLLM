# METH-61: fixed weight-only R8 scale correction with BF16 attention

**Uncertainty and decision.** METH-60's BF16-attention/R8-head+FFN arm
fits its ideal active-byte allotment at 548.320 MB/token but misses the
≥95% prompt top-1 gate by five of 4,125 positions. Before adding stored
precision, test whether the original R8 row-maximum scale rule leaves
recoverable rounding error in the remaining head and FFN codes. A
passing diagnostic merits packed mixed-format export and an entirely new
source-disjoint quality audit; a fail rejects this exact weight-only
scale rule and points toward calibration or quantization-aware adaptation.

**Bound inputs.** Original Instruct source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-59 R8 core SHA-256
`5b6ace4a027d127a6eb49ca99197edd4a1b810ebe681d8e3b6ab7edc088ca9bd`,
METH-56 checkpoint SHA-256
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`,
METH-57 manifest SHA-256
`9571f5d61c29e34b27c05ec531232a08b9eb6cd81bdf85aa5648ff74f1db3d75`.
Use the same 24 already viewed METH-57 prompts only for the diagnostic;
do not use their outcomes to choose scales, per-row iteration counts or
which layers are modified.

**Frozen numerical rule and arms.** For each selected original BF16
weight row `w`, start from `s=max(abs(w))/127` and its rounded/clipped
integer codes `q`. Repeat three times: set `s=(w·q)/(q·q)` clamped
to at least `1e-12`, then set `q=clip(round(w/s),−127,127)`. Finish with
one more least-squares scale calculation for the final `q`. Do not add
new per-row metadata. Keep all attention matrices BF16 and all controls
unchanged. Score four arms using the fixed METH-56 adapter:

1. METH-60's BF16-attention baseline with original R8 head and FFN.
2. Corrected R8 head only; FFN original R8.
3. Corrected R8 FFN only; head original R8.
4. Corrected R8 head and FFN.

The original BF16 adapter control must again match 3,997/4,125 donor
prompt IDs, and arm 1 must match METH-60's 3,914/4,125. Report pooled
and per-category top-1, per-organ weight reconstruction L2, and ideal
addressed bytes. A diagnostic arm passes only at ≥95% pooled and ≥90%
per category, with ≤560 million ideal core+router+selected-expert bytes.
Any passing arm is still a viewed-prompt diagnostic, not a quality-valid
native model.

**Cost and stop.** Local RTX 3060 only, ≤10.5 GiB peak allocated GPU,
≤20 GiB RSS, ≤10 minutes. Stop on identity or resource-gate failure.
No T4. This probe does not measure generation, PIQA, semantic behavior,
actual DRAM traffic or full-model accepted tokens/s.
