# METH-40: CPU cost of the stored rank-64 router sketch at 10× E

**Uncertainty.** METH-36's saved rank-64 int8 router sketch retrieves
most exact E128 routes. METH-37 rejects C64 after route replacement;
METH-38's C96 arm nearly removes omissions on reused prompts, but
is not quality-promoted. Independent of that pending quality test,
the sketch still scans every expert. Determine whether its CPU
scan alone is compatible with a 20 ms/token end-to-end target
when `E` grows by about 10× from the hypothetical ~10B to ~100B
capacity geometry of METH-26.

Bind the stored METH-36 sketch artifact SHA-256
`133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`
and its L24/D896/E128/rank64 format. A Python exporter verifies
all tensor shapes, hashes and metadata, then writes a raw binary
seed with each layer's fp32 basis, int8 codes and fp32 scales.
The raw [seed export ledger](meth40_router_seed_export.json) was
written before CPU timing. Its 5,713,944-byte local binary
`results/native_expert_scaling/meth40_e128_router_seed.bin` has
SHA-256
`e0bca061fb9f42b17b22ddf10d17f4da37599e9995cbe698ee19d380aa489262`.
The C benchmark expands the 128 stored sketch rows deterministically
to `E=27,355` and `E=273,547`, with a small deterministic scale
jitter to break repeated-row ties. **These extra rows are synthetic,
not distinct trained experts.** The test measures cost only and
cannot establish route fidelity or quality at larger E.
Compile with `clang -O3 -mavx2 -mfma -march=znver2 -fopenmp ... -lm`.
The pre-timing local binary SHA-256 is
`53710db22afb013355ee9b7befde83de65647f3a0afa9cbe1ea09c6cbb1c5515`.

For each token/layer, project a fixed synthetic D896 input into
rank64 using the saved fp32 basis; scan every row using an AVX2
int8-code × fp32-activation dot and fp32 row scale; keep the best
96 rows by a deterministic score/ID heap. Use independent
per-thread heaps and merge them, so six-thread timing does not
also include a serial E-row score-buffer pass. A scalar-vs-AVX2
small-row self-test must pass first. Each one- and six-thread arm
uses 8 warmup tokens, then four timed repetitions: 64 tokens for
E27,355 and 16 tokens for E273,547. Report median ms/token,
projection, scan and merge components, checksum equality between
thread counts, exact addressed code/scale/basis bytes per token,
and pool footprint. No fine-router exact rescoring, selected
expert LUT, core or generation is included; do not call this
end-to-end speed.

**Decision bounds.** If the fastest measured scan+selection at
E273,547 alone exceeds 20 ms/token, this one-byte exhaustive
rank64 scan cannot meet the ≥50 tok/s end-to-end goal on this
host. If it is ≤20 ms, the scan remains a candidate only; report
its remaining time within 20 ms and require exact candidate
rescore, selected experts, core and model quality before promotion.
The arithmetic 40 GB/s lower bound for 420,168,192 code bytes is
10.50 ms/token, before scales and work. Measure rather than assume
the actual host bandwidth or cache behavior. The 10× ratio is
reported, not gated independently; a RAM-scaled expert count
still needs end-to-end verification.

Local CPU only. Maximum 8 GiB RSS for the largest sketch/scale/
basis pool, 15 minutes per complete sweep, no T4. If the raw
seed or self-test fails, stop without timing. Leave the existing
`engine.c` inference modes unchanged; this is a component probe
for deciding whether to integrate the index.
