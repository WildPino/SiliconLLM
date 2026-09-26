# METH-23: composed R8 diagnostic passes

Command: `.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth23_int8_core_composition_diagnostic.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth23_int8_core_composition_diagnostic.json`.
This tested the fixed 56 METH-19 documents and 24 METH-20 prefixes,
and therefore is a diagnostic, not an independent promotion gate.

The original BF16 donor scored **0.996779 BPB** and its E128 factor-0.50
adapter **0.996042 BPB**, reproducing METH-19. With R8 applied to both
the tied head and all body matrices, donor BPB rose **+0.001045** and
the adapter scored **+0.000466 BPB** against the original donor. The
adapter remained **−0.000579 BPB** better than its same-R8 donor.
Against the original donor, composed adapter deltas were **+0.006846**
for code, **−0.002158** for prose, and **−0.010801** for technical text.
On 6,144 fixed next-token positions, exact top-1 agreement versus each
arm's original core was **95.426%** for donor and **95.231%** for adapter.
Both pass the frozen 95% floor, narrowly.

The head alone raised donor BPB +0.000309; body alone +0.001025.
The composed quantizer has **493,961,216 int8 code bytes** and
**1,824,256 fp32 row-scale bytes** for 169 matrices. These are planned
packed bytes: this PyTorch run expanded matrices back to BF16 and did
not measure compressed inference. Runtime was 193.6 s, peak allocated
GPU memory 3.646 GB, and final RSS 3.896 GB on the RTX 3060.

**Decision:** diagnostic gate passes; export the exact R8 rule and
evaluate the stored artifact on independent documents. No native speed
or large-E conclusion follows.
