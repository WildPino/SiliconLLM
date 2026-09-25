# METH-00: GigaChat base traffic target before a conversion experiment

**Question.** Is preserving the pretrained GigaChat 3.1 base topology in a
direct C port enough to make ≥50 tok/s plausible on the Ryzen 5 3600X, and
which organ treatments could create a defensible latency margin? This is a
**source-derived arithmetic preflight**, not a measured decoder rate, quality
test, or new artifact. It reuses the pinned
[11.480B-parameter donor organ ledger](../donor_adaptation/audits/STRAT_01_GIGACHAT31_10B_METADATA_SCREEN_20260918.md)
and the existing BF16/Q4 quality/binding results. No model was loaded and no
GPU, T4, or C benchmark ran. CPU cost: under one minute of arithmetic.

## Identity, assumptions and reproducible calculation

Donor: `ai-sage/GigaChat3.1-10B-A1.8B-bf16`, revision
`189fff27a1dee68473960c3d5bca53e0e07a3191`, base 26 layers (no MTP).
The organ counts below are **active large-matrix elements per autoregressive
token** from the source/config/header audit, not distinct total parameters or
measured DRAM bytes. Each selected expert is charged once; embedding lookup,
norms, KV/state, quantization scales/padding, cache behavior and kernel
overhead are outside this minimum. `W4` and `W2` mean hypothetical exactly
4 and 2 payload bits per weight, not the actual mixed `Q4_K_M` file layout
or a quality-valid 2-bit format.

```python
organs = dict(mla=650051584, routed=589824000, shared=147456000,
              dense0=41287680, router=2457600, head=197001216)
active = sum(organs.values())
w4_bytes = active / 2
budget20 = 40e9 * .020
budget14 = 40e9 * .014
for name, count in organs.items():
    print(name, count / 2 / 1e6, (w4_bytes - count / 4) / 1e6)
print(active, w4_bytes / 1e6, (w4_bytes-budget14)/1e6,
      (w4_bytes-(organs['mla']+organs['routed'])/4)/1e6)
```

The `40 GB/s` denominator is a **measured aggregate DRAM anchor** in
[PHASE64_BUDGET.md](../../PHASE64_BUDGET.md), used here only as a favorable
traffic yardstick. It is not the measured effective rate for the donor's MLA,
irregular experts, or a 2-bit kernel. The 14 ms streaming allotment leaves
6 ms of the 20 ms/token target for all other work; it is a design budget,
not a passed gate.

| Organ | Active weights/token | Ideal W4 payload MB/token | Saving from W4→W2 on this organ alone, MB/token |
|---|---:|---:|---:|
| MLA projections | 650,051,584 | 325.026 | 162.513 |
| Routed experts | 589,824,000 | 294.912 | 147.456 |
| Main head | 197,001,216 | 98.501 | 49.250 |
| Shared experts | 147,456,000 | 73.728 | 36.864 |
| Dense layer-0 FFN | 41,287,680 | 20.644 | 10.322 |
| Router | 2,457,600 | 1.229 | 0.614 |
| **Total** | **1,628,078,080** | **814.039** | — |

At 50 tok/s, ideal W4 large-matrix traffic alone is **40.702 GB/s**. At
40 GB/s, its payload floor is 20.351 ms/token, already above 20 ms and far
above the 14 ms streaming allotment. Meeting 14 ms at that rate requires
removing **254.039 MB/token (31.21%)** of this idealized payload. Treating
only MLA as W2 leaves 651.526 MB; only routed experts as W2 leaves 666.583
MB. Neither single-organ treatment meets the 560 MB payload allotment.
Treating **both** MLA and routed experts as ideal W2 leaves 504.070 MB,
equivalent to 12.602 ms at the favorable 40 GB/s yardstick; real scales,
dequantization, routing and smaller matrices can erase that margin. This pair
is a cost *candidate*, not a measured speed or quality claim.

## Decision, prior failures and stop boundary

A direct Q4 donor port is useful as a fidelity/cost baseline; parameter count
and partial parity cannot establish the joint quality/rate result. Under the
declared 40 GB/s and 14 ms streaming design budget, the ideal W4 payload
requires a substantial reduction, and neither MLA-only nor expert-only ideal
W2 meets that budget. A different measured bandwidth, cache behavior or
kernel could change this conclusion; the actual Q4 layout must be measured.
The next scientific choice is an organ-specific **quality sensitivity** cell
on the already bound GigaChat artifact, with the other organs fixed, followed
by a separately checked composition only if its parts are viable. A selective
2-bit expert treatment is one candidate, but the prior
[STRAT-02E W2 expert PTQ](../donor_adaptation/probes/STRAT_02E_W2_BF16_EXPERT_SCOUT_RESULT.md)
was noncompetitive on a different model and format; repeating that exact
construction without a changed donor/quantizer/training hypothesis is not
licensed. Any GigaChat cell must state the changed coordinate and freeze its
paired donor-relative quality and byte gates first.

This preflight **does not justify T4 work or resuming the old normalization
parity queue**. Stop the preflight here: more desk arithmetic cannot establish
quality, kernel throughput or actual cache behavior. If a viable format/organ
route is identified, measure its integrated C path on the same checkpoint;
otherwise compare the dense-source joint-training route documented in
[METHOD.md](METHOD.md). No 100B inference follows from these numbers.
