# METH-134: CPU-master sparse training path for a third expert tier

## Uncertainty, prior and decision

METH-121/123 establishes donor-relative quality for 1,280 distinct
learned centered BF16 children of an E128 parent. METH-126 stores the
same factors without duplicate sibling A rows, and METH-125 measures
their actual selected-factor CPU cost. Q7 and Q15 compact child-B
formats failed independent blind grounding (METH-131/133), so the
next learned-count rung must start from exact BF16 factors. Naively
putting 12,800 trainable child-B copies and AdamW states on the local
12 GB GPU cannot preserve this path. The question here is whether a
CPU-master factor bank can gather and update only selected grandchild
rows with correct gradients while starting at the exact E1280
function. Passing this apparatus gate permits a separately registered
E12800 training and quality experiment; it does not prove useful
learned E12800 capacity.

Bind METH-126's bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`
and METH-125's 256 actual E1280 state vectors SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
Use layer 0 for the bounded apparatus experiment. Decode the stored
BF16 B exactly to FP32 and create ten identical grandchildren for each
of the 1,280 children. Preserve the original parent router, child
router, four selected experts, gate weights and shared A. A seeded
independent 32-wide projection and ten keys per child select one
grandchild per selected child; these new keys are **untrained**. Store
the E12800 layer-0 B tensor on CPU and transfer only selected rows to
the local RTX 3060.

## Frozen apparatus gates

1. Check bank/vector format and all source hashes. On 256 actual
   layer-0 vectors, assert grandchild ID divided by ten equals the
   selected E1280 ID, each selected grandchild B equals its source
   child B bit-for-bit in FP32, and the factor output before any update
   is exact against E1280 FP32 factor arithmetic. Record distinct
   selected source children/grandchildren and transfer bytes. This is
   an initialization identity, not quality or learned capacity.
2. On a deterministic small bank with repeated selected IDs, compare
   a custom CPU-master sparse gather/backward path with ordinary dense
   PyTorch autograd on the same GPU, FP32 inputs and squared-output
   loss. Require maximum absolute selected-row gradient and one SGD
   update discrepancy <=1e-6, and exactly unchanged unselected rows.
3. Run one actual 256-position layer-0 factor backward through the
   E12800 CPU-master bank. Require a finite nonzero gradient, storage
   proportional to unique selected rows rather than all 12,800,
   finite post-update weights, changes only to selected rows, and
   no full E12800 bank copy or dense gradient on GPU. Record transfer,
   gradient, resident memory, GPU peak and elapsed time.

Stop at a failed binding/parity gate, ten minutes elapsed, 4 GiB
process RSS, 4 GiB allocated GPU memory or 500 MB newly written disk.
The local RTX 3060 is the only accelerator; no T4. Do not save a
12,800-row checkpoint as evidence of useful experts. If the apparatus
passes, next freeze a real E1280→E12800 training recipe, new-source
quality/grounding and routing/CPU cost gates before claiming scaling.
The projected full 24-layer FP32 CPU B master is 8.81 GB, before
optimizer state and workspace; the BF16 inference B payload is 4.40
GB. These are arithmetic projections, not measured full-model memory.
