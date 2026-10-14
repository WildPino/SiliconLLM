# Actual contextual generator codecs: precision and charged C cost

Prospective14 October2026. Goal remains pretrained CHATBOT to compact reusable
SSM/SWA plus useful selective capacity, quality and accepted>=50 CPU tokens/s
on the same artifact, physical DRAM, families/scales and conversion cost.
This inquiry resolves whether full-depth source generators can be represented
more cheaply; it is not a generic dense-runtime goal substitute.

## Frozen source-only encodings

All23 original BF16 in_proj matrices6704x2048, source sites1..23. No layer,
channel, attention, source function or capacity is removed. No weight fit,
data calibration or new model execution. First-site F0 remains outside.

- I8-row: per-row maxabs/127 rounded toF32 scale; actual BF coefficient divided
  by that stored scale inF64, nearest-even integer, clipped[-127,127]. Zero
  row has zero scale/codes. Store315785216 I8 bytes+616768 F32 scale bytes.
- I4-g128: sixteen128-column groups per row. Scale maxabs/7 rounded toF32;
  F64 division/nearest-even/clipped[-7,7]. Adjacent codes q+7 in low/high nibble;
  code15 invalid. Store157892608 packed bytes+9868288 scale bytes.
- All scales/nonzero conventions/order frozen before output inspection. Export
  full matrix, not selected positions. Runtime keeps packed storage, no decoded
  coefficient array. Coefficient/data movement budgets are distinct from DRAM.

Three physical workers, masks inferred/verified by the qualified topology
binder, static row partitions, PASSIVE/zero blocktime, same binary/compiler
and profile for BF source control/I8/I4. No worker-count or group-size grid.
BF h decoded once toF32; FMA8 lanes. I8 dot sums q*h then applies row scale.
I4 each128-column partial uses8 lanes, group scale folded into orderedF32 lane
accumulation. Scalar lane reduction thenF32 raw->BF GG, actual BF MuP multiply
inF32->BF fields. These are specified new approximate operators, not bit-exact
source generators. FP control0x1f80 on every numeric worker.

## Complete declared measurements and audit

Same four FIT/DEV short/long own-history records and final16 positions as the
qualified source generator screen. Both resident/parameter-evicted64-position
modes. Rotate all six permutations of BF/I8/I4, one warm group per recipe/record,
retain all tails, no overhead subtraction or favorable-repeat selection.
Charge nibble/sign unpack, conversions/scales, all23 calls and dispatch inside
timing. Evict all actual coefficients/scales plus MuP separately per recipe;
flush retained outside. Archive every I8/I4 raw/GG/field word, not summary-only.
Every new BF control output must match the qualified full original archive.

Independent stored auditor reconstructs ALL315785216 coefficients per codec
from actual BF source and manual nearest-even integer rounding, verifies scale
and every packed code. For every captured h, contract decoded coefficients
inF64, require native raw relative<=1e-4, reconstruct rawF32->BF->MuP exactly,
and replay every native call against stored words. Report actual-source field
error overall and gate/x/B/C/dt slices for all records/sites. The historical1%
source-function fidelity screen is reported separately; do not relax/relabel it.
Approximate-codec whole-model eligibility must use separately frozen own-history
FIT/DEV/task/followup gates, rather than infer it from this local proxy or speed.
This screen makes no whole-model selection, useful-n or quality promotion.

Complete audit even on local fidelity or cost failure. Full input/output seal,
source offsets and physical packet ledger, worker/thread identities, terminal
and explicit closure. Conversion CPU seconds and temporary/storage bytes charged.
No GPU, teacher collection, optimizer, RESERVED, or numerical training rerun.
If a codec fails20ms alone it cannot provide the proposed pipeline under this
profile; a passing generator-only cost still leaves state/readout/FFN/attention/
head/router/frontend. No reciprocal generator timing is reported as token rate.

One new named E family; pipeline wall<=3600s, OS<=4GiB, family<=1GiB, CPU<=6
physical workers for initialization/3 during measurement, no GPU. Frozen source,
driver/auditor/protocol before launch. On long wait pause; user resumes manually.
Old M303/305/306 concern different GigaChat/synthetic/integer-input apparatus;
their cost failures remain, and this new measurement does not borrow their rates.
