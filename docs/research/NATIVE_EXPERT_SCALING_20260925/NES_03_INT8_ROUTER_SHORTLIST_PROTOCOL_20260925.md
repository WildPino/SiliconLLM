# NES-03: int8 router shortlist, frozen protocol

**Status:** frozen before numerical and CPU results. This is a router-cost
experiment, not a pretrained conversion, distinct E1280 training, or a
100B quality/rate claim.

## Decision and controls

NES-02 measured a 10.002× increase in router+selection time from trained
E128 to synthetic E1280 at fixed top-8. An fp32 exhaustive E169094 router
would address about 1.039 GB/token at the current L6/D256 geometry. This
cell asks whether an int8 full-row sketch plus a *fixed* shortlist can
retain the exact router's top-8 while cutting actual E1280 CPU time. It
changes router scoring/selection only; backbone, experts, top-8, data and
output precision remain fixed. The E128 generation failure is a separate
blocker, so any router speed win is provisional until useful quality passes.

The exact controls are the NES-01 E128 C export SHA-256
`259e1aa73593e8fa8f72a997aaa63d30961342676fa0be587a22d920c804d2cf`
and NES-02 E1280 throughput artifact SHA-256
`25f99a300a26ed90948d04990ed75df9f94ab9a288ef48e6ea0c46fc0a3dcf2e`.
The source checkpoint is SHA-256
`58c84a8730b03c37e98481b8508f5cc84e15757de5513c3e6354f09b975cd462`.
Held-out token IDs are `results/phase55/ids.u16` SHA-256
`33b8cba2a26653599f7f87a4d8e05b38be051ba850d4cbc5d09b561aae133889`.
Use the fixed 90/10 split, 16 nonoverlapping 512-token windows starting at
validation offset 8192, and collect the six pre-router normalized inputs
per position from the unmodified E128 reference model. The same inputs can
probe synthetic E1280 router rows; that arm tests ranking arithmetic on real
inputs, **not** its self-consistent generated trajectory or quality.
Before interpreting shortlist recall, require the offline fp32 dot products
to reproduce at least 99.9% of captured PyTorch E128 top-8 IDs; otherwise
stop and diagnose the reference apparatus.

## Frozen numerical ladder

For each layer independently, quantize each router row symmetrically to
signed int8 with scale `max(abs(W_row))/127`. Quantize each input vector to
signed values in `[-63,63]` with scale `max(abs(x))/63`; store the input as
unsigned byte `q_x+64` for AVX2 `maddubs`, and subtract `64*sum(q_w)` from
each integer dot product. Include the original fp32 bias in the sketch
score. Rank every expert by approximate score, using lower expert ID for
ties; first try a **32-candidate** shortlist, then **64** only if 32 fails.
Rescore those candidates with the original fp32 router rows and bias, select
the exact top-8 within the shortlist, and normalize their logits over the
selected eight. The latter is mathematically the same top-8 softmax after
renormalization in the current C path, apart from fp32 rounding.

For each arm and layer, record top-8 ID recall, fraction of positions with
all eight exact IDs included, rank of the eighth exact expert in the sketch,
the exact eighth/ninth logit margin and the count of changed final routes.
Proceed to native C timing only if the **same** shortlist size passes on
both E128 and E1280 for every layer: at least 99.9% exact top-8 ID recall
and at least 99% positions containing all eight exact experts. These
thresholds protect the top-8 route, not useful generation. If 64 fails,
stop the int8-shortlist candidate and retain the numerical failure; do not
silently widen candidates or change precision.

## Native gate if the numerical ladder passes

Implement the chosen shortlist as an explicit C option, leaving the current
fp32 serial router as default. Verify C sketch/fp32-rescore choices against
the offline probe and measure real E128 C BPB plus the frozen NES-01 16×128
greedy continuations relative to E128's exact C path. Allow BPB degradation
at most +0.001 on the same 20,480-token slice, no more than one additional
looping continuation, and at least 99% top-1 agreement. The exact E128 model
already failed the absolute generation gate; these are *nonregression* gates
for the routing approximation, not promotion of E128 quality.

On the Ryzen 5 3600X, six OpenMP threads, same byte LUT/fast exponent and
3,000 validation input tokens as NES-02, run four separate timings per arm
and discard the first. Measure the new int8 path both with serial sketch
rows and with the existing opt-in `--router-parallel` row scheduling at
E≥512; report both rather than selecting an unrecorded fastest run. Either
configuration may advance if it independently passes every cost and quality
gate. Relative to NES-02's serial E1280 median, require
router+selection at most 268.05 µs/token (0.5×536.1) and total latency at
most 1,598.34 µs/token (0.85×1,880.4); E128 total must stay within 1.05×
1,148.4 µs/token. Report all repeats and code/router RAM. Even a pass is
only a cost/fidelity result at small geometry; 50 accepted tok/s at 10B/100B
requires a quality-valid artifact and end-to-end measurement.

The numerical probe should take under 15 minutes on local RTX 3060 plus CPU
analysis and produce about 50 MB of router-input arrays. The native phase,
if admitted, is one C implementation/build and four CPU repeats per arm;
stop on any numerical, quality or cost gate failure. No T4 or new training
is authorized by this protocol.
