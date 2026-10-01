# METH-220: quantization-stage mismatch and float-input cost rejection

The [protocol](METH_220_FFN_INTERMEDIATE_PROTOCOL_20261001.md) and both
runners were frozen at `7ccd57a`. The instrumented C extension reproduces
all **384 existing W8A8 outputs bit for bit**, validating the tape before
interpreting it. These are consumed METH-125 states, not a full-model
trace of the compact candidate.

## Arithmetic diagnosis

The frozen *hidden-boundary-only* hypothesis fails: **25 input-code
differences occur in 23 token/layer rows**, and input group scales also
have small FP32 differences. Those 23 rows all exceed the METH-219
0.0005 output-error bound. One additional failing row, token 3/layer 21,
has equal input codes, scale difference 7.45e-9, and one changed hidden
code only 7.63e-6 from a rounding boundary. Total failing rows: **24/384**.
Do not attribute every failure to hidden quantization alone.

Replacing GPU hidden codes/scales with the exact taped C values lowers
the worst output relative L2 to **7.446626e-7** across every row. This
controlled stage replacement localizes the output discrepancy upstream
of the down projection. It does not establish the precise instruction
or division policy responsible for the input differences, nor universal
full-model emulator parity. The GPU/C quantizers are not interchangeable
under METH-219's frozen tolerance.

Worst W8A8 difference is token 8/layer 3: relative L2 **0.006068950**,
one changed input code, 188 changed hidden codes. The emulator median
is 4.095281e-7. That median cannot waive the outliers.

## Fixed float-input kernel

The alternative streams the identical physical Q8 bytes, sign extends
eight weights with AVX2, accumulates FP32 FMA lanes per group, then
applies FP16-decoded scales. It uses FP32 inputs and hidden activations.

| Frozen component check | Measurement | Limit | Outcome |
| --- | ---: | ---: | --- |
| Median output relative L2, 384 rows | 4.307608e-7 | <=1e-4 | Pass |
| Maximum output relative L2 | 5.455058e-7 | <=5e-4 | Pass |
| Median 24-layer FFN ms/token, six threads | **12.539723** | **<=10** | **Fail** |

Three passes: **12.610512 / 12.539723 / 12.460089 ms/token**, all failing
the feasibility budget. These are one run's component timings, not a
statistically matched speed comparison with W8A8 or full end-to-end
throughput. Stop this specific float-input kernel; no layout, reduction
or threshold retry after outcome. No quality/generation/task sources
were consumed. No T4 was used.

[Raw result](meth220_ffn_native_diagnostic_result.json), SHA256
`bf615fd93cc3c343fdec6fab5957b812e176ed2c30ab74135edf68d72e7877ae`,
binds every token/layer, script, source, executable, physical core,
FFN bytes, fixture and tape. Tape: 27,515,932 local bytes, SHA256
`927ea702d6cb830186953b4574d49a6306d495ce238734dbd48603067fface8b`.
Runtime after imports 12.719 s, ending RSS 1.645 GB and GPU peak 85.5 MB.
Session 2394 exited 0; METH-219 sessions 21814/61631 exited 1 with
preserved failure records. No project inference job remains active.

## Method consequence and next decision

Saved grouped-Q8 storage is real; BF16 quality fails METH-214, existing
W8A8 native cost fails METH-182, and this W8A32 kernel now fails cost.
Changing compute precision alone has not yielded a licensed complete
compact composition. This does not exclude all grouped-Q8 kernels or
all compressed representations, and does not measure W8A32 model quality.

The next transfer mechanism should give conditional experts a measured
pretrained reconstruction target caused by a cheaper core, rather than
add weak descendants to the already accurate dense donor. A possible
bounded route is packed-Q6 native feasibility *for a changed conditional
recovery method*: existing METH-186 storage has more active-byte margin,
but direct quality and global rank corrections already fail. Check native
cost before training a route-specific activation-output correction;
do not re-evaluate the uncorrected Q6 candidate as a quality solution.
Use fit/validation training-corpus states, freeze selection before
scoring, and require useful conditional function on unseen source states
before an n ladder. The new variable must be conditional target/function
learning; repeating METH-189/191 global correction, METH-175 weak-child
training or METH-218 static bias fitting is not licensed by this result.
Full donor-relative independent quality, real CPU LUT/DRAM large-n cost,
multi-family/approximately-10B transfer and >=50 accepted tok/s remain open.
