# METH-453: fixed compact WI and exact sparse native I8 WO

5 October 2026. PROSPECTIVE protocol, scientific code unimported at writing.
Freeze this protocol and both new sources before the first import, compilation,
activation count or numerical observation. ONE execution, retain first failure;
any repair needs a new numbered source/protocol. Goal remains ACTIVE/INCOMPLETE.

## Question and evidence binding

451 qualifies fixed signed Walsh bases and their uncompressed control, but its
both-I4 representation fails the unchanged argmax gate. 452's factorial native
heads show WO-only compression crosses all four selected compact-changed pairs;
WI-only crosses two. This motivates a NEW joint precision/layout method:
reuse EXACT 451 compact WI bytes, retain ORIGINAL-coordinate native I8 WO bytes
and row scales, execute only columns whose ACTUAL original A16 code is nonzero.
No WO basis transform, new quantization, fit, thresholding, seed or data search.

Bind raw451 SHA256
`26e6ee00de51ea1cfc4cf6617fc69cef36950e1298ae19f153357545dd19f46f`
and raw452
`857128c717284ab7796384e1ad6d3d214d64e46da43158efd127a0839080e3c7`.
Rehash their complete output archives and committed helper/protocol sources.
Bind source128 export380 manifest/config/tensor entries and all 7,541,946,880B
payload bytes, original engine and original374/389 binaries. Independently parse
the whole manifest, preserve payload size/mtime. No old controller main/rerun.
The 336 consumed prefixes, original IDs/probabilities, books18..23 and full heads
are the exact retained451 data. These data are not fresh held-out evaluation.

## Exact integer and floating point contract

Original WO q[r,j] is signed I8 without -128, shape [768,3072]. Its positive finite
F32 row scales s[r] are preserved. Store exact transpose [3072,768] contiguous
I8. Original quantizer: F32 absmax/32767, zero scale1, F32 division, nearest-even
rounding and clamp[-32767,32767], yielding alpha F32 and c[j] I16 without -32768.

    J = {j : c[j] != 0}
    N[r] = sum(j in J) I64(q[r,j]) * I64(c[j])
    out[r] = F32((F64(N[r]) * F64(s[r])) * F64(alpha))

Every omitted term is EXACT zero. Increasing original j, partition J into at most
512 indices per I32 partial, then combine partials in I64. The signed magnitude
bound512*127*32767=2,130,641,408<2^31 applies to every partial. The full dot may
exceed I32 and must never use an overflowing I32 total. Independently sum FULL
matrix products in I64 including zeros, then require integer equality. Require
output BYTE equality to the original qualified native I8 operator at EVERY
actual activation, with unchanged F64 multiplication order and final F32 cast.

Tiny tests BEFORE real numerical forwards: positive/negative127 coefficients,
positive/negative32767 activation codes, widths4/512/1024, Python-integer full
dot oracle, dyadic scale/cast Fraction oracle, alternating signs, all-zero input,
single final column, invalid -128 coefficient and zero-scale rejection. Qualify
unaligned original row-major line unions independently at offset phases0/17/63.

WI uses 451's exact wi_packed[128,3072,384]U8, wi_scales[128,3072,12]F32 and
wi_signs[768]I8. Q_D is signed block256 Walsh /16, computed F64 then cast F32;
672 ACTUAL basis calls independently checked against explicit Walsh products
at relative error<=1e-12. Qualified block64 I4/A16 integer-primal checks remain
enabled. Correct/wrong raw WI outputs and IDs must be BYTE identical to451.
ReLU remains in ORIGINAL neuron coordinates before WO A16 quantization.
Probability, original residual, F32/F64 RMS conventions, final normalization,
original tied I8/A16 vocabulary head and argmax are retained in full.

## Stored geometry and logical screening

Per expert: compact WI1,327,104B plus original WO2,362,368B =3,689,472B.
128 stored functions:472,252,416B plus768 shared WI signs =472,253,184B before
archive headers. Original128 bank605,945,856B. This is about78% original storage,
MORE precision/storage than the failed both-I4 recipe. Its old60% stored gate
stays FAILED and unchanged. The NEW80% storage gate is declared here prospectively;
it is useful only if active execution/RAM/whole quality and rate subsequently pass.
Transpose inverse all coefficients/scales and reload saved bank all arrays BYTE
exact. Retain real per-expert fingerprints; uniqueness is not useful-n evidence.

For s actual nonzero codes, coefficient/scale bytes =1,327,104+768*s+3072.
Add these declared prospective per-function buffers, conservatively as distinct:

| Component | Bytes |
| --- | ---: |
| Shared WI signs |768|
| Original/transformed F32 WI inputs |6144|
| F64 in-place Walsh workspace |6144|
| WI A16 codes |1536|
| WI I32 block partials/F64 weighted accumulator |36864|
| F32 raw/ReLU neurons |24576|
| WO A16 codes/allocated U16 indices |12288|
| WO I32 partial/I64 accumulator/F64 scaled output |15360|
| F32 WO output |3072|
| Residual/final/head F32 inputs |9216|
| FFN/final F32 norm weights |6144|
| Two A16 scales/probability/index count |16|

This is unique logical addressed footprint for a prospective direct packed-WI /
column-WO implementation, divided by ORIGINAL4,733,952B expert coefficient/scale
payload. Original workspace excluded from denominator makes this comparison
conservative. It is NOT actual Python allocations or C traffic/latency/DRAM.
No hardware cache residency assumed. Temporary reference arrays in this Python
qualification count toward measured process peak, not a deployable descriptor.
In-place F64 scalar Walsh may realize the declared workspace, but C primal must
prove it; exact pair-activation LUT tables/build/gathers still need extra charging.
Full core, routing, head, cache, prefill and integration costs are still open.

Candidate future64-aligned WO columns occupy12 lines of64 bytes each; NPZ storage
does not implement runtime alignment. Actual original row-major offset phase is
read from the source manifest. With G=unique floor((j+phase)/64), source union is
768*|G|-767 when BOTH groups0 and48 occur, otherwise768*|G|. Independently enumerate
the line set in tiny cases. Report per-position correct/wrong counts/lines/bytes,
mean/p95/max, all six book means and each expert's observed column union. These
unions concern this cohort only and do not establish hardware DRAM or future reuse.

## Ordered stages and retained data

1. Fresh bindings/runtime/process admission; all originals preserved.
2. Tiny new arithmetic/fault/line qualifications; exact all128 WO transpose and
   saved bank reload; no coefficient omissions or WI re-encoding.
3. All336 original full native FFN/probability/post/final/head replay BYTE exact;
   WO scales/codes exact captured values; sparse and dense original WO outputs
   exact. Retain original complete logits/prefixes again as this experiment's data.
4. Full original-vocabulary predictions for correct ID, ID+1 modulo128 and removed.
   Correct/wrong WI states exact451; every new WO matches original native WO at
   THAT actual new activation. Removed complete heads exact451. Original p remains
   fixed under ID intervention; this is a local function control, not changed routing.
5. Independently stable F64 KL against the original posterior: cross entropy minus
   original entropy agrees with dot(p,logp-logq) within1e-10, finite, >=-1e-10.
   Retain336 full logits/control, all intermediate states and actual IDs, KL arrays,
   changed masks, book means, correct/wrong WO A16 codes/scales and128 column unions.
6. Report logical screening; comparison with452 WI-only is descriptive because WO
   input basis/rounding differs. No quality or bit identity inherited from that hybrid.

Expected1008 new sparse-WO projections =336 original+672 correct/wrong;
672 qualified WI bases/projections. Ten archives retain approximately692MB:
bank, originals/prefixes, three complete logits/states pairs, WO quantized inputs.
Stream-hash every retained output. Hash counter counts file streams; per-function
in-memory fingerprints are additional and not included in that counter.

## Prospectively fixed gates and decision

ALL ten controller apparatus gates MUST pass: fresh parents/source/runtime, all
128 transposes/reused WI/saved bank,336 originals,672 retained WI states,1008 exact
sparse/full/native WO projections,672 WI bases/primal checks, independent KL,
removed heads, tiny bounds/faults/lines and original identities/private nonlinear
gates/probabilities/norm/head. Invalid apparatus never yields quality evidence.

ALL seven feasibility gates for the CORRECT-ID local control:

- Mean original-posterior KL<=.01.
- Every book mean KL<=.05.
- Changed argmax fraction<=.01: at most3 of336.
- ID+1 minus correct mean KL>=.01.
- NEW nominal stored bank ratio<=.80.
- Logical footprint mean AND every book mean ratio<=.60.
- Logical footprint p95 ratio<=.65.

ALL PASS licenses ONLY a NEW frozen actual C cost/primal experiment, with packed
direct versus exact pair-LUT WI and exact sparse-column WO, every builder/scan/
reduction/glue charged. Any feasibility FAIL closes this fixed local recipe before
export/timing. Do not tune thresholds or sweep seeds/clipping on these positions.
Readout/activation-aware correction needs a separately justified metric/exposure
protocol; retained validation outcomes cannot become training samples.

## Resources, stops and scope

CPU-only, one physical CPU affinity[0], Torch/BLAS one thread, Torch2.6.0+cu124,
NumPy2.4.6, qualified DLL hashes. NO GPU, new model/data download, optimizer,
engine edit or deployment. Preserve only the exact approved pythonw publisher
daemon invocation; reject other Python/meth.exe model jobs. Free disk>=2GiB.
Bound admission300s, numerical600s, total900s, process peak4GiB, outputs704MiB.
Expected approximately180..300s and2.5..3.5GB; guards terminate and retain failure
stage/partial hashes. No completed experiment is rerun on shell timeout.

ONE command after freeze and committed operational resumption:

    .venv\Scripts\python.exe benchmarks\native_expert_scaling\meth453_switch_sparse_wo_pilot.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth453_switch_sparse_wo_result.json

Keep the whole objective: useful RAM-scaled n, viable CPU LUT/routing/realDRAM,
full fresh donor-relative prediction/generation/tasks AND>=50 accepted batch1
IDs/s on SAME artifact, actual other-family/~100B transfer. All128 real functions
in one finalbank and consumed prefixes cannot establish those global conditions.
