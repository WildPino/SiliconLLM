# Implementation work remaining: complete276 archive in phase60 C

This is a source-based implementation plan,not executed native evidence or
a frozen qualification protocol.283 full PIQA now passes;actual native
observations still require a separate apparatus/protocol freeze. No further source-core fitting or source replacement is
needed to answer the immediate same-artifact execution question.

## Exact input and reusable code

Use `meth276_qwen05b_i16_private128_diagnostic_repair1.safetensors`,
SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`,
1,329,447,260bytes,225,360-byte JSON header plus8-byte length and
1,329,221,892-byte tensor payload.725 archived fields include218 non-FFN
organs,360 source-FFN fields,144 conditional fields,one LUT and two head
proposal fields. No external donor/checkpoint weight fallback.

`meth274_i16_shared_input_cpu.c` already implements dynamic ties-even I16
inputs/shared Q8 integer dots,private128 BF16 input rows,513-point SiLU
interpolation,Q8/escape down and BF16 rank32 residual. Its actual6144
numeric checks pass;its fixed10ms allocation fails. Reuse the operator
without reopening that cost gate or substituting275's physical layout.

`meth124_centered_factor_cpu.c` and the127 additions in
`benchmarks/donor_adaptation/engine/donor_engine.c` provide CPU parent/child
selection and BF16 factors. They use unaliased contiguous B banks;the new
path must resolve `bank.{layer}.leaf_map[child]` before accessing the
unique B row. All1280 route addresses and30,556 functions/164 aliases stay
fixed. Existing generic FP32 donor forward is an implementation reference;
it is not a qualified276 execution path.

## Fields and executed arithmetic to bind explicitly

Source inspection of the actual archive finds:

| Archived field family | Stored type/shape | Requirement for the C path |
| --- | --- | --- |
| Input/post/final RMSNorm and q/k/v biases | F32 | Python276 copies them into BF16 model parameters;apply that cast before reproducing its equation |
| q/k/v/o projections | BF16,including q/o896x896,k/v128x896 | Accumulate and round at the same BF16 operator boundaries;no donor weight reads |
| Tied embedding/full head | BF16151936x896 | Share the same buffer;round full-head row output to BF16 for reference scoring |
| Source gate/up/down codes/scales | I8 plus F32 row scale | Dynamic I16 input quantization each executed layer;private replacements,escape indices,all biases and residuals retained |
| Source private IDs/gate/up | U16[128],BF16[128,896] | Preserve exact IDs/weights;no new row selection |
| Router/child projection/keys | F32[58880]/[32,896]/[128,10,32] | Preserve original parent128/top4,child10 selection and actual selected weights |
| Parent A/unique B/leaf map | BF16[128,8,896]/[count,896,8]/I32[1280] | Same selected functions;map aliases explicitly,keep exact BF16 SiLU conditional correction |
| Head proposal codes/scale | I8[151936,896]/F16[151936] | Row8-bit proposal,not rank8;round dequantized proposal weights as Python does before finite K64 qualification |

Native RMSNorm,residual additions,RoPE and attention must account for
Python276's BF16 rounding. Its source-FFN FP32 result is cast to BF16 BEFORE
adding the BF16 conditional term;that sum and later residual sum also
round. Reusing the old FP32 donor forward without those boundaries changes
the executed recipe. Exact archive readback alone would not qualify it.

Installed `transformers/models/qwen2/modeling_qwen2.py` inspection confirms
RMSNorm computes FP32 variance/normalization,casts normalized values back
to input dtype,then multiplies the learned weight. RoPE returns cos/sin
in input dtype and forms the two products and their sum in that dtype.
Layer residual additions occur after attention and after MLP. Bind the
installed implementation again at native qualification;current source SHA
`99fa98c5676604cf6ef505892b70fda5c1c4cd835971459f38d61090cccab1e4`.
The SDPA integration file SHA is
`fdf62fb0eb9b5dc0a6796a7e526015865267f271a0d5ac8377968ad05d1454b8`;
native attention cannot claim bitwise agreement with its CUDA kernel from
source formulas alone. Full-model numeric/choice/quality checks remain due.

Qwen configuration:24layers,D896,H4864,V151936,NH14,NKV2,HD64,
RMS epsilon1e-6,RoPE theta1,000,000,full attention,no sliding window,
tied embeddings,EOS151645. Source schema/revision is case-specific.

## Concrete order of work and decision

1. Bind terminal283 pass/raw SHA
   `2ed1e54df9722c32287445eeb2b8d40590742b1893656c374cff6490912a7725`;
   all four gates and same actual archive are required.
2. Implement a phase60 entry path that directly binds all725 archived
   fields and configuration. A generated fixed-artifact offset catalog
   can avoid a new weight container,provided whole-file/tensor identities
   and range/type/shape coverage are verified. Keep donor/source/checkpoint
   fallback impossible. Do not expand all conditional weights into FP32.
3. Freeze native qualification before observations. Reproduce6144 source
   vectors with actual archive bindings and separately qualify routes,
   aliases and isolated BF16 conditional contributions. Then compare whole
   fixed-prefix hidden/logit/choice behavior against the same Python loader.
   Retain original donor control and explicit cache checks;the264 failed
   cached Python recipe remains closed. Stop before speed on a failed guard.
4. Qualify held-out prediction,actual native generations and full task
   behavior with the existing source/data exclusions and prospective
   native execution criteria. References already measured in280/281/283
   may be reused only after exact binding checks. Changes in arithmetic or
   cache are explicitly qualified,not inferred from readback/component pass.
5. Measure actual whole batch1 accepted decoding on this SAME artifact
   with no GPU/model job overlap. Separate prefill/TTFT and charge source
   core,router/selection,selected experts,LUT,head,cache/context and glue.
   Report real resident/active/read bytes and observed locality/DRAM cost.
   Final lowerCI95>=50tok/s remains mandatory. No sum of historical timings
   or theoretical bandwidth is a substitute for the measured whole rate.

A joint failure motivates a changed architecture or execution recipe;
do not quietly enlarge gates or promote a component. Even a small-donor
joint pass is only the reproducible first path. Useful tenfold learned
capacity,larger-RAM route competition and cross-family10B/100B transfer
still require separate actual evidence.
