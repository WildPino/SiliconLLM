# METH-247: rank32 preserves local fit gain, consumed count gain fails

Frozen at `df779ea`;session59971 completes,exit0. All176 original physical
coefficient/bias replays exact,raw fit metrics equal METH-244. All basis,
mean,readback,absolute-function controls pass;176 decoded composite FP64
matrices distinct. Only zero-response child156 uses the frozen source-
coefficient SVD fallback. No noise/duplicate-count or strength/rank retry.

| Normalized SSE | Raw E16 | Factored E16 | Raw E160 | Factored E160 |
| --- | ---: | ---: | ---: | ---: |
| Fit |.00002054765|.00004587007|.00001036981|.00004049225|
| Consumed validation |Not scored|.0001176413|Not scored|.0001202956|

Actual factored E160 fit gain **11.7240%** passes10%;new E16 fit improves
over old stored parent. On consumed validation,E160 loses **2.2563%**
against equally factored E16. Bootstrap P05/P95 **-2.896605e-6/-2.412669e-6**,
negative. E160 beats rotated.0001354738,child prior.0001658301 and parent
prior.0001856472 by10%,but gain versus old stored E16.0001279367 is only
5.97%,also failing10%. All128 original control energies/prior/base scores
exact. Absolute1% function passes;useful-count acceptance does not.

**Stop this fixed activity-weighted rank32 residual pair.** No native
learned-bank/full-model/fresh-quality promotion. Native9.637ms is source
component evidence and cannot be combined into an accepted full model.
Next diagnose raw consumed validation before choosing representation
versus continuous hierarchy;do not just increase rank or adjust tau.

[Raw result](meth247_weighted_residual_pair_result.json),SHA256
`a433b502960ee2c50c5ab1fa1b214d61327e4584935d27fa10682b08a4ab7ee9`.
Ignored actual snapshot `results/native_expert_scaling/meth247_layer12_weighted_residual_functions.safetensors`,
**151,680,804bytes**,SHA256
`02615b1717b125d417f8b137569a00984760545d60da76c20172652c8887d3b4`.
Shared stored base16 plus actual BF16 factor/bias functions;not the old
1.591GB fully expanded bank. Runtime100.735s,RSS3,983,265,792bytes,GPU
peak3,434,100,736bytes,local3060/six host threads,no T4. Same176 frozen fit
replays and fit-only basis selection,one consumed validation. No new source.
Next [METH-248](METH_248_RAW_VALIDATION_PROTOCOL_20261001.md).
