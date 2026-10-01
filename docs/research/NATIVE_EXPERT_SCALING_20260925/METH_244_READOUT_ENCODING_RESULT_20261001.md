# METH-244: continuous fit count gain is erased by readout encoding

Original replay frozen at `acc61cd`,session29975 completes all176 fit
systems and exact physical coefficient/bias checks,then exits1 on pooled
FP64 summation equality. The preserved [aggregation repair](METH_244_AGGREGATION_REPAIR_20261001.md)
is frozen at `95e47b7`;recovery session9416 completes,exit0,without repeating
fits. Common capture energy1,346,611.609791067. All176 coefficient/bias
replays remain exact;normalized score discrepancies1.11e-16/6.66e-16 are
within the declared1e-12 aggregation tolerance. Maximum SSE identity
closure3.324e-16;all normal residual gates pass.

| Fit SSE/energy | E16 | E160 |
| --- | ---: | ---: |
| Nondeployable FP64 continuous solution |.00002054765|**.00001036981**|
| Decoded physical coefficients,FP64 arithmetic |.00005742063|.00009942231|
| Actual mixed FP32 stored operator |.00005742063|.00009942231|
| Encoding distortion alone |.00003349831|.00006603061|
| Signed encoding/error cross term |.00000337467|.00002302189|
| Actual-versus-decoded arithmetic difference |3.06573e-13|2.80121e-13|

**Raw E160 fit error is49.5329% lower than raw E16.** Actual encoded E160
instead has the known worse fit. Encoding penalty(actual-minus-raw) is
2.1202times its positive actual E160–E16 loss gap. Both frozen diagnostic
decisions pass:continuous10% fit gain exists and encoding penalty exceeds
half the gap. The SSE decomposition includes its positive cross term;
coefficient norms alone do not explain output loss. Forward arithmetic
differences are far smaller than encoding distortion.

**Decision:** change representation to preserve the continuous correction
without requantizing the entire adapted parent. This qualifies an encoding
investigation,not an accepted bank. Raw solutions inherit METH-240's stored
priors,not unencoded ancestors. Only fit data are read;no raw held-out count
gain,independent quality,native rate or larger-donor capacity is established.

[Raw result](meth244_readout_encoding_result.json),SHA256
`5164e4deb8ee6866b9d3b1f62f0d3468f6e7d814c517000876ccbf8305949849`.
All176 per-cell errors,prior/parent comparisons,cross terms and closure
checks retained. Original attempt78.562s,local3060,six threads,resource
bounds monitored per row but actual peak bytes absent from exception JSON.
Recovery0.312s,RSS1,442,476,032bytes,GPU peak1,174,405,120bytes. No T4,new
fitting,validation or replacement checkpoint. Original executed runner
SHA256 `46fb32b7a017c095a665baba9b1d854589ffe698ff3920518257786b81ab15c4`.

Next: [separate conditional residual proposal](SEPARATE_CONDITIONAL_RESIDUAL_PROPOSAL_20261001.md).
First price a fixed rank32 BF16 residual over the unchanged mixed operator
on CPU. Learned factor truncation/rounding and held-out count gain remain open.
