# METH-344: head activation precision attribution on consumed343 states

Freeze138dffb; authoritative exec19959 exit0. Raw SHA
`5936c80f3cd02914547e5baa029ce3d5d9c6067519b96d3727b960c97186ce98`.
MAIN194.734s/max checked RSS2,090,258,432B, below20min/8GiB. All three
prospective oracle/primitive gates PASS. Actual original F32 head comes from
byte-identical retained lookup, no58GB source teacher reload or whole forward.

ALL96 per-vector source F32 head reconstructions close original logits with
relative<=1e-6/exact choices; actual native I8/I64 head oracle BYTE EXACT.
A16 nearest-even/scalar-I64 controls pass; max4096 dot17,045,131,264 requires
wide global sum. Counterfactuals use the SAME immutable I8 head/scales and
saved normalized states; new logits/SHAs retained for ALL96 cases.

## Descriptive counterfactual results, not untouched quality

| Fixed states | Head operation | Original-top1 agreement | Old55 differences recovered / new differences introduced | Mask-token mean NLL delta |
| --- | --- | --- | --- | --- |
| Original | I8+A8 | 95.0758% | 30 / 27 | +.024750 |
| Original | I8+A16 | 98.5795% | 49 / 9 | +.002690 |
| Original | I8+unquantized activation | 98.5795% | 49 / 9 | +.002681 |
| Original | originalF32/F64 dot | 100% | 55 / 0 | approximately0 |
| Actual native | I8+A16 | 96.6856% (35/1056 differ) | 28 / 8 | +.006507 |
| Actual native | I8+unquantized activation | 96.6856% | 28 / 8 | +.006514 |
| Actual native | originalF32/F64 dot | 96.6856% | 30 / 10 | +.003813 |

Original-states F32/F64 numerical floor relative1.156e-7; source-head I8+A16
relative.003671 versus I8+A8.012117. At actual native states, even sourceF32
head retains35 discrepancies and mean logit relative.047705: upstream change
remains. Head-only A16 mean logit relative.048289 and all-target NLL delta
-.008592; it does not restore original upstream state.

## Concrete next variable and limits

Select **HEAD activation16 only**, keep core/experts activation8 and ALL
serialized I8 weights/scales/banks/lookup unchanged. At these fixed states its
35 choice differences equal both unquantized-activation and source-F32-head
counts; avoids extra F32-head read bytes. This is a finite descriptive inference
supporting the smallest next actual implementation, not whole quality proof.

Freeze new AVX2 I8-weight/I16-input dot: lane I32 accumulation bound for
cols<=4096, global I64 horizontal sum, prescribed F64 scaling/F32 output.
Require independent exact integer/Tiny/fault/native old-state composition,
actual complete CPU margin at unchanged gates and then NEW source exclusions/
whole original-primary quality. Free generation can change future inputs,
so these fixed-state results cannot stand for generated trajectories.
34395% FAIL remains; no threshold change or old-data promotion. No useful
larger-n/LUT/physical DRAM/accepted rate/family result. Details/command in
[344 protocol](METH_344_SWITCH_HEAD_ATTRIBUTION_PROTOCOL_20261003.md).
