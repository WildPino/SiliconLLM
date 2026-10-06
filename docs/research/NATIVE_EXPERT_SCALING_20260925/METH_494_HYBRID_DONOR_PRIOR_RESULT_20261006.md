# METH494 — hybrid donor prior: result and limits

6 October 2026. Goal ACTIVE/INCOMPLETE. Branch `research/native-expert-scaling`.

## Decision

The complete bank 11 Gaussian prior and logical cost contract are independently
admitted: all 7 compiler, 9 independent audit and 5 final admission gates pass.
This makes one concrete, source-informed initialization reproducible. It is
an untrained prior, with no target fit, local quality, whole-model quality,
timing, DRAM or useful expert-count result.

The changed geometry is

```
F_e(x) = V_e ReLU(W_e x)
       = (V_e W_e/2)x + (V_e/2)|W_e x|
G_e(x) = L_e x + B_e |A x| + c_e.
```

The first equality/decomposition concerns the ideal real ReLU operator with
the original quantized weight values. It is not byte equivalence to the
source F32/A16 program. Private L has 768 input directions; the complete
candidate therefore has no structural rank-512 Jacobian ceiling. Actual
rank, empirical quality and the usefulness of individual functions are not
established by this dimension statement.

## Scope and artifacts

All 128 original source expert IDs remain. Four weight-selected hidden rows
per expert provide 512 shared ternary directions; all coordinate ties and
packing are verified. The selection score is the individual isotropic
Gaussian ReLU second moment, .5 ||V_col||² ||W_row||², before considering
cross-neuron terms. It is not a measured importance on the donor distribution.

Covariance and calibration use only the 11,721 canonical development inputs,
once each. The 5,819 previously consumed validation inputs do not determine
the prior. Expert 0 has no development input and retains its weight-derived
prior with the explicitly marked global amplitude initialization. Preserving
that slot does not prove its utility or calibrate its amplitude.

| Output in `results/native_expert_scaling/meth494_hybrid_prior` | Bytes | Meaning |
| --- | ---: | --- |
| `bank_prior.bin` | 127,232,136 | Packed shared ternary A, initial centroid keys, weighted I8 L/B, F32 scales/bias |
| `L0_prior.bin` | 301,989,912 | Unweighted F32 linear coefficients for all 128 IDs |
| `C0_prior.bin` | 201,719,832 | Unweighted F32 even-feature/bias priors for all 128 IDs |
| `geometry.bin` | 20,475,956 | Development S, regularized C/Cholesky, Gaussian K/Kreg/eigensystem |

Selection, calibration, new controls and logical cost are also retained.
The prior directory inventory before late terminal serialization is
651,534,179 bytes. Full hashes and schemas are in the raw/audit records.
The large F32 files are conversion/audit data; the physical bank alone is
127,232,136 bytes. Slots and source provenance do not by themselves establish
128 distinct useful functions.

## Independent numerical evidence

The auditor imports neither compiler nor math module. It independently
reconstructs the development covariance with blocks of 127 rather than 256,
uses exact rational means and integer half-even rounding for calibration,
decodes every ternary group, checks all source row selections and performs
alternate factorizations of the source linear product.

The even Gaussian kernel uses the acos angle formula in the auditor and
the asin formula in the compiler. All 128 serialized readouts satisfy the
prospectively fixed normal-equation envelope, including F32 coefficient
rounding. All exported L/B codes, scales, bias and centroid keys agree BYTE.

| Numerical quantity | Result |
| --- | ---: |
| Covariance ridge | 0.13940687996880982 |
| Kernel ridge | 0.3731011458105516 |
| Numerical smallest Kreg eigenvalue | 0.3932588132165184 |
| Numerical largest Kreg eigenvalue | 154405.94992493256 |
| Audited computed-Gram eigen lower | 0.3932433537861806 |
| Eigenvector orthogonality Frobenius error | 5.219421221473775e-14 |
| Eigensystem reconstruction Frobenius error | 3.4078401605937164e-10 |
| Maximum linear rounding-envelope ratio | 0.2498091859363185, required ≤1 |
| Maximum equation rounding/arithmetic-envelope ratio | 0.062253738349296164, required ≤1 |
| Computed-Gram objective-gap expressions | 0.0025529452119519482 … 0.026483823104943505 |

There were 218 computed correlations just outside [-1,1], admitted only
after the frozen ≤1+2e-12 check and clipped as specified. These numerical
checks are not formal outward-rounded interval proofs. The objective-gap
expression concerns the chosen computed regularized Gaussian projection;
it is not approximation error against the real donor or a quality bound.

## Cost contract

Per expert, per bank, private physical storage is 992,256 bytes. For 12
equivalent banks, private storage is 11,907,072*n bytes; shared A is 1,204,224
bytes, and the current n=128 keys are 1,475,712 bytes.

Consulting one expert per bank gives 14,587,008 logical weight bytes per
12-bank token, plus 4,718,592 logical LUT read bytes and 746,496 table write
bytes. Quantizers, input/features/output, core, head, attention/state and
workspace are additional. This is an independent byte recount, not a DRAM
measurement, cache assumption or tokens/s result.

The fixed per-bank operator counts are 983,040 integer MACs, 98,304 group-4
lookups and 30,720 key F64 MACs, plus 24 exponentials/two normalizations.
The original FFN has 4,718,592 MACs; MAC ratios do not determine measured
speed. L needs I64 accumulation; B at width 512 is I32-safe under the stated
I8/A16 bounds. No native evaluator was invoked in 494.

## Actual execution and resources

| Stage | Actual tool/session/terminal | Exit | Late wall | OS peak |
| --- | --- | ---: | ---: | ---: |
| Binding, after source freeze `3487eea` | `432796` | 0 | 3.094 s | 40,624,128 B |
| Compiler, after binding commit `6b08cda` | `a90000` / 29847 / `8768a5` | 0 | 149.125 s | 249,159,680 B |
| Independent auditor, after raw commit `b574abc` | `84ebc6` / 50889 / `b3dca3` | 0 | 130.797 s | 271,634,432 B |
| Audit Windows query and finalizer | `1997cc` | 0 | metadata only | — |

Limits: builder 90 s/256 MiB; compiler 600 s/512 MiB; auditor 900 s/512 MiB;
CPU 0 and one BLAS thread. Hashing, source reads, solves and output writing
are charged; OS-peak observer/watchdog continue through full record writing.
Both actual Windows Event 1000 queries are available, with no matching fault.
No main/audit rerun or numbered repair. Engine and the three foreign tracked
changes retain their original hashes. No GPU, new resource or source FFN/model
call. Compiler performs 128 Gaussian projection solves, not target fitting;
auditor performs no projection solve. Gradient updates are zero.

Important hashes:

- Binding: `91fb15d7db330d6a36b2f25809bfa769b99fb62fe9df3b9c074d884c25b55b58`.
- Compiler raw: `ab3501d8add8164ef0adb2ccc2c2512ea56166a491afff0b24f887752124e527`.
- Audit raw: `4436ef88bd926c6c5b42e7ebbec6db92ccfe31bbf099c80134f319c3fee72029`.
- Admission: `432054f90a0b426b386001e391609851981e3fd111d506005dd13ea588c2eb59`.

[Protocol](METH_494_HYBRID_DONOR_PRIOR_PROTOCOL_20261006.md),
[compiler raw](meth494_hybrid_prior_result.json),
[independent audit](RETENTION_494_20261006.json),
[admission](ADMISSION_494_20261006.json).
See [whole-project algebra and next](METH_494_WHOLE_ALGEBRA_AND_NEXT_20261006.md).
