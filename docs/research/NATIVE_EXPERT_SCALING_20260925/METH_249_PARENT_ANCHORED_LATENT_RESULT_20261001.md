# METH-249: full-parent anchor is stable but count gain is too small

**Decision: close this fixed full-parent/tied-parent-left hierarchy.**
Continuous E160 improves consumed E16 error by **0.0829%**, below the
frozen10% gate; rotated-choice improvement is0.2251%, also below10%.
Positive paired-window gain alone does not qualify useful tenfold capacity.

Frozen producer318da16/session48737 completes160 fits but stops on pooled
parent-score equality before saving/validation. Read-only8e27e39 diagnostic
replays all16 old SSEs exactly. Retention repair5240e53/session76007
saves the full snapshot, then confirms the same16 exact scores; only
`+=` versus the original Python `sum()` changes the ratio's last bit.
[Frozen recovery](METH_249_AGGREGATION_RECOVERY_20261001.md)0c70956,
session86289 exits0: no new fits, original exact aggregation restored,
all readback/conditioning/mean/normal/growth/distinctness controls pass,
and the original consumed validation executes once. No scientific gate
or fitting choice changed; failures and diagnostic remain preserved.

| SSE / target energy | Actual parent E16 | Anchored E160 | Rotated E160 |
|---|---:|---:|---:|
| Fit | .00004587007460894334 | .00004450186296649443 | Not scored |
| Consumed validation, FP64 subtraction | .00011764126288971341 | .00011754371865320873 | .00011780892029672325 |

Fit gain2.9828%, nonincrease prerequisite passes. Absolute1% and both
source-prior10% comparisons pass. All unchanged original FP32 per-window
parent/prior controls are exact. Paired10,000-window-bootstrap gain P05
8.043806510626859e-8, P95 1.149078622979161e-7, seed249250; positive.
Count and rotated10% comparisons fail. These windows are consumed
development data, not independent document-level quality.

All176 decoded composite coefficient matrices are distinct. Child156 is
the only zero-centered-feature cell; its slope uses the preregistered
source sensitivity, not fitted information or a BF16 derivative. Its
feature/JVP/parent-gradient/QR/projection/growth controls pass. Full actual
parent fields remain unchanged. Distinctness does not establish usefulness.

Physical FP64 pilot checkpoint298,727,124bytes SHA256
`5b95ecc272b257929388ac6520cfcbdf3643a9e4aa4aa21c0acd1e69482df7a0`.
[Raw recovery](meth249_parent_anchored_latent_result.json) SHA256
`d4bd08d5e5c0773ef33a5ddbd1fa3aa2292e585bbc3f198697ba543c88a78c0e`.
Retention producer23.515s; recovery11.391s, RSS3,876,233,216bytes,
GPU peak3,474,795,008bytes. Recovery peaks are not original fit peaks.
Local RTX3060/six threads; no T4/download or new C/native timing.

## Changed question before another fit

The fixed left basis was selected from raw-parent-minus-stored-base
corrections in METH-247. It was not selected from the error that remains
after executing the complete actual parent. METH-249 proves insufficient
gain in that inherited output space; it does not prove that every shared
rank32 child correction is ineffective. Diagnose how much centered actual
parent error lies in the inherited space versus a fit-only residual
covariance basis. This changes the output directions while retaining the
complete parent and rank, not its strength or thresholds. Only a separately
frozen optimistic fit-space diagnostic can license a new continuous pilot.
No codec/native private bank, large-RAM routing/DRAM, full quality/rate,
or10B/100B/family transfer is promoted.
