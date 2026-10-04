# METH-394: compact full-score router representation with exact refinement

Prospective after393 retained77a3268. Uncertainty: can a source-weight-only
low-rank router preserve actual selected identities AND full normalization
while reducing addressed matrix coefficients?393 rejected top64-only
normalization; this variant keeps estimates for ALL candidates and refines a
fixed small selected subset exactly. It changes router scores, not expert
outputs or their dimensionality. GigaChat316 full-FFN latent-output failure is
not evidence about this different classifier-score representation.

Read both original128/256 actual F32 router segments from338/380 payloads,
independent368 full manifest parser and every segment SHA bound to original
source hashes. Complete payload hashes inherited393, not freshly rehashed;
size/mtime unchanged. Freshly hash ALL384393 teacher/natural trace files,
exact header/indices/phases/input/score shapes. Existing consumed quality
cohorts are diagnostic inputs only, not new whole-quality acceptance data.

For EACH12 banks/mode, independent F64 actual-input/full-F32-weight product
rounded F32 must match captured scores relativeL2<=1e-6 and selected IDs exact.
Compute source-only F64 economy SVD; fixed r=8,16,32,64, save/readback/hash F32
basis Vh[:r] and codes U[:,:r]*S[:r]. No fitting to captured responses/training.
F64 products of F32 factors/F64 actual input then F32 scores. Fixed shortlists
m=1,4,8,16, stable approximate-score ranking; select original lowest-index
maximum among candidate exact captured scores (simulated original-row dot).
Refine those scores exactly, retain approximate scores for every other
candidate; full mixed normalization using F32 subtraction/exp rounded F32,
F64 sum, selected numerator and final F32 probability. Stable maximum centering
even if an unrefined estimate outranks the exact selected candidate.

For every bank/mode and each fixed(r,m), local screen gates: identity agreement
>=.9999; selected probability absolute-relative p95<=.01 AND maximum<=.05;
addressed coefficient proxy ((768+n)*r+768*m)/(768*n)<=.5. All24 bank/modes must
pass for eligibility on that source. No pooled bank override or postscore
grid/threshold changes. If none pass, reject this unchanged fixed source-only
SVD/refinement grid. Any PASS only licenses a NEW native numerical and NEW
whole original-primary quality protocol; no speed, quality or useful n claim.

Full original routers remain stored for refinement, with additional F32
factors; report both. Coefficient/byte accounting excludes activation access,
rank selection/normalization/dispatch and is not physical DRAM or timing.
Independent numerical control, all source/trace hashes and every factor
roundtrip are apparatus gates; candidate rejection is a valid scientific
outcome. First apparatus failure retained before any changed experiment.

Budget MAIN20min/RSS<=8GiB, expected<=2min, NumPy2.4.6/OpenBLAS capped1 thread
via installed threadpoolctl. Sequential after393 native process exit0 consumed;
no model/native timing/download/GPU/T4 overlap. Fresh factor/output paths,
complete local library configuration and hashes in record. No actual C model
or acquired weights modified; final useful>256/LUT/DRAM/cross-family/~100B goal
remains open.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth394_switch_router_rank_screen.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth394_switch_router_rank_screen_result.json
```
