# METH-313: BF16 repair verified, complete target export stopped on routing

Freeze `316efa1`, exit1 after 48.641s; stage `layer13_test_targets`; `AssertionError(('captured source route slot mismatch', 13, 'test', 0))`. Entire six-case dataset is NOT qualified. Three archives preserved, no source/state omission or threshold relaxation.

The source-supported BF16 activation repair passes261120 finite rounding patterns; truncation fault detected97920 cases. Captured post-SwiGLU maximum relativeL2 in completed layer1fit/layer1test/layer13fit cases is3.010e-7/3.087e-7/3.152e-7, below SAME1e-4 gate. This verifies the packing explanation for312; it is not a compact model quality result.

| Case | Unique actual inputs | Archive bytes |
| --- | --- | --- |
| layer1/fit | 8948 | 385159524 |
| layer1/test | 8318 | 358041804 |
| layer13/fit | 5042 | 217029660 |

Exact hashes/paths/nonlinear controls retained in [failure record](meth313_gigachat_mixture_targets_result.failure.json), SHA `a707295a6fd0285776ba431fabdf6fb3abf6f4951de6f1a2fb742a5971d3eef2`. Controller `42a014c335bda2021fd61c22a97e236db35fa2edcbcb0d7a0ef5e00fc95b9c2d`, protocol `091fbfcf3ededbfdf9fa753b228fbf35c1f07dc330566121968a374bba1830ba`. Archive/source payloads remain under `results/native_expert_scaling/meth313_mixture_targets`; partial archives are not the complete dataset.

Router replay uses FP64 dot roundedF32 and NumPy stable sorting; source CPU uses originalF32 matrix kernel and std::sort of biased probabilities. Pinned5b335f4 `ggml/src/ggml-cpu/ops.cpp` shows comparator data[a]>data[b] without lowest-ID tie clause. This identifies two operator mismatches, without attributing the observed failing row to one unmeasured cause. Next use original GGML libraries for mul_mat/sigmoid/bias/argsort_top_k/normalization, reproducing128-token batch geometry with zero-padded unknown columns. Require every original source anchor SLOT at every coordinate, original BF16/state controls and full six-case archive gates unchanged. No row/slot tolerance or favorable subset rescue.
