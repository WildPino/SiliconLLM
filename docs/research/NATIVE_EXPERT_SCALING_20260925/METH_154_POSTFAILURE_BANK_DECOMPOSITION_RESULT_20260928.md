# METH-154: postfailure B-bank decomposition

**Finding: the exported E12800 bank has specialist differences but almost no learned common B shift.** This is a training-artifact diagnostic after the METH-153 quality failure, not a new quality test or causal proof. The [checker](../../../benchmarks/native_expert_scaling/meth154_bank_decomposition.py) binds the METH-126 original BF16 source, METH-136 continued E1280 control, METH-152 candidate and both result files by SHA-256. It reads all 24 layers without touching the METH-153 held-out text or scoring a model. The [result](meth154_bank_decomposition_result.json) is SHA-256 `7d7d5ab76418c5cb0b38eb7fe8473baa00b856c8e2633de0a8610dd12b28cd94`.

| B-bank quantity, across 24 layers | Observed range |
|---|---:|
| Continued E1280 control minus original source, elementwise RMS | 8.26–10.53 × 10⁻⁵ |
| Candidate ten-grandchild mean minus original source, elementwise RMS | up to 1.97 × 10⁻⁶ |
| Candidate nine content grandchildren minus original source, elementwise RMS | 4.01–4.62 × 10⁻⁵ |
| Candidate content grandchildren minus continued control, elementwise RMS | 9.22–11.54 × 10⁻⁵ |
| Continued control BF16-changed rows/layer | 1,183–1,278 of 1,280 |

METH-136's control export retains its learned common B update. METH-152's candidate export explicitly subtracts each ten-grandchild mean shift and restores the *original* source B before BF16 rounding. The measured near-zero candidate mean verifies that this operation occurred, while nonzero content residuals verify that specialists changed. The at least 20× smaller candidate mean RMS than its content-specialist RMS, and much larger control-minus-source shift, support testing a shared learned B base plus conditional residuals. They do not show that preserving the base will improve held-out BPB; undertraining of individual content rows or other dynamics may also matter.

Any new training comparison must freeze its design before running, keep route and CPU traffic gates, and evaluate on a new source/fragment-disjoint corpus. The METH-153 corpus is consumed and cannot select or validate the next candidate. The previous failure remains: no useful extra expert capacity, 10B/100B transfer, full native CPU factor cost or >=50 accepted tokens/s result is established.
