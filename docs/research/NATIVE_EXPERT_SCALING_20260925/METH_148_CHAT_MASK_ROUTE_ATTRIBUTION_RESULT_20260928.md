# METH-148: the answer-context mask still contains repeated hot routes

**Decision: a bypass of only the common 55-token prefix is insufficient.** The frozen METH-136 chat loss mask separates the final-bank METH-145 replay into raw windows, identical initial chat template, remaining prompt context and input positions that predict teacher continuation. The final segment still has worst hot-parent grandchild shares of **32.20–99.20%** across layers, above the 25% target in every layer. The remainder of the prompt reaches 98.44%; raw windows stay at or below 17.73%. All segment histograms merge exactly back to METH-147's previously committed counts. METH-145 remains rejected, and METH-146 remains unopened.

The [protocol](METH_148_CHAT_MASK_ROUTE_ATTRIBUTION_PROTOCOL_20260928.md) was committed at `00d3fc1`, the [replay](../../../benchmarks/native_expert_scaling/meth148_chat_mask_route_attribution.py) at `7b55ec1`, and the [result](meth148_chat_mask_route_attribution_result.json) has SHA-256 `05b6f9650a2245b009fbeb79354cdb3a280698d411c514b1f4742ffec60a5c92`. The exact [segment counts](meth148_mask_segment_counts.npz), SHA-256 `a9f5ef08cbeef6aa23e028ee90b7af5c67635ac0958250f34a41912b3d165a81`, verify byte-for-byte readback and merge into the METH-147 histogram. This is a replay at final weights, not the changing-weight training histogram.

| Frozen mask partition | Input tokens | Worst hot-parent share across layers | Minimum per-layer coverage |
|---|---:|---:|---:|
| Raw | 32,512 | 17.73% | 7,898 |
| Identical chat positions 0–54 | 14,080 | 100% | 189 |
| Remaining prompt input | 26,306 | 98.44% | 7,233 |
| Input predicting continuation | 13,709 | 99.20% | 5,185 |

The loss-mask boundary includes the last prompt token that predicts the first continuation token. A model-free count of `(current token, previous token, input position)` among these 13,709 answer-context positions finds five tuples recurring at least 100 times, totaling 846 positions; the most frequent appears 196 times. These correspond to repeated opening scaffolding around positions 158–162 in the bound chat inputs. This supports a mechanism for the remaining hot routes, but the aggregate per-layer histogram does not map each hot parent to an individual tuple. A future route should identify highly repeated causal contexts, account for their full CPU traffic, and test content-route balance and quality separately. It cannot promote the existing bank by retrospectively excluding these inputs.

The local RTX 3060 replay took 122.093 seconds, with 16.505 GB final process RSS and 1.243 GB peak allocated GPU memory. No T4 was used.
