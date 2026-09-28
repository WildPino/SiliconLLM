# METH-147: repeated chat structure explains much of the hash-route concentration

**Decision: diagnostic only; METH-145 remains rejected.** A final-bank replay of the same 256 raw windows and 256 chat examples splits grandchild selections into raw content, the identical 55-token chat prefix, and the remainder of each chat sequence. All 24 layers have a **100%** worst hot-parent share inside the fixed prefix. Raw windows remain comparatively balanced, with worst share 17.73%. The nominally variable chat remainder still has worst shares of **52.38–96.58%** across layers, so merely exempting the first 55 positions would not repair the training mixture.

The [protocol](METH_147_TEMPLATE_ROUTE_ATTRIBUTION_PROTOCOL_20260928.md) was committed at `ad01ef7` and the [replay](../../../benchmarks/native_expert_scaling/meth147_template_route_attribution.py) at `ac02fd9`. It binds the rejected METH-145 candidate bank SHA-256 `881a43fed3b1132a75bb4a4cfce2ada64cc13c40993248412bbadf52adf64c4f`, uses its **final** BF16 weights, and replays the exact METH-136 draws. The [result](meth147_template_route_attribution_result.json) SHA-256 is `9803731ea64fc0ffcb067852fede6c785eb76090bf9aaab819457d7bb9fefc7c`; the full per-layer/segment slot histograms are in [counts](meth147_segment_counts.npz), SHA-256 `77696ab89af95bce0b607b0d49894dd0e1bca82d5cbc9562d9b9e7c99c7c5fdb`, with exact readback.

| Segment | Input tokens | Worst hot-parent share across layers | Per-layer coverage range |
|---|---:|---:|---:|
| Raw windows | 32,512 | 17.73% | 7,898 or more |
| Identical chat prefix, positions 0–54 | 14,080 | 100% | 189 or more |
| Chat positions 55 onward | 40,015 | 96.58% | 8,262 or more |

The replayed combined route histogram has the same number of selections per layer as the saved **changing-weight** METH-145 training histogram. Their normalized L1 count differences range from 0% to 3.67%; they are close, but the final-bank replay does not replace the frozen training-time counts. One layer-6 grandchild receives 1,705 final selections, 1,536 from the identical prefix. Other top slots also receive substantial traffic after position 55.

A model-free count of `(current token, previous token, absolute input position)` in the 256 bound chat sequences finds 11 tuples recurring at least 100 times **after** position 55, totaling 2,067 of 40,015 such positions. Repeated assistant delimiters and prompt scaffolding around positions 153–162 are visible in those tuples. This is further evidence that the “variable chat” label includes repeated structural text. It does not yet attribute each hot selected parent to those exact tuples. The next diagnostic should split the chat prompt scaffold from answer continuation using the frozen training mask, and account for repeated contexts without hiding their total CPU traffic. A new training method must define a shared/structural route or change the data mixture, then preregister load and quality gates. No METH-146 quality run follows this rejected bank.

The replay took 121.453 seconds on the local RTX 3060, with 16.500 GB final process RSS and 1.243 GB peak allocated GPU memory. No T4 was used.
