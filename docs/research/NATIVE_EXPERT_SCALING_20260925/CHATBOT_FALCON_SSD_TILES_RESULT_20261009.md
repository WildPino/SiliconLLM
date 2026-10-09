# Source SSD storage: exact observed logits with lower memory

9 October2026. Repair1 COMPLETE/PASS;original pre-forward substitution failure
remains [retained](CHATBOT_FALCON_SSD_TILES_REPAIR1_PROTOCOL_20261009.md).
This qualifies source acquisition scaffolding,not the deployed engine model.

Freezeffb92001628be578248c9600526aac0dabf6361c;
[binding](chatbot_falcon_ssd_tiles_binding_repair1_20261009.json) SHA256
42b900af40cd1d3e1873daaf72f394b5e14cf83b641083bea9b9fbc8a90dc243;
[result](chatbot_falcon_ssd_tiles_result_repair1_20261009.json) SHA256
b5d6a994dc6cf1f9e674bb3a484828181ebd33bec983f8ceabf86be9dd722ffe;
[terminal](chatbot_falcon_ssd_tiles_result_repair1_20261009.terminal.json).

| Fixed saved trajectory | Input IDs | Compared full-vocabulary rows | Different BF16 coordinates | Different winners |
|---|---:|---:|---:|---:|
| broad_fit_apigen_80k_023 |362|118|0|0|
| broad_fit_apigen_80k_002 |741|33|0|0|

All151 rows/9,896,087 BF16 coordinates bit-identical;new packet hashes equal
the original saved packet hashes. Source forward calls151/GENERATIONS0;fixed
preceding saved IDs are forced through cache,no original label regeneration.
All finite/lossless bit/ID/input/resource gates PASS. Only three source SSD
temporary contractions are tiled along their existing independent chunk index;
each reduction receives its complete original operands. Chunk128/weights/
precision/core/attention/multipliers remain unchanged,source files unmodified.
Generated worker-local source method text is retained with the observations.

Actual allocated GPU peak4,622,627,328B vs original failed family10,764,644,352B;
reserved6,146,752,512B vs12,425,625,600B. These are separate actual executions:
original cached generation vs tiled forced trajectory,with two matching full
output packets;they do not isolate individual allocations or assert universal
prefill/throughput gains. Algebraic temporary extents explain the expected saving.
Unseen trajectory bit equality is not established by this two-case comparison.

Family46.563s/worker38.047s;launcher4652/worker12424,exit0/session54843 closed.
Held OS peak3,723,444,224B;5 outputs/19,803,524B.35 input hashes/resource gates
PASS. No student/update/native/T4 calls or quality/rate/useful-n admission.
[Capture repair1](CHATBOT_BROAD_CAPTURE_REPAIR1_PROTOCOL_20261009.md) may now reuse
the two original packets and generate only46 remaining cases under unchanged
caps. The original source capture GPU admission stays FALSE. Final compact
original-LUT/ternary/SSM target remains unchanged.
