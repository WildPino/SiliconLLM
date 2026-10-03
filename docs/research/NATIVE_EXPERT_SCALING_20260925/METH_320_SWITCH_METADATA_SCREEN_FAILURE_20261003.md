# METH-320: two metadata cases complete, legacy large config assertion fails

Freeze `bef9b80`, exit1 after7.328s in measured main (imports precede timer).
Stage google/switch-large-128_meta_graph_and_ledger. No weights downloaded.
Full three-case screen incomplete; no global promotion.

Both base128 and base256 complete official config/index namespace/size/META
controls. Inferred unique parameters7,415,217,408 /14,664,154,368. Complete
decoder F32 descriptors497,536,512 /499,895,808bytes; BF16 scenarios250,005,504
/252,364,800bytes, W8 scenarios126,774,020 /129,133,316bytes. Same123,764,736
large-matrix decode coefficients; router grows with real expert count.
All counts remain INFERRED shapes/metadata, not physical tensor/unique-function
verification, source quality or native speed.

Large128 official legacy config omits num_selected_experts. Installed5.13.1
SwitchTransformersConfig does not define that absent attribute;320 incorrectly
assumes it exists before constructing the META graph. This is apparatus/config
compatibility, not a failed top1 architecture or measured large128 cost.
Original config and all six downloaded config/index files remain preserved.
Installed built-in SwitchTransformersTop1Router forward uses max+one_hot;
every actual META router module can verify that class without inserting a
missing field into the source config or accepting top2 routing.

A NEW repaired screen must retain original configs/revisions/index names/
cost rules/gates, verify all constructed routers are that EXACT top1 class,
respect any explicit num_selected_experts field==1, and reproduce both
completed base-case ledgers EXACTLY. No new weight/model download or favorable
case subset. Large128 remains unsupported until its own checks complete.

[Preserved partial record](meth320_switch_metadata_screen_result.failure.json)
SHA `f5cfa26e823e1369765795259cb0c34c300237344976a7b01a51508a43a580fd`.
Built-in model sourceSHAf1eab2d3e02e70d93d20d6fb07dd3acd20519fe6f88a57d7a1b214bc5b305ecd;
config sourceSHA4a8446dcc0e5b95388f51532be67d012895b16c77fc425335ea4f0205c80744f.
Asset files under `results/native_expert_scaling/meth320_switch_metadata`.
No live process after terminal exit. Useful full pretrained-transfer goal stays open.
