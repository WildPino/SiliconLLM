# METH-70: 10× dense expert count worsens routing and donor retention

**Decision:** reject both 16-update arms at the precommitted ≥95%
donor-position top-1 gate. With the same pretrained Instruct donor,
training examples, METH-55 dense AdamW recipe and newly frozen prompt
set, E1280 falls to 92.865% versus E128's 93.888%. The 10× expert
bank is physically trainable, but increased route choice does not
preserve quality under this rule. Neither checkpoint advances to
longer training or external semantic audit.

The [protocol](METH_70_DENSE_E128_E1280_SCALE_PROTOCOL_20260927.md)
and [new development manifest](meth70_dense_scale_dev_manifest.json),
SHA-256 `fa77b745fa63c953cb8f50963e6e6f93a9dc26d7bc8f0152c93380b4a9f91607`,
were committed before training. The [generalized router/expert module](../../../benchmarks/donor_adaptation/s1/meth70_dense_product_key.py)
reproduced METH-55 E128 A/B/router initialization and routes bit for
bit before the run. The [trainer](../../../benchmarks/donor_adaptation/s1/meth70_dense_e128_e1280_smoke.py)
used identical sample draws and objective in each arm. The E128
[raw result](meth70_dense_e128_result.json), SHA-256
`1853decfcc70a3da40cc8ddb2c54a04176ea6d83bbd73c55defb882e69434f68`,
was committed before E1280 execution. The E1280
[raw result](meth70_dense_e1280_result.json), SHA-256
`cbb56e6dcba14539d7dc214ac9fba1af49a96a8aa1bff4052f737878826d3077`,
contains its outcome.

| Frozen terminal measure | E128 | E1280 |
|---|---:|---:|
| Initial donor/student logits, raw and chat prefixes | exact | exact |
| Held-out raw ΔBPB | −0.002752 | −0.002321 |
| Donor prompt top-1, 3,812 positions | 3,579 = 93.888% | 3,540 = 92.865% |
| Minimum changed B slots/layer | 117/128 | 751/1,280 |
| Minimum selected slots/layer on dev prompts | 91/128 | 474/1,280 |
| Worst maximum/mean route load | 20.80× | 129.19× |
| Peak allocated GPU | 2.582 GB | 10.072 GB |
| Full run including checkpoint save/hash | 65.203 s | 321.625 s |

The 16 training updates themselves finish at 34.453 s for E128 and
33.343 s for E1280 on this local GPU. This does not measure inference
or native accepted-token speed; E1280's large checkpoint serialization
and hash dominate total wall time. The E128 checkpoint is 545,530,346
bytes, SHA-256
`729350de9ed0b64acdf9b2c02f14ea09c8f967848e378ac1bdd87eaeeab01402`.
The E1280 checkpoint is 5,302,756,776 bytes, SHA-256
`ab8f8004e33fb960c7af1ee634bfdec81adef7549839748dc9a55f450a7e08a5`.
Both remain local diagnostic artifacts with optimizer and RNG state.

The E128 arm also fails on this new set, showing that 16-step donor
retention is not robust across this training draw and development
selection. The E1280 arm loses another 39 donor decisions, selects
fewer than 40% of its slots in the least-covered layer on the dev
prompts, and reaches a 129× load peak. The new checkpoint has many
changed B tensors but does not yet demonstrate useful learned roles
across the much larger bank. The next training rule should directly
address route concentration while strengthening donor retention, and
must be evaluated on new disjoint material. METH-70 prompts are viewed
and cannot be reused as a promotion gate. A CPU LUT and full native
`engine.c` rate remain untested on this candidate.
