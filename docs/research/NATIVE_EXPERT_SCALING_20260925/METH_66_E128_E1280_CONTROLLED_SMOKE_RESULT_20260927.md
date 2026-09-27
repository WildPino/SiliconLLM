# METH-66: 10× expert count improves the matched smoke, but both arms fail

**Decision:** stop both 16-update arms at the precommitted donor-position
top-1 gate. E1280 is 38/3,810 positions closer to the donor than the
matched E128 control, but both fall below 95%. The favorable held-out
raw BPB and widespread B-slot changes cannot substitute for that
failed quality screen. Neither checkpoint is eligible for the planned
longer continuation or an external semantic audit under this protocol.

The [protocol](METH_66_E128_E1280_CONTROLLED_SMOKE_PROTOCOL_20260927.md)
and [new 24-prompt manifest](meth66_e1280_dev_manifest.json), SHA-256
`56cb7987f344f4a7cc8f108a9aa17fe5ed481174aebab8f8eeb0c3a115d3ed09`,
were committed before training. The
[runner](../../../benchmarks/donor_adaptation/s1/meth66_e128_e1280_smoke.py)
uses the exact Instruct donor, the same training-example RNG schedule,
four raw/chat microbatches per update, CPU-selected factors, sparse
row-wise Adam and GPU product-key AdamW for both grids. The E128
[raw result](meth66_e128_smoke_result.json), SHA-256
`acfea430af757b48e4ce733829ffc560360df5baa4b4233bad0a9f5a1d615b41`,
was committed before the E1280 run. The E1280
[raw result](meth66_e1280_smoke_result.json), SHA-256
`1e46310790b578e68088632cab8d745be42670fbde02182d008fa187d14cf0d7`,
records its independent outcome.

| Frozen terminal measure | E128 | E1280 |
|---|---:|---:|
| Initial donor/student raw ΔBPB | 0 | 0 |
| After 16 updates raw ΔBPB | −0.001879 | −0.002020 |
| Donor prompt top-1 agreement | 3,577/3,810 = 93.885% | 3,615/3,810 = 94.882% |
| Minimum changed B slots/layer | 123/128 | 846/1,280 |
| Minimum selected slots/layer on dev prompts | 112/128 | 538/1,280 |
| Worst layer maximum/mean route load | 22.02× | 71.06× |
| Peak allocated GPU | 3.661 GB | 3.662 GB |
| End process RSS | 3.264 GB | 7.784 GB |
| Elapsed including checkpoint save/hash | 129.407 s | 304.250 s |

The E128 checkpoint is 546,123,270 bytes, SHA-256
`07ce42a3edeb7ed99fd8775ef7324bc986b29ea46885c41cb7fd3341b4b603c3`.
The E1280 checkpoint is 4,423,942,208 bytes, SHA-256
`50b031e03dc289d64402dde0931a086b7e14539c41614716c6c7fe699f3a29f8`.
Both local checkpoints include CPU factors, sparse optimizer moments,
product keys, router optimizer and RNG state. They are retained for
diagnosis, not promoted. Product-key top-four matched exhaustive
pair scoring throughout the evaluated inputs.

More slots changed at E1280, but changed weights alone do not prove
distinct learned useful behavior. Its 71× worst load ratio and only
538/1,280 minimum selected slots on the development prompts show that
route coverage remains a concern. This comparison has one seed, a
16-update horizon and one donor family/scale. It cannot establish a
10× quality scaling law. The sparse optimizer and initialization differ
from the earlier METH-55 E128 training recipe, so this E128 control
does not replace the independently quality-valid METH-57 checkpoint.
The METH-66 prompts are now viewed; a new continuation rule and new
development/external sources are required for any subsequent candidate.
