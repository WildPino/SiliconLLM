# METH-56: product-key E128 retention survives 512 updates

The [frozen protocol](METH_56_PRODUCT_KEY_RETENTION_PROTOCOL_20260927.md)
resumes the exact METH-55 joint product-key/E128 checkpoint with
stronger donor KL (raw 2.0, full chat 4.0), factor LR 1e-4 and router
LR 1e-5. The [new 24-prompt manifest](meth56_product_key_dev_manifest.json),
seed 565656, excludes the 256 chat-training rows and all 72 earlier
METH-44/47/55 development rows. These plus the new 24 rows are
excluded from raw sampling. The bound METH-55 donor/student raw BPB
values reproduce at resume.

| Update | Chat top-1 vs donor | Held-out raw ΔBPB | Min selected pairs/layer | Worst max/mean load |
|---:|---:|---:|---:|---:|
| 16 resume | 3,668/3,810 = 96.273% | −0.001552 | 98/128 | 22.55× |
| 256 | 3,740/3,810 = **98.163%** | −0.013933 | 110/128 | 14.04× |
| 512 | 3,689/3,810 = **96.824%** | −0.014485 | 112/128 | 14.77× |

The update-256 gate passes, so training continues to 512. The terminal
chat agreement drops 1.339 percentage points from the peak at 256
but remains above the predeclared 95% gate. All 128 B slots per layer
have changed by 512. The 16-pair search agrees with exhaustive E128
pair scoring on actual evaluation hidden states throughout. Route
load becomes less concentrated than at resume but remains uneven;
the terminal worst expert receives 14.77× the layer mean. These are
observations on the new development prompts, not proof that more
experts will retain quality.

**Decision:** the frozen terminal gates pass, making the **update-512**
checkpoint eligible for a separately frozen external audit. The
checkpoint was selected by the stated terminal rule; the better
update-256 ranking is reported but not used to select a model here.
External source-grounded generations and task retention must resolve
semantic quality before native promotion or scaling the expert count.

The [result JSON](meth56_product_key_retention_result.json) SHA-256 is
`06aa8934d849fc3bbe2ae2fb85c5ab4d302c9f3f1c5922138604bf1347273a39`.
The local update-256 and update-512 checkpoints are each 545,529,924
bytes, with SHA-256
`8f950d764959705fd055c6b6de3cf6b294fc8ba443a5b3e2a7c31f25e0b861a3`
and
`8371262a1461a44fcedf12d129059d8b8ab1fd9860e032df8661a30eff187072`
respectively. The [runner](../../../benchmarks/donor_adaptation/s1/meth56_product_key_retention.py)
SHA-256 is
`4b6a4d408b7f356cf49770f8c04a24e04035ffad00d13154237c90512ae9a1d8`.
Reproduce with:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth56_product_key_dev_manifest.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth56_product_key_dev_manifest.json
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth56_product_key_retention.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth56_product_key_retention_result.json --checkpoint-dir benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth56_checkpoints
```

Elapsed 779.84 s on local RTX 3060, peak allocated GPU
2,561,858,048 bytes and end RSS 3,910,180,864 bytes; both below
the frozen limits. No T4 was used. This is still a BF16 donor wrapper
at E128, without `engine.c` parity, ternary LUT factor quality,
distinct large-E experts or a same-artifact ≥50 accepted tok/s result.
