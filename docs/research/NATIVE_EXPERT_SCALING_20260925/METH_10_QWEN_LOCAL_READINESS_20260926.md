# METH-10 readiness: tractable dense-source conditional transfer branch

**Status: source and historical training assets present; no new model
trained or scored in this record.** The GigaChat METH-06–09 sequence
measured a compact low-bit quality failure and separated a small Q2_K
expert benefit from a larger head/dense precision payment. A different
mechanism is required. The local Qwen2.5-1.5B branch is the tractable
existing signal for **jointly trained** conditional structure; it is
neither a 10B result nor a substitute for GigaChat's large-scale and
native C gates.

## Verified local anchors

The pinned Hugging Face snapshot is `Qwen/Qwen2.5-1.5B` revision
`8faed761d45a263340a0528343f099c05c9a4323` under the local
Hugging Face cache. The snapshot entries are symbolic links; the
resolved blob sizes and SHA-256 values, rather than the link's apparent
zero-byte size in PowerShell, were verified:

| Resolved source asset | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 3,087,467,144 | `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796` |
| `config.json` | 684 | `0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77` |
| `tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |

The config declares `Qwen2ForCausalLM`, 28 layers, hidden width 1536,
FFN intermediate width 8960, 12 attention heads, two KV heads and
151,936 vocabulary rows. This is a dense donor, unlike the 64-expert
GigaChat source. The local
`benchmarks/donor_adaptation/s1/_h1_bundle/h1_trained_s2.npz` is
1,334,711,974 bytes, SHA-256
`4030d2af63aed924d7e17559bc3dc84ba8db519056c389cdb068b38b58a7e3b3`,
matching the [H1 bundle manifest](../../../benchmarks/donor_adaptation/s1/_h1_bundle/MANIFEST.json).
That bundle also contains the frozen H0 factors, E256 labels, activation
statistics, train/calibration and heldout ID files recorded in the manifest.
The **later H1 session 3** checkpoint is also present at
`D:/_ktmp/h1_kaggle_s3_v2_output/h1_trained_s3.npz`,
1,334,711,974 bytes, SHA-256
`4d6d761c5f1fd3f8adcb4cfc8a6aae44c4b6ca3bc681d8a3e98397caf7c5ce4a`.
Its identity matches the [frozen S3 CPU adjudication](../../../benchmarks/donor_adaptation/s1/results/h1/h1_eval_h1_s3_adjudication.json).
The hashes above establish local identity and availability; they do not
validate a new conversion.

## What earlier experiments establish

[H1](../donor_adaptation/probes/H1_THE_CARVE_TRAINED.md) jointly trained
carved FFN masters and routers on eight of 28 layers, with E256/top-16.
The applied eight-layer carve scored 1.096636 BPB; the intact donor was
0.767595. The historical S2 checkpoint scored 0.962593 BPB on the frozen
CPU-fp32 slice. The [later S3 evaluation](../../../benchmarks/donor_adaptation/s1/results/h1/h1_eval_h1_s3.json)
scored **0.923490 BPB** on that same slice, still **+0.155895 BPB**
versus donor. Its [adjudication](../../../benchmarks/donor_adaptation/s1/results/h1/h1_eval_h1_s3_adjudication.json)
records 765 completed steps, a time cap after 39,619.6 s (about 11 h)
and a valid CPU-fp32 gate. H1 has no native `engine.c` artifact and
does not establish quality retention. Adam state was restarted between
sessions, so their increments do not support a smooth learning curve
extrapolation. [H5](../donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md)
installed separately trained H4 attention and H2I FFN components
post hoc; BPB worsened by +0.245430 and free generation deteriorated.
That result rejects the tested frozen composition, not fresh joint
training. These histories make a new jointly specified geometry and
step-zero control necessary before another expensive run.

## Decision boundary for the next cell

The next substantive cell should first specify one conditional Qwen
geometry with distinct expert weights, source-to-target tensor mapping,
router selection, precision and projected stored/active bytes for a
native export. Run a step-zero CPU-fp32 donor-relative control and a
bounded pilot before training; require the exact scorer, source IDs and
heldout split to be fixed in advance. Then estimate optimizer memory,
token exposure per expert, runtime and a training stop on available
hardware. Reusing the H1 frozen heldout alone would not establish fresh
generalization. Any GPU/T4 run needs a stated reason, budget and stop
criterion before dispatch. A passing Qwen 1.5B proof of concept would
still need larger donor families, useful expert-count scaling with RAM,
CPU LUT/router measurements and quality plus ≥50 accepted tok/s on the
same native C artifact.
