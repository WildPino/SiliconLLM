# METH-44: full-chat retention smoke result

The [protocol](METH_44_FULL_CHAT_RETENTION_PROTOCOL_20260927.md),
[new 24-prompt development manifest](meth44_instruct_fresh_chat_manifest.json)
and [runner](../../../benchmarks/donor_adaptation/s1/meth44_instruct_full_chat_smoke.py)
were committed before this run. The new prompt rows are disjoint from all
256 METH-43 teacher-chat rows and excluded from METH-44's raw samples.
Training started anew at exact Qwen2.5-0.5B-Instruct donor logits, not from
the failed METH-43 checkpoint. The full prompt plus saved response supplied
all-position KL; response tokens alone supplied chat CE.

The [measured result](meth44_instruct_full_chat_smoke_result.json), SHA-256
`437ba07965ee59b6bde5d92e1d0250b6dd5abb648fd0ad9eabed82e8ff77d300`,
records 16 finite updates. Initial raw and chat maximum logit difference
was exactly zero. Every layer had a positive output-factor gradient at
update 1 and router gradient from update 2; minimum changed output slots
per layer was 124/128. Donor heldout raw BPB remained 0.971255, and the
student reached 0.969412 (student-minus-donor **−0.001843**). New prompt
top-1 agreement was **3654/3807 = 95.981%**, above the fixed 95%
development gate. Runtime was 27.500 s, with 2.603 GB peak allocated GPU
and 3.155 GB final process RSS. The local optimizer/RNG checkpoint is
`benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth44_instruct_e128_smoke.pt`,
SHA-256 `4fade15804f5e2297ff60cafacf870e5ae11926ba886786a524110dea94f16da`.

**Decision: eligible for a preregistered longer continuation.** This only
passes a same-corpus development retention screen after 16 updates at
0.5B/E128. It does not establish response-generation quality, independent
task/document retention, useful specialization, or scale to more experts.
METH-42's viewed 24 prompts were not used to tune or gate METH-44.
The next run needs a fixed training budget and disjoint external document,
task and chat-generation evaluations before it starts. Promotion further
requires an actual quality-valid packed expert artifact, bounded large-E
router and measured `benchmarks/phase60/engine.c` accepted batch-1 rate.
