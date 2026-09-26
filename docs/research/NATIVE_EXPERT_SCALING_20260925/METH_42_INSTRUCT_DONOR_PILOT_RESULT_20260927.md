# METH-42: Instruct donor repairs chat-format loops; direct adapter graft fails ranking

The [prospective protocol](METH_42_INSTRUCT_DONOR_PILOT_PROTOCOL_20260927.md)
compared Qwen2.5-0.5B base, same-geometry Qwen2.5-0.5B-Instruct
and Instruct with the already trained base-donor E128 residual
adapter attached. All three BF16 arms used exactly the same 24
chat-format prompt ID sequences from the frozen
[manifest](meth42_instruct_prompt_manifest.json), SHA-256
`09813338debb3ba962c7d2108f743e6adc4f5e6c076542c49b837eacd39858fa`.
The prompts use the existing METH-41 8/8/8 code/prose/technical
sources, so this is a donor-choice diagnostic, not an independent
final adapter-quality test. The Instruct revision was
`7ae557604adf67be50417f59c2c2f167def9a775`; its 988,097,824-byte
weight file SHA-256 is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
It matches the base model's L24/D896 architecture and tokenizer
fingerprint. The graft used unchanged METH-19 base-trained
rank-8/top-4/E128/0.50 factors and exact fp32 routing; no
new training, R8 core, C96 shortlist or C engine was involved.

| Chat-format greedy generation, 128-token cap | Base BF16 | Instruct BF16 | Instruct + base adapter |
|---|---:|---:|---:|
| Repeated 8-gram ≥3×, pooled | 7/24 | **0/24** | **0/24** |
| Repeated 8-gram ≥3×, code | 3/8 | **0/8** | **0/8** |
| EOS terminated | 0/24 | **24/24** | 23/24 |
| Mean distinct-2 | 0.735 | 0.924 | 0.913 |

Neither Instruct arm had an early non-EOS continuation under 16
tokens. The Instruct donor passes the predeclared ≤4 pooled/≤2
code repetition screen. The base arm received identical chat ID
inputs, so its 7→0 difference with Instruct isolates a donor
effect within this protocol. METH-27's 19/24 base loop count
used raw document continuation prompts; its difference from
7/24 here also reflects the changed task/input format.

| Document category | Instruct BPB | Instruct + graft BPB | Graft − Instruct |
|---|---:|---:|---:|
| Code | 0.683083 | 0.690855 | +0.007772 |
| Prose | 1.116857 | 1.109956 | −0.006901 |
| Technical | 1.657486 | 1.664778 | +0.007292 |
| Pooled | 1.152475 | 1.155196 | **+0.002721** |

The graft passes the predeclared document-loss and relative
repetition limits, but prompt-position top-1 agreement is
**3,409/4,045 = 84.277%**, below the fixed ≥95% gate.
The large ranking change means neither near-flat pooled BPB nor
zero repetition licenses direct transfer of base-trained
factors to the Instruct donor. The graft candidate gate fails.

The 24 saved Instruct responses were inspected. Many identify
their excerpt and include a concrete detail, but factual errors
remain: for example, an excerpt where `RSS` denotes process
memory was summarized as involving RSS feeds. This pilot's
automatic repetition gate is a behavior screen, not a task
accuracy or semantic-retention test. The base and Instruct
families cannot be called equally useful based only on loop
counts; an independent task audit is required.

The [raw result](meth42_instruct_donor_pilot_result.json), SHA-256
`8953a880c364d3b25f01b8c381ae431f1627df0562e060b2caee165f587571ec`,
retains every continuation ID/text, per-document nats and
prompt top-1 streams. A separate readback verified the prompt
IDs against the manifest, all BPB, repetition counts and
3,409 matches. Reproduction command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth42_instruct_donor_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth42_instruct_donor_pilot_result.json
```

The local RTX 3060 scoring phase took **281.45 s**, peaked at
**2.437 GB** allocated GPU memory and ended at **2.740 GB**
RSS, within the frozen limits. The pinned model download was
988 MB; no T4 was used.

**Decision:** select same-geometry Instruct as a candidate
donor for a new zero-initialized residual-expert adaptation,
because it passes the chat-format generation screen on this
pilot. Reject the direct base-trained adapter graft under its
prospective ranking gate. The next method experiment should
start exactly at Instruct donor logits, train new expert/router
weights while preserving instruction behavior, and test on
new documents, independent instruction/task prompts and
generation. It must still meet packed export, scalable routing,
native `engine.c` parity and ≥50 accepted tok/s on the same
quality-valid artifact; none is established here.
