# METH-83: grouped-Q4 FFN core fails fresh composed quality

**Decision.** Stop the METH-82 BF16-head+attention/grouped-Q4-FFN core.
Its stored format and ideal payload pass, but its composition with
the METH-56 E128 learned adapter fails every precommitted fresh-source
document and prompt gate. The severe prompt-ranking loss also appears
with experts disabled, localizing the dominant failure to the core
conversion under this test. Do not proceed to generation, task,
blind semantic review or native `engine.c` integration of this map.

The [protocol](METH_83_Q4_CORE_COMPOSITION_PROTOCOL_20260927.md) and
[24-document manifest](meth83_q4_core_dev_manifest.json) were frozen
at `edbb979` before inference; manifest SHA-256 is
`3ecc2193d8c5436b73a43fd959a99e6b8551f48d92392bbc92dad992a81d85fb`.
Its 8 code, 8 PG19 prose and 8 technical/general source IDs and
fragments exclude earlier selections through METH-72. The
[evaluator](../../../benchmarks/donor_adaptation/s1/meth83_q4_core_composition.py)
was committed at `6d1f897` before scoring. It checked the bound donor,
METH-56 adapter and METH-82 artifact hashes, reread the 72 saved FFN
code/scale tensors, and compared BF16 donor, BF16+E128, Q4 donor and
Q4+E128 on identical documents and prompts. The
[machine result](meth83_q4_core_composition_result.json) contains
per-source nats and prompt agreement; staged partial rows were saved
during execution.

| Fixed fresh-source measure | BF16 donor | BF16+E128 | Q4 donor | Q4+E128 |
|---|---:|---:|---:|---:|
| Pooled document BPB, 98,280 bytes | 1.219962 | 1.215112 | 1.271708 | 1.265070 |
| Prompt top-1 versus BF16 donor, 4,199 positions | 4,199 | 4,045 = 96.332% | 2,847 = 67.802% | 2,875 = 68.469% |

Q4+E128's pooled ΔBPB is **+0.045108**, above the registered +0.02
limit. Each category also exceeds its +0.04 limit: code +0.045254,
prose +0.045217 and technical/general +0.044853. Its category prompt
agreements are 71.016%, 69.984% and 64.694%, all below 90%. It is
27.864 percentage points below the BF16+E128 prompt agreement, above
the allowed one-point loss. Thus all five grouped gates fail.

The BF16+E128 control still improves pooled document BPB by 0.004850
and retains 96.332% prompt top-1 on these untouched sources, so the
set does not by itself explain the Q4 failure. Q4 donor and Q4+E128
are close on both endpoints, indicating that the existing adapter
does not compensate for this FFN quantization. This is not a general
rejection of grouped low-bit FFN representations: this fixed symmetric
group-64/FP16-scale rule, on one 0.5B donor, is the tested failure.

The run completed in 37.813 seconds after device initialization,
peaked at 2.443 GB allocated GPU and 3.161 GB RSS. No T4 was used.
The packed export remains a reproducible artifact and a useful
negative control. The next core candidate requires an explicit
repair mechanism or different representation chosen using separate
development evidence, followed by new source-disjoint quality and
actual C traffic/rate measurements. The large-E distinct-expert
retention and utility gates remain open independently.
