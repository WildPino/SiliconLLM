# Large-family selection screen: 20B to120B with a protected-core falsifier

15 October2026. **Candidate family for falsification, not an admitted solution.**
Native MoE satisfies the owner's corrected scope. A successful small Granite
kernel is not the very-large-family result.

The pinned official configurations for
[20B](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)
and [120B](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json)
shareD2880/FF2880, attention geometry, vocabulary201088 and top4. Layers grow
24->36 and experts32->128: expert capacity grows6x, attention/state1.5x,
head stays fixed. This is a useful family test. These are configuration-derived
counts, not complete tensor-header/value audits. Raw metadata and exact hashes
are preserved in [retrieval record](large_family_metadata/retrieval.json).

## What must become cheaper

[Recomputed ledger](large_family_metadata/analytical_screen.json) for120B:

| Work | Count/payload hypothesis |
| --- | ---: |
| Attention projections |955,514,880 coefficients |
| Vocabulary head |579,133,440 coefficients |
| Parent router |53,084,160 bytes F32 |
| Active routed expert matrices |3,583,180,800 coefficients |
| Ideal4bit attention+head, F32 router |820,408,320 bytes |
| Head-free ideal4bit attention+F32 router |530,841,600 bytes |
| Logical F16 KV reads, full declared windows at context2048 |80,216,064 bytes |
| Physical Q4_K attention with2880->3072 input padding |557,383,680 bytes |

At the favorable assumed40GB/s, the head-free ideal core+KV costs15.28ms
before experts or arithmetic. Q4_K padding/scales raise that byte-only scenario
to17.27ms. Actual CPU performance can be worse; this is not a measured DRAM
floor. Even removing the head requires roughly an order-of-magnitude expert
active-work reduction, not the small512->2x256 pilot's2x alone.

First falsifier: execute a favorable packed protected-core screen with head and
experts artificially free, using declared source geometry, padding, router and
included operations. Any omitted attention/KV work must remain explicit; a fast
partial screen cannot promote the family. A>=20ms result rejects that physical
variant.14--20ms requires another core transformation; <10ms only provides room
for a further donor-specific test. No quality or accepted-rate claim follows.
Do not acquire the whole donor or train selectors from a metadata row alone.

## Distinct transformations, each still unproven

[Retrieved vocabulary rows](PROTECTED_CORE_INTERVENTIONS.md) can avoid the dense
head scan. A full-I8 scan plus F32 shortlist rerank is not that intervention.
A separate possible core intervention is calibrated low-rank Q/O with K/V
preserved; at rank512 the attention coefficient count falls analytically to
363,331,584, but quality preservation and physical cost are unknown. Rank is
not selected from DEV, and this is not permission to declare the reduced
artifact equivalent. Strong expert selection must then preserve useful capacity
with charged child IDs and contribution masses.

The [pinned official reference](https://github.com/openai/gpt-oss/blob/5681a6b6552635e7b29cde94a61ca6a7aee90ebd/gpt_oss/torch/model.py)
uses selected-top4 softmax, clamped modified SwiGLU and expert biases. Splitting
must apply down bias once per parent; duplicating it in every child changes the
function. Source experts use MXFP4 block32, distinct from our Q4_K block256.
The synthetic Granite partition control therefore does not certify this export.

The [pinned index](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/model.safetensors.index.json)
reports65,248,815,744 bytes of tensor payload. That is not peak RSS; conversion
must keep the expert pool packed, stream chunks and bound adapters/optimizer
state. Materializing the entire pool in BF16 would exceed local RAM. No weights
were downloaded by this metadata screen. Falcon prompts/splits can be reused;
Falcon teacher scores and tokenizer IDs cannot validate this family.
