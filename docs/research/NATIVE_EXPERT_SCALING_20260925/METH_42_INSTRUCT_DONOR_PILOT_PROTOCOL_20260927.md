# METH-42: generation suitability of a same-geometry Instruct donor

**Uncertainty.** METH-27's raw-document continuation screen
loops for 19/24 original Qwen2.5-0.5B base-donor prompts and
16/24 stored R8+E128 prompts. METH-41's independent E128 C96
route preserves relative quality but still loops on 15/24 raw
continuations. Decide whether an instruction-tuned donor of the
same geometry supplies a more useful starting capability and
whether the existing base-trained E128 residual adapter can
transfer without immediate retraining. This is a donor/adapter
feasibility pilot; it cannot promote a native artifact or infer
large-E quality or CPU rate.

Pin base donor Qwen2.5-0.5B SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
The Instruct donor is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, local
`model.safetensors` SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
(988,097,824 bytes). It has L24/D896/intermediate4864,
14 attention heads, 2 KV heads, tied embeddings and the same
tokenizer fingerprint `4efeeb9382a77a06` as base. The
base-trained E128 adapter SHA-256 is
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`.
It is **not** presumed to transfer.

The [prompt manifest](meth42_instruct_prompt_manifest.json),
SHA-256 `09813338debb3ba962c7d2108f743e6adc4f5e6c076542c49b837eacd39858fa`,
was frozen before model scoring. Its [builder](../../../benchmarks/donor_adaptation/s1/meth42_instruct_prompt_manifest.py)
uses the already selected METH-41 8/8/8 code/prose/technical
documents (source manifest SHA-256
`8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f`),
the first 384 characters of each selected span and the pinned
Instruct chat template. It stores the exact 24 prompt ID
sequences and hashes, up to 203 input IDs. All model arms receive
identical prompt IDs. These documents were used for C96 routing
quality, so the pilot is diagnostic for **donor choice**, not an
independent final adapter-quality gate.

Compare three BF16 arms with the same 128-token greedy limit:
base donor, Instruct donor, and Instruct donor with the existing
base-trained E128/top-4/0.50 adapter attached to all 24 layers.
The graft uses the saved adapter weights and exact fp32 router;
there is no packed R8 core or C96 shortlist in this pilot.
Save all generated IDs and decoded text. Count repeated 8-gram
≥3×, EOS, early non-EOS under 16 tokens and distinct-2 pooled
and by category. Also score the 24 METH-41 documents with the
METH-17 EOS-prefix/512-target/512-left-context BPB rule for
the two Instruct arms and compare their prompt-position
next-token top-1 IDs. Inspect the saved responses for actual
summary relevance; mechanical repetition is necessary but
not sufficient to claim usefulness.

**Prespecified decision bounds.** The Instruct donor is worth
testing as a new transfer anchor if it has ≤4/24 repeated-8-gram
continuations pooled and ≤2/8 in code, with no early non-EOS
outputs. The graft is a candidate for a new independent audit
only if the donor passes that screen; graft-minus-Instruct
document BPB ≤+0.01 pooled and ≤+0.02 in every category;
graft-versus-Instruct prompt top-1 agreement ≥95%; and at most
one additional repeated-8-gram or early non-EOS continuation
pooled and in each category. Record the base arm as a common-ID
diagnostic. If Instruct itself fails the screen, this donor/prompt
pair does not fix generation. If the graft fails, initialize
new residual experts from the Instruct donor or adapt jointly;
do not promote base-trained factors across donors from shape
compatibility alone.

One local RTX 3060 scoring run with ≤20 minutes wall time after
model loading begins, ≤10.5 GiB peak allocated GPU memory and
≤20 GiB RSS. No T4. The source download is a pinned 988 MB
one-time local asset; stop on hash or shape mismatch. A passing
pilot still needs new training or independent data, task and
generation validation, a packed artifact, native `engine.c`
parity/rate and learned large-E evidence.
