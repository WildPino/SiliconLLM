# METH-37: quality of the stored int8 router when it actually routes

METH-36 passes a route-recall gate while retaining the original
exhaustive route during every model forward. This cell replaces
the E128 model's route with that stored rank-64 int8 sketch and
64-candidate exact rescore. It asks whether the retrieval misses
change document loss, next-token ranking or free generation.
The original R8 core, E128 factors, fp32 fine router, top-4 and
factor-0.50 amplitude stay fixed. No new experts are trained.

Bind Qwen2.5-0.5B source SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
stored router sketch SHA-256
`133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`,
and METH-19 document manifest SHA-256
`1eb4193e521df4d25d6367a84d8478d4ea733841fe9a77d550de01796ddf3d70`.
The exact-route control uses METH-25 result SHA-256
`025c333a9edb4dcf73f6e421f91cdf54ead8cd11351e3fa17d8a6956d7e56a08`.
The METH-19 documents predate this index and are source-ID disjoint
from METH-17 and METH-25/27. They were previously used to score
the parent amplitude, so this is independent of *router-index
selection*, not a fresh evaluation of the complete parent method.

Select eight documents per category from METH-19 by sorting
SHA-256 of `meth37-37037|source_id`; retain their full selected
text spans for BPB and use their first 256 tokenizer IDs as
generation/ranking prompts. Freeze source IDs, text and prompt
hashes in a manifest before scoring. Validate the exact-route
control by reproducing all 24 saved METH-25 R8+adapter top-1
streams exactly. Score the original BF16 donor, exact-route
stored-R8+E128 model, and int8-sketch-route stored-R8+E128
model on all 24 METH-19 documents using the METH-17
EOS-prefix/512-target/512-left-context BPB procedure.
The selected [manifest](meth37_route_quality_manifest.json) was
written before scoring, SHA-256
`0c9004f36803a600deda1f884591e61a1a5ffb2b1ac0558ba75727565d30a913`.

The replacement route scores all E128 stored int8 sketches after
fp32 projection, selects 64 candidates, and scores those candidates
with the unchanged fine fp32 router. Its top-4 exact candidate
scores determine both selected expert IDs and softmax gates.
Do not use the exhaustive scores to choose or gate the replacement
arm. Run the exact and replacement model on independent trajectories.
On the 24 fixed prompts, compare 6,144 next-token top-1 IDs,
then generate up to 128 greedy tokens with each arm and record
full IDs/text and repeated-8-gram≥3×, early non-EOS and EOS counts.

**Paired route-replacement gate:** packed-minus-exact pooled BPB
≤+0.001, each category ≤+0.002, category-stratified 20,000-draw
one-sided 95% bootstrap upper ≤+0.002 (seed 373737), and
top-1 agreement ≥99%. On generation, replacement may have at
most one additional repeated-8-gram continuation and at most
one additional early non-EOS continuation per category and pooled
versus exact route. Also retain a separate donor-relative screen:
replacement-minus-original-donor BPB ≤+0.01 pooled and code,
≤+0.03 prose and technical. Report these gates separately; a
relative pass does not repair the parent's existing METH-27
absolute generation failure.

The result may justify a C CPU route implementation only if the
paired gate passes. It does not establish large-E route quality,
native rate or complete useful-model promotion. One local RTX
3060 run; stop at 20 minutes wall time, 10.5 GiB allocated GPU
memory and 20 GiB RSS. No T4 job is planned.
