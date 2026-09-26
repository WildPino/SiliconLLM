# METH-35: low-rank shortlist for the trained E128 router

**Question.** METH-29/30's fixed coarse groups omit too many exact
top-4 experts. Test whether a low-rank sketch of the *same trained
flat router* can rank a short candidate list while addressing far
fewer router coordinates per expert. This changes the index rule;
the core, adapter, fine router, factors, top-4 and model trajectory
stay fixed. The result is a route diagnostic, not a model-quality or
native-speed result.

Bind the Qwen2.5-0.5B source SHA-256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
tokenizer fingerprint `4efeeb9382a77a06`, R8 core SHA-256
`c307fa48015fd01ccc43ab6a90debe42c10245b926fc3a6ee2572afa27`,
E128 adapter SHA-256
`3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
and 24-prompt METH-27 manifest SHA-256
`94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6`.
The exact-route stream must reproduce METH-29's saved SHA-256 per
layer. These already inspected prompts supply no independent quality
evidence.

For each layer compute the deterministic full SVD of the frozen fp32
`128×896` fine-router matrix. Let `V_r` be its first `r` right
singular vectors, with `r ∈ {16,32,64}`. For a hidden input `x`,
rank all 128 experts by `(x V_r) (W V_r)^T`, take the first
`C ∈ {32,64}` expert IDs, then use original fp32 `W` to exact-score
only those candidates and return their top-4. Use the exact original
route for every model forward pass. Compare before replacing routes,
for every one of 6,144 positions and 24 layers. Record spectrum
energy, selected-ID inclusion, complete top-4 set match, missed
softmax gate mass, per-category/layer counts and stream hashes.

Six fixed arms are the Cartesian product of the three ranks and two
candidate budgets; do not tune ranks or candidate counts after seeing
the outcomes. The route gate is METH-29's: pooled inclusion ≥99.9%,
pooled full-set match ≥99%, and inclusion ≥99% in each code, prose
and technical category. A passing arm permits byte-quantized sketch
and fresh route-replacement quality tests, followed by a C CPU cost
test; it does not itself promote an index. A failing arm rejects only
this untrained SVD sketch at the tested budgets.

The float arithmetic for a rank-r scan plus exact C-row rescore is
`r×896 + 128×r + C×896` multiply terms per layer at E128.
At the hypothetical E273,547 point, a one-byte *stored* rank-r
sketch would address `24×273547×r` code bytes/token plus scales
and exact candidate weights. This byte projection is distinct from
the present fp32 GPU diagnostic; neither 10B/100B quality nor CPU
speed is inferred from it. Report these costs for each arm.

One local RTX 3060 run, capped at 20 minutes wall time, 10.5 GiB
allocated GPU memory and 20 GiB RSS. Stop if model or hash controls
fail. No T4 job is planned.
