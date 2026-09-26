# METH-30: learned coarse gate for the trained E128 router

METH-29 rejects the *weight-only mean-centroid* index: 64/128
candidates include only 85.848% of exact top-4 expert IDs on real
Qwen inputs. This cell changes one coordinate: fit a small nonlinear
coarse gate to predict which of the **same frozen 16 balanced groups**
contain the flat router's exact top-4. Expert factors, flat fine
router, R8 core, group assignments, top-4 and evaluation prompts
remain fixed. The gate is an index distillation, not donor/adapter
retraining or an E>128 learned-capacity demonstration.

Verify the Qwen source, tokenizer, R8 core and E128 adapter identities
as in METH-29. Freeze the METH-29 group assignments by verifying its
machine result SHA-256
`249836830ed41e77b9e2c6036292e71838359221e430eac29295e6bb7944d502`.
The source training IDs are
`benchmarks/donor_adaptation/s1/results/h0/h0_train.npz`, file SHA-256
`0cdaa28f405c3a8c6a8589af8e88788b33131bc7d4f1096da6c1fedf78caa9d9`,
array SHA-256
`9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da`.
Rank its 31,250 row IDs by SHA-256 of `meth30-30030|row_id`.
Use the first **32 different rows** for gate training and next **8**
for internal validation, taking their first 256 IDs. No METH-25/27
document or PIQA item enters gate training.
The frozen row manifest is
`meth30_learned_coarse_router_manifest.json`, SHA-256
`07d14a2296048a12d821e26b6d98635d410b51224c48a914fe19001b20407108`.
The final external route
audit uses the unchanged 24 METH-27 prompts and manifest SHA-256
`94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6`.
Those prompts were already inspected in METH-29, so a pass is a
diagnostic only; promotion requires another document set.

On the exact R8+adapter teacher trajectory, capture the fp32 input
to each of 24 residual routers at all 256 positions. The 16-bit
target for each position/layer marks groups containing any of the
flat router's top-4 IDs. Fit one independent coarse two-layer MLP
per layer: input D896, 64 SiLU hidden units, 16 group outputs.
Train all layer gates jointly for exactly **20 epochs** with
minibatches of 512 positions, AdamW learning rate 0.003 and zero
weight decay, binary cross-entropy with logits, torch seed 303030.
Normalize each layer's input by the training-set per-feature mean
and one scalar RMS; freeze those statistics for validation and
external prompts. The core, fine router and experts are frozen.
No epoch selection or hyperparameter adjustment based on validation
or external scores is allowed; use epoch 20. Record training loss,
internal validation and external scores.

At inference, score all 16 group logits, choose 4 or 8 groups,
exact-rescore their 32 or 64 original fp32 expert rows, and compare
against the exhaustive flat top-4 on the exact model trajectory.
Use METH-29's same 147,456 external input-layer cases and gate:
≥99.9% exact selected-ID inclusion pooled, ≥99% full-set match,
and ≥99% ID inclusion in each code/prose/technical category.
Report missed selected gate mass and per-layer minima. The gate's
fp32 parameter count is `24×(896×64 + 64 + 64×16 + 16) = 1,402,752`;
at E128, multiply counts before nonlinear/dispatch work are
`64×896 + 64×16 + 32×896` or the same with 64 candidates,
equivalent to about 97.14 or 129.14 D896 row dots against 128
exhaustive. This pilot does **not** claim E128 CPU speedup. At large E, group count, group
size, selected candidate rows, packed bytes and measured CPU cost
must be repriced; no 100B rate follows from this probe.

Stop at 25 minutes wall time, 10.5 GiB allocated GPU memory or
20 GiB RSS on local RTX 3060. A numerical pass permits a new
independent route/quality test and C kernel benchmark. A fail
rejects this fixed distillation recipe, not all trained hierarchies.
No T4 requested.
