# METH-14: distinguish route error from hard-carve loss

**Prospective CPU diagnostic.** METH-13's Qwen2.5-0.5B E128/top-32
input-only router passed its local mass gate, but the all-layer sparse
model lost +1.360474 BPB. This cell changes **only the route choice**
to a deliberately non-deployable local-activation oracle. It asks whether
better group selection alone could plausibly save the same donor-channel
geometry before designing training. It is not a new quality candidate.

## Bound source, artifacts and data

- Donor: `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`, local
  safetensors SHA-256
  `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
  tokenizer fingerprint `4efeeb9382a77a06`.
- METH-13's 24 E128 balanced partitions, 38 neurons per group,
  `results/native_expert_scaling/meth13_qwen05b/meth13_labels_E128.npz`,
  SHA-256
  `1c6840399ace72afee3e7200ad21a5a1519b12a688545d5035143168d8c637b1`.
  The fitted routers are bound at
  `results/native_expert_scaling/meth13_qwen05b/meth13_routers_E128.npz`,
  SHA-256
  `55f01e1d5c8c5ef9df5db8a813036d48d889f4ad9286046ca6fefb191e509518`.
- Same internal diagnostic slice as METH-13: `heldout` 2×512,
  seed 1234, token-ID SHA-256
  `3808624ff6fc5cb8fa5911c9fbfb37a8bcbf4f08db83bb5b146887c267a9e62c`,
  4,396 scored bytes. This slice is **not fresh** and will not be
  used to tune a route objective or claim final generalization.

## Fixed arms and decision

Load one CPU fp32/eager donor with six PyTorch threads. Score paired BPB
on identical IDs in this order: intact donor, METH-13 fitted hard top-32,
then local-mass oracle hard top-32. The fitted arm must reproduce
METH-13's donor **1.083022** and fitted **2.443496** BPB each within
1e-4; otherwise invalidate the diagnostic. Also verify the donor versus
all-groups wrapper on the first 16 logits within 1e-4 and verify that
both sparse routes change logits.

For each FFN at its **current input trajectory**, calculate full
post-SwiGLU `z`, sum `z²` within each of the 128 fixed groups, and choose
the 32 groups with highest mass (stable lower-ID tie). Zero `z` outside
those groups before the donor `down_proj`. This oracle uses information
that a real sparse inference path would have to compute densely; its
time is not a routing/LUT speed estimate. It maximizes retained local
activation mass, not whole-model likelihood, so its BPB is not a formal
upper bound on every possible learned router.

If oracle donor-relative loss is **≤+0.40 BPB** while fitted loss is
>+0.40, prioritize a changed router/training objective. If oracle loss
is **>+0.40**, stop router-only repair of this particular E128/top-32
donor-channel geometry and design a different shared/conditional path.
Either outcome does not validate joint expert learning, scaling to 10B
or 100B, a CPU LUT, or a native export.

Runtime stop: 15 minutes or 20 GiB process RSS; CPU only, no GPU.
Save exact metrics, route arms, source/artifact/data hashes, runtime and
process RSS. No parameter/threshold changes after observing the result.
