# METH-184: content-score plus hash-noise third-tier load screen

**Question and prior evidence.** METH-183 shows functional alignment of
the learned E1280 content route. METH-136's frozen random-key third tier
then updated B only and developed severe E12800 load concentration;
METH-139–141's fitted scalar deciles also failed source and context
shift. The balanced hash route of METH-150/175 kept load but did not
produce useful route-specific function. Test whether a deterministic
ten-way rule can retain measurable dependence on the METH-135 content
scores while bounding traffic across sources. This is a route-load and
content-dependence screen before any B training, not a quality test.

**Frozen source and rule.** Use the exact METH-126 E1280 centered BF16
factor bank and METH-135 32-dimensional third projection/ten keys,
with their hashes checked by `meth136_matched_sparse_train.py`. Run the
same fixed BF16 Qwen2.5-0.5B-Instruct E1280 control model, first two
routes and top-four gates. Capture its pre-MLP hidden states and chosen
E1280 child IDs. For each selected child, calculate all ten METH-135
scores, subtract their mean and divide by their population standard
deviation clamped at `1e-6`. Independently calculate ten deterministic
Gumbel values from splitmix64 of current token, previous token, window
position, E1280 child ID, layer ID and candidate local ID. Convert the
upper 53 bits to `(bits+0.5)/2^53`, then `-log(-log(u))`. Route to the
argmax of normalized score plus `beta * Gumbel`. Ties choose the
lowest local ID. The hash and score are stateless; no fitted scalar
threshold or source label is used. All ten slots are content slots;
there is no special structural slot in this screen. Every grandchild
ID must divide by ten to its unchanged source child.

**Selection and validation.** Evaluate fixed beta candidates
`0.5, 1, 2, 4, 8` on the METH-136 deterministic training raw/chat
draws, using METH-150's window construction. Choose the *smallest*
beta passing every training cell/layer: (i) candidate/control
maximum-to-mean load ratio at most 1.25, (ii) no grandchild receives
more than 25% of a parent selected at least 250 times, (iii) at least
4,000 grandchild IDs selected per layer, and (iv) mean selected
normalized content score exceeds the same-input pure-hash winner's
score by at least 0.05 and raw-argmax agreement is at least 15%.
If none passes, stop before validation. Freeze that beta and evaluate
the 24 METH-150 source-separated documents at both 128- and 512-token
windows, requiring the same four conditions in every cell/layer.
The METH-150 documents were already inspected for another route and
are therefore a source/context transfer diagnostic, not an untouched
promotion cohort or quality evidence. Record all 24 layers and every
beta training summary, plus source-binding hashes and resource use.

**Decision.** A pass licenses a native C route-cost test and then
matched B-only training with a new untouched quality set. A failure
rejects this fixed score-plus-hash rule under these gates; it does not
rule out content-aware routing generally. Content-score advantage
tests dependence on state, not whether it selects a useful specialist.
Neither outcome proves CPU LUT cost, useful E12800 capacity, the
10B/100B rung or end-to-end throughput.

**Budget.** Local RTX 3060 only, no T4; six CPU threads, 15 minutes,
10.5 GiB peak allocated GPU memory, 20 GiB process RSS and 1 GB new
disk. Stop on binding/parity failure, absence of a training-passing
beta, or any exceeded resource limit; retain failed stage and partial
rows. The result record will give exact command and code revision.
