# METH-71: donor retention plus product-key route balancing at 10× E

**Motive.** METH-70 uses the same dense recipe at E128/E1280 and
fails donor top-1 in both; E1280 falls to 92.865%, selects only
474/1,280 slots in the least-covered development layer and reaches
129.19× worst maximum/mean load. Test one fixed joint rule that
strengthens donor retention and penalizes product-key axis
concentration during training. A raw BPB gain or changed weights
alone cannot qualify the model.

**Arms and source.** Start new E128=8×16 and E1280=32×40 models from
the bound Qwen2.5-0.5B-Instruct BF16 donor, with independent rank-8
FP32 residual factors, rank-64 FP32 product keys and zero B donor
parity. No copied expert rows. Use the same fixed training draws in
both arms, two raw and two full-chat microbatches per update, donor
CE+full-distribution KL with weights 2.0 raw and 4.0 chat, dense GPU
AdamW with factor LR 1e-4/router LR 1e-5, no decay, global-norm clip
1.0, and 64 updates. These retention settings come from METH-56;
the balance term below is the only new routing objective.

**Balance objective.** For each layer and each A/B product-key axis,
calculate hard top-four axis selection frequency `f` over the current
student batch, divided by four times token count, and detach it.
Calculate `p = mean_tokens softmax(axis_scores / τ)`, with τ the
detached batch standard deviation of axis scores clamped to 1e-3.
For axis width `n`, use `n * sum_i(f_i * p_i) - 1`. Add 0.02 times the
sum of both axis terms over all 24 layers to each microbatch's
CE+KL objective. Do not modify inference routing, top-four or expert
arithmetic. Record balance loss, hard route coverage and load.

**Fresh evidence.** Before training, freeze seed-717171 24 chat
development prompts from the bound corpus, excluding chat-training
rows, METH-44/47/55/56/66/70 development rows, and the raw rows
actually sampled in METH-70. Exclude all these plus new dev rows from
METH-71 raw training. Use training RNG seed 717172 in both arms.
The prompt set can only be used for this terminal development gate;
passing requires a separate source-disjoint generation, task and
blind semantic audit.

**Terminal gates.** At update 64 require donor/student initial logit
identity, finite nonzero B/router gradients (the balance term gives the
router a gradient even at the first zero-B update), donor-position top-1
≥95% on new prompts,
held-out raw ΔBPB≤+0.05, ≥64 changed B slots/layer for E128 or
≥640 for E1280, and exact product-key top-four versus exhaustive
pair scores on evaluated hidden states. For route utility, require
minimum selected slots/layer ≥96 at E128 or ≥640 at E1280 and worst
maximum/mean load ≤32× at E128 or ≤64× at E1280. Failure of any gate
stops promotion. Report all layer-level counts and checkpoints.

Local RTX 3060 only, ≤10.5 GiB peak allocated GPU, ≤20 GiB RSS and
≤20 minutes per arm including save/hash. Stop and preserve partial
evidence on a budget or nonfinite failure. No T4. A joint pass is a
development candidate, not proof of semantic quality, a LUT bank,
full `engine.c` parity or ≥50 accepted tok/s on the same artifact.
