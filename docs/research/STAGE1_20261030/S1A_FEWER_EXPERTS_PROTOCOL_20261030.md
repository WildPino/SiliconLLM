# Stage 1 (a): fewer routed experts per token on Ling-mini, without training, and the quality panel

**Status:** PROTOCOL, 30 October 2026. CPU only, ~31 h. No GPU.
**Code:** `benchmarks/stage1/`:
- `s1_moe_hook.h`, `s1_llama_score.cpp`, `s1_llama_generate.cpp`, `s1_build.sh`;
- `s1_dry.py`, `s1_calibrate.py`, `s1_run.py`, `s1_verdict.py`.

## Question

How many points does Ling-mini-2.0 lose when each token uses fewer than its 8 routed experts? The test runs at k = 6,
4 and 2, with no training. Does keeping the experts with the highest router score beat keeping the same number at
random?

This protocol also registers the **quality panel** that every stage 1 test reports against. It is fixed here, before
the first conversion.

## The quality panel (registered for all of stage 1)

- **Reference:** Ling-mini-2.0 Q4_K_M with imatrix (sha256 `44de7315…`), at k = 8 with the original routing.
- **Items:** C50-S2's frozen inputs, as recorded by their manifests:

  | Panel | nq | tqa | gsm8k | arc | mmlu | chat | Documents |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Items | 600 | 600 | 100 | 600 | 570 | 40 | 72 |
- **Points:** the unweighted mean of the six item panels' accuracies in percent, scored by `c50_s2_checks.py`
  unchanged.
  - The budget: at most 5 points lost against the reference, paired on the same items.
  - The chat must stay usable (chat gate below).
- **Against the original**, on the 72 documents (165,858 positions):
  - top-1 agreement;
  - KL(reference ‖ test) on the reference's top-32 ids plus one bin for the rest. Merging bins can only lower a KL, so
    this is a lower bound of the full KL.
  - the paired dBPB.

  These are measured on fixed reference text, so they do not show drift during free generation. The rollout flags
  below cover that.
- **Noise floor:** a *replay* arm runs the reference again at ubatch 256, a numerically different but equivalent run.
  Its Δ and its document metrics are the floor every other arm is read against.
- **Trivial control:** every test names its own.
- **Runtime:**
  - All arms run through one binary per tool: the pinned llama.cpp `5b335f4` AVX2 static libraries, linked by g++
    16.2 after the C: drive was reinstalled on 30 October.
  - Settings: 6 threads, ubatch 512, F16 KV, flash attention off, greedy decoding, the C50 stop ids.
  - The dry run checks the new binaries against the old tools and against C50-A4's recorded outputs.

## Arms

All arms, the reference included, run through one evaluation callback (`s1_moe_hook.h`):
- The callback rewrites `ffn_moe_weights-<layer>`, the weights of the 8 chosen slots, before llama.cpp normalizes them.
- llama.cpp orders the slots by **biased** router score; the bias balances expert load. So keeping slots 0..k−1 is
  exactly llama.cpp's own top-k.
- A dropped slot gets weight 0, and the kept slots are renormalized as `norm_topk_prob` does.

| Arm | What changes | Experts per token | Control |
| --- | --- | --- | --- |
| ref | nothing | 8 | — |
| replay | nothing, ubatch 256 | 8 | — (noise floor) |
| top6, top4, top2 | keep slots 0..k−1, at every position | 6, 4, 2 | rnd6, rnd4, rnd2 |
| topw4 | keep the 4 slots with the highest unbiased weight | 4 | rnd4 |
| rnd6, rnd4, rnd2 | keep k of the 8 slots uniformly at random (seed 20261030) | 6, 4, 2 | — |
| top4d | as top4, on generated tokens only; the prompt keeps 8 | 4 on decode | — |
| topp | Top-P: sort the 8 slots by unbiased weight and keep the shortest prefix reaching p | 4.0 on average | rndp |
| rndp | keep as many slots as topp, at random | as topp | — |

- **topw4:** on the dry-run inputs, the biased slot order differs from the weight order on 75-77% of tokens, so
  llama.cpp's top-4 and the top-4 by weight keep different experts. Without topw4, a top4 failure could not separate
  "4 experts are too few" from "the biased score ranks them badly". It is compared with top4 at exactly the same
  compute.
- **top4d:** the speed thesis is decode speed. Its likelihood panels (arc, mmlu, documents) are prompt positions only,
  so it takes the reference's outputs there.
- **Calibration of p** (`s1_calibrate.py`, before any arm):
  - data: the first 200 items of C50-A4's PIQA inputs, which this panel does not use;
  - rule: the grid p from 0.30 to 0.95 whose mean number of kept slots is closest to 4.0.

  **Result: p = 0.65, mean 3.989 kept** (0.64: 3.909; 0.66: 4.072). Topp is compared with top4 only if its
  realized mean over the panel is ≤ 4.05.

## Literature (read in the papers' tables)

| Source | Setting | Reported |
| --- | --- | --- |
| ACE (2609.05228), Table 1 | Qwen3-30B-A3B: 128 experts, top-8, no training. Mean of 7 tasks, BF16 = 80.67 | **30% of slots skipped:** −0.9 (ACE), −2.7 (router-score threshold). **50%:** −6.4 (ACE, Top-P), −14.0 (router score). **60%:** −17 or worse |
| SeqTopK (2511.06494), Table 1 | OLMoE: 64 experts, top-8, fine-tuned 200 steps on one task each | Average 29.74 at k = 8. At k = 4: 24.70 (TopK). At k = 2: 14.49 (TopK), 22.04 (SeqTopK) |
| Lu et al. (2402.14800), Table 5 | Mixtral 8x7B, top-2 of 8, no training | Skipping the 2nd expert below the median weight ratio: 67.58 → 66.37 |
| Lynx (2411.08982) | MoE serving | Reassigning experts hurts prefill and barely hurts decode; tolerance varies by layer (figures only, no numbers used) |

Known ways to fail:
- the loss accelerates beyond ~40% of slots skipped;
- fixed rules lose about twice what dynamic ones lose at the same mean;
- pruning by activation frequency is the worst rule (Lu et al., Table 2);
- the training results found are single-task fine-tuning.

Two limits of this literature:
- every source uses softmax routers, while Ling's is sigmoid, group-limited (8 groups, 4 used) and load-biased, and
  none of them tests cutting k on such a router;
- none randomizes the dropped slots.

## Predictions (registered before any arm runs)

| Arm | Predicted Δ points |
| --- | --- |
| top6 | −0.5 to −3 |
| top4 | −4 to −12 (OVER BUDGET or UNDECIDED) |
| top4d | at least 2 points better than top4 |
| topw4 | within 1 point of top4 either way |
| topp | at least 2 points better than top4 |
| top2 | below −15 |

- rnd4 and rnd2 lose more than top4 and top2.
- Document top-1 agreement falls, and KL rises, monotonically with fewer experts.

## Decision rule (`s1_verdict.py`, separate from the runner; selftest PASS, 17 planted cases)

- **Δ:** points(arm) − points(ref), with a CI95 from 10,000 paired bootstrap draws over the items within each panel.
- **Labels, on the CI95:**

  | Label | Condition |
  | --- | --- |
  | WITHIN SHARE | lower end ≥ −2.5 (the share of the budget declared for this lever) |
  | WITHIN BUDGET | lower end ≥ −5 |
  | OVER BUDGET | upper end < −5 |
  | UNDECIDED | anything else; every decision treats it as not fitting |

- **Chat gate:** CHAT OK iff both hold:
  - the arm passes at least as many chat items as the reference minus 4, out of 40;
  - its degenerate rollouts on chat and gsm8k (C50-A4's flags: a 32-token run, an 8-token loop repeated three times,
    or nothing generated) are at most the reference's plus 3.
- **FITS (THE SHARE):** the label is within budget (within share) **and** CHAT OK.
- **Control:** BEATS CONTROL iff the CI95 of Δ(treatment) − Δ(control) lies above 0. Otherwise:
  - "router no better than chance" only if the control itself measurably loses (its CI95 lies below 0);
  - else CONTROL UNINFORMATIVE.
- **What the rule fixes:**
  - The k of the short training (part 2 of (a), its own protocol) is 4 unless top4, topw4 or top4d FITS THE SHARE;
    then it is 2 unless top2 FITS; then there is none.
  - The form of k = 4 carried into stage 2 is the first that fits the share among: top4 (everywhere, biased score),
    topw4 (everywhere, weight), top4d (decode only).

## What a result changes (rule 5)

- **Neither k = 4 form fits the share (predicted):** the lever lives only with training, and part 2 trains at k = 4,
  with the topp rule if topp beat top4 at matched compute. A top4 that is only WITHIN BUDGET would spend the whole
  budget on one lever, so it also counts as not fitting.
- **Only top4d fits the share:** k = 4 is free at decode; part 2 trains at k = 2 at decode.
- **top4 or topw4 fits the share:** stage 2 assumes k = 4 everywhere, with that arm's rule, and part 2 trains at
  k = 2.
- **topw4 beats top4 beyond its CI:** the load-balancing bias costs quality at k < 8; part 2 ranks by weight.
- **A top arm that does not beat a measurably losing control:** the router is no better than chance at that k. Part 2
  then trains the router first.

## Thesis line (rule 4)

On the stage 0 desk model, at context 2048:
- the routed experts are 780 of the 1,531 MB that an all-ternary Ling-flash reads per token;
- k = 4 removes half of them, which takes the ceiling at the RAM wall from 19.2 to 25.0 tok/s, just above the 24.7
  success threshold;
- with today's packed kernel the gain stays under 1.3×.

So a PASS at k = 4 is one of the conditions of success, not success itself.

It is measured at Ling-mini's scale. Ling-flash shares the routing (256 experts, top-8, 8 groups) but not the expert
width (mini 512, flash 1024). The transfer is checked at stage 2. Until then this is a mechanism test, with a
mechanism-test budget: CPU only, ~28 h.

## What this protocol takes from, or changes in, the design (rule 1)

| Design element (stage 1 (a)) | Here | Why |
| --- | --- | --- |
| Literature first | done | — |
| k = 6, 4, 2 without training, random control | applied | — |
| Short training "on the lowest k that breaks the budget" | **not here, and its k is read differently** | its k is set by this result. The failure criterion of (a) is stated at k = 4 with training, so training targets 4 whenever 4 does not fit the share, not 2 |
| Look at attention before closing the structural route | not here | it applies only if (a) fails with training |
| Added: replay, topw4, top4d, topp and rndp arms | added | noise floor; the ranking rule apart from k; the deployment form of decode speed; a dynamic rule at the same mean |

## Dry run

`s1_dry.py`, with its checks registered in the script (`results/s1a_dry*.json`). It ran on the first 2 documents (6,915
positions), 20 nq prompts and 5 gsm8k prompts.

| Run | Binary | Outcome |
| --- | --- | --- |
| 1 | hook before topw | **FAIL** on one check: the slot count. It expected 19 layers × input tokens, but llama.cpp runs the last layer only on the rows whose logits are requested (in the generator, the last prompt token and the generated ones). The count was off by exactly prompt tokens − items (852 on nq, 491 on gsm8k). The check was corrected, not the hook |
| 2 | same | PASS, 18/18 (`s1a_dry_2_pass_prior_binary.json`) |
| 3 | final, with topw (score `a3b0fa08…`, generate `f2961771…`) | **PASS, 19/19** (`s1a_dry.json`) |

What run 3 shows:
- **Runtime:** the C50 scorer, run after the reinstall, equals C50-A4's recorded P0 outputs bit for bit.
- **Parity, scorer:** the s1 tools without the callback, and with it in mode none, equal the C50 tools bit for bit.
- **Parity, top-k:** top-k through the callback equals llama.cpp's own top-k bit for bit:
  - at k = 6, 4 and 2 in the scorer;
  - at k = 4 on every decode step of the generator.
- **The modes act:**
  - decode-only keeps 4.0 on decode tokens and more overall;
  - topw4 differs from top4.
- **Top-1 agreement with ref on the 2 documents** (instrument check only, not a result):

  | top4 | topw4 | rnd4 |
  | ---: | ---: | ---: |
  | 0.924 | 0.925 | 0.835 |

## Run

The arms run one at a time, in the order ref, replay, top4, rnd4, topw4, top4d, topp, rndp, top6, rnd6, top2, rnd2. Then
`c50_s2_checks.py` scores each arm, and `s1_verdict.py` applies the rule. It is a quality run, not a timing.

## Independent review (rule 6, before the push)

A fresh reviewer read the thesis, the stage design, this protocol and its code. It confirmed four things:
- the hook acts on the right tensor and reproduces llama.cpp's own top-k;
- the group mask does not depend on k;
- the random draws are deterministic;
- the top-32 KL is a valid lower bound and is computed correctly.

Its objections:

| # | Objection | Answer |
| --- | --- | --- |
| 1 (blocking) | Labels on the point estimate sit inside the noise (CI ±2.5–3 points); no noise floor | Labels on the CI with UNDECIDED; replay arm; documents within replay reported |
| 2 (blocking) | The chat condition of the budget was dropped | Chat gate on chat passes and degenerate rollouts |
| 3 (blocking) | training_k passed WITHIN BUDGET, against the declared 2.5-point share; missing row | training_k = 4 unless a k = 4 form fits the share; the rows are written; the reading of "lowest k" is in the rule-1 table |
| 4 (blocking) | topp took a prefix in biased order, so it was not Top-P | topp and rndp sort by unbiased weight; p calibrated afterwards. The hook counts the tokens whose biased order differs from weight order: 75-77%, so the optional top-4-by-weight arm was added (topw4) |
| 5 | topp vs top4 not at matched compute; decode not counted apart | Compared only if the realized mean is ≤ 4.05; the hook counts decode tokens apart |
| 6 | The random control is nearly sure to lose; "does not beat" misread | Read as "chance" only if the control measurably loses, else UNINFORMATIVE |
| 7 | Decode-only, the configuration of the speed thesis, left as a menu | top4d arm, with its decision registered |
| 8 | Hook checks: layers not counted; parity not registered; decode path unchecked | MOE lines assert 8 × 19 × input tokens; `s1_dry.py` registers parity at k = 6, 4, 2 and on the decode path |
| 9 | Literature on softmax routers only; document metrics are teacher-forced | Both stated above |
