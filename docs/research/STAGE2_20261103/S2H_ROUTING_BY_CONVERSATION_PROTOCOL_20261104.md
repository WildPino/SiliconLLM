# Stage 2: does a document use a stable subset of experts? A measure of routing. PROTOCOL

2026-11-04. Ling-mini with P0, CPU only, after test 3b. No change to the model.

## Question

Inside one document, does the model keep to a stable subset of its experts? If the first tokens can name that subset, the
other experts need not sit in RAM until a trigger calls them.

## Thesis line (rule 4)

With every expert in RAM this buys no speed: a token still reads k experts per layer. What it can buy is memory. On Ling-flash
at 2 bits the experts weigh about 26 GB, and 30% of them about 8 GB. A miss reads one expert, 3.1 MB, from an SSD at about
3 GB/s: about 1 ms on the desk. At k = 4 a token has 124 slots, so a miss rate m costs 124·m ms. The room under 24.7 is
1.12 ms: coverage must reach 99%. At 97% the cost is 3.7 ms: 39.37 + 3.7 = 43.07 ms, about 23.2 tok/s. Nothing here is a measured SSD.

## Literature (read in the papers, except where marked)

| Source | What it gives here | Known ways to fail |
| --- | --- | --- |
| MoE-Infinity, arXiv 2401.14361, §4.1-4.4, Fig. 2 | traces experts per request: fewer than 5% repeatedly active in models of about 100 experts, about 25% on Mixtral; the skew shows per request and evens out over 1,000 sequences | transitions between activation groups are hard to predict (highest about 0.3); no per-layer table |
| "The Illusion of Specialization", arXiv 2601.03425, §3.2, Table 5 | a committee of 1-5 experts per layer holds 14-70% of the routing, the same across domains, on DeepSeek-V2-Lite, Qwen3-30B-A3B, OLMoE | routing recorded at the last token; a shared committee is what the frequency control should catch |
| OpenMoE, arXiv 2402.01739 (abstract) | routing follows token ids, with little regard to context | if so, a document's subset is its vocabulary, not its topic |
| GRIFFIN, arXiv 2404.01365 (abstract) | "flocking": feed-forward neurons chosen per sequence, 50% kept with little loss | dense models; not routed experts |
| ELDR, arXiv 2607.00466 (abstract) | a request's prefill names its experts well enough to route requests by it | batch serving on GPUs, a 5.9-13.9% gain |

The trivial control for RAM is the operating system's page cache under `mmap`: the LRU row below stands for it.

## Design (`s2r_route_hook.h`, `s2r_run.py`, `s2r_verdict.py`)

The graph callback writes each token's chosen ids and weights, per layer. Dry run: with the recorder on, the records on the
panel's first two documents equal ref's byte for byte.

| Set | What | Runs |
| --- | --- | --- |
| docs | the 72 documents | k = 8; k = 4 with k₂ = 16 on every position |
| gen | prompts and answers of test 2b's set; the 40 chat answers are read from it | both |
| pairs | 24 pairs, the first 1,024 tokens of a document and of a document of another category | both |

Measures per document and MoE layer, averaged over layers, then documents; CI95 by 10,000 draws of documents:
- **Spread:** experts covering 90, 95 and 99% of the slots, and of the router's mass.
- **Hot set:** the experts most used in the first 256 tokens, at 10, 30 and 50% of the 256. The share of later slots and mass
  it covers, and the share of tokens with every slot inside it.
- **Trivial control (freq):** the same size, by frequency over the other documents.
- **Shuffle:** the hot set of a document of another category. Of the same category: reported.
- **Plant:** in a pair, the set from the first half's first 256 tokens must lose coverage after the switch. Fall = coverage of
  tokens 256-1,023 minus coverage of tokens 1,024-2,047, against the same fall on single documents.
- **LRU, reported:** a cache of the same size, warmed on the first 256 tokens.

Chat answers run about 50 tokens after a prompt of about 45: a hot set from 256 tokens cannot be read there, so chat reports
the spread only.

## Line (registered; `s2r_verdict.py`, selftest PASS, 6 cases)

At k = 4, 30% of the experts, slots:
- **A STABLE SUBSET:** coverage ≥ 0.97, above freq and shuffle by the CI95.
- **A LEANING SUBSET:** at least 0.10 above freq and 0.05 above shuffle, by the CI95's lower ends.
- **NO SUBSET:** otherwise.

**VOID:** the hot set beats shuffle by 0.03 or more, yet the plant's fall does not exceed the single documents' (CI95 above 0):
the measure would claim a topic it cannot see. A silent plant with no gap is not VOID: that is NO SUBSET. k = 8 is reported.

## Predictions

| | Predicted |
| --- | --- |
| Hot set at 30%, k = 4 | 0.60-0.75 |
| at 10%, 50% | 0.25-0.38, 0.82-0.92 |
| freq, shuffle at 30% | 0.40-0.52, 0.50-0.62 |
| Experts for 90% of the slots | 120-170 of 256 |
| Plant's fall, single documents' | 0.03-0.10, 0-0.02 |
| Outcome | LEANING 60%, NO 38%, STABLE 2% |

## What a result changes (rule 5)

The prediction is against the idea, and the decision is whether it goes on at all:
- **STABLE:** the delivery proposes a stage for it: experts on disk, the hot set in RAM, the first real timing.
- **LEANING:** the subset exists but misses too often for a static set. The delivery says what the LRU row adds, and the idea
  waits for a reason to pay about 4 ms or more.
- **NO SUBSET:** the idea closes on routed experts. Neurons inside the organs every token reads stay unmeasured.

## Parts specific to Ling

- **Specific:** the numbers: 256 experts, 8 per token, sigmoid routing with a selection bias that balances load, groups of 32
  with 4 kept, a shared expert that takes what every token needs.
- **Not specific:** the recorder and the measures need only a top-k MoE.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| A conversation | documents of 1,800-4,600 tokens | the panel's chat is too short for a set; documents are the longest coherent text on disk |
| The first prompt | the first 256 tokens | as asked |
| The trigger that reloads | not simulated | it needs the miss cost, which is a desk number |
| Ling-flash | Ling-mini | same experts per layer, twice the width, 32 layers: not measured |
| Decode-only k = 4 | k = 4 on every position | documents have no generated tokens; the routing is the same form |
| Neurons inside an organ | not measured | as asked: experts only |
| Effect on quality | none: the model is not changed | a measure of coverage, not of loss |

## Cost

Six recordings, about 3.5 h of CPU beside the trainer. No GPU.
