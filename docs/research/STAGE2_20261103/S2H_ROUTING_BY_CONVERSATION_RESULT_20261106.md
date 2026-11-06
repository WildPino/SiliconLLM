# Stage 2: does a document use a stable subset of experts? RESULT

2026-11-06. Protocol `S2H_ROUTING_BY_CONVERSATION_PROTOCOL_20261104.md`. Records: `benchmarks/stage2/results/s2r_verdict.json`.

## Outcome: NO SUBSET at k = 4 (A LEANING SUBSET at k = 8, reported)

The hot set beats shuffle by far, and beats the frequency control by 0.11, whose CI95 lower end falls short of the 0.10 line at k = 4. The plant fires, so the measure can see a change of topic. Neither k comes near the 0.97 the idea needs.

| 30% of the experts, slots | k = 4 | k = 8 |
| --- | --- | --- |
| Hot set, later tokens | **0.811 [0.789, 0.830]** | 0.792 [0.771, 0.811] |
| freq | 0.700 | 0.662 |
| Hot minus freq | [+0.087, +0.135] | [+0.107, +0.154] |
| shuffle, another category | 0.332 | 0.321 |
| Hot minus shuffle | [+0.448, +0.508] | [+0.437, +0.504] |
| shuffle, same category | 0.667 | 0.667 |
| LRU, reported | 0.911 | 0.876 |
| Tokens with every slot in the hot set | 0.52 | 0.30 |
| Plant's fall, single documents' | 0.479, 0.037 | 0.462, 0.034 |

| k = 4, other sizes | 10% | 50% |
| --- | --- | --- |
| Hot set | 0.569 | 0.899 |
| freq | 0.392 | 0.863 |
| LRU | 0.707 | 0.971 |

Experts covering 90% of the slots, per document and layer: 69 of 256 at k = 4, 85 at k = 8. On chat answers, 81 tokens long: 47 at k = 4. Layers 1-2 hold the weakest hot sets (0.65).

## What it says

- **Routing follows the kind of text, and the first 256 tokens read it.** Another category covers 0.33, about what chance gives (0.30); the same category 0.67. This is a real leaning.
- **It is far from a static set.** On the desk, a miss rate m costs 124·m ms: the hot set at 30% costs 23 ms, and the LRU 11 ms. Only the LRU at 50% gets near, at 3.6 ms, with about 13 GB of experts in RAM in place of 26. The room is 1.12 ms. No SSD was measured.
- **The predictions were wrong on structure:** they assumed a shared committee and mild topic effects; the categories are nearly disjoint at this size.

## Predictions

| | Predicted | Read | |
| --- | --- | --- | --- |
| Hot set at 30%, k = 4 | 0.60-0.75 | 0.811 | miss, higher |
| at 10%, 50% | 0.25-0.38, 0.82-0.92 | 0.569, 0.899 | miss, hit |
| freq, shuffle at 30% | 0.40-0.52, 0.50-0.62 | 0.700, 0.332 | miss, miss |
| Experts for 90% of the slots | 120-170 | 69 | miss |
| Plant's fall, single documents' | 0.03-0.10, 0-0.02 | 0.479, 0.037 | miss, miss |
| Outcome | LEANING 60%, NO 38%, STABLE 2% | NO | 38% |

## Thesis line

With every expert in RAM this buys no speed. As a way to hold fewer experts in RAM it does not reach the room under 24.7 on the desk. This measures Ling-mini on documents, not Ling-flash on conversations.

## Parts specific to Ling

The numbers: 256 experts, 8 per token, sigmoid routing with a balancing bias, groups of 32 with 4 kept, a shared expert beside them. The recorder and the measures need only a top-k MoE.

## Recommendation

**The idea closes on routed experts, as registered.** Neurons inside the organs every token reads stay unmeasured. The LRU row at 50% goes in the delivery as a fact for a machine with less RAM, not as a plan.

**What would disprove it:** a measured SSD whose miss costs under 0.3 ms, which would bring the LRU at 50% inside the room.
