# C50-S3 sizing: the teacher-signal options in tokens/s and GB, before the owner decides

19 October 2026. ROADMAP §5 S3: "Before the owner decides, S3 is sized in tokens/s and in GB." This note sizes the
four options. It decides nothing; the choice is the owner's (COMMUNICATION item 2).

## What is measured here, and what is not

- **Local throughput** comes from S2's F0 panels (Ling-flash-2.0 Q4_K_M, no imatrix, 19 October). These are wall
  times of a quality run on a box that was in use, so they are **throughputs for sizing, not rates**.
- **Hosted prices and features** come from third-party aggregator pages, read on 19 October. **None of them is
  verified** on the provider's own page. They are marked `[unverified]` and enter no decision until they are read
  there, together with the terms of use.

## Local F0 throughput (measured wall, S2)

| Work | Panel | Tokens | Wall s | Throughput |
| --- | --- | --- | --- | --- |
| Teacher-forced scoring (logits for every position) | docs | 165,858 positions | 8,061 | 20.6 tok/s |
| | mmlu | 129,316 | 6,350 | 20.4 tok/s |
| | arc | 88,749 | 4,749 | 18.7 tok/s |
| Generation | gsm8k | 23,701 generated (+9,993 prompt) | 4,510 | ~5.3 generated tok/s |
| | chat | 4,087 generated (+1,219 prompt) | 878 | ~4.7 generated tok/s |

So the local teacher scores about **20 tok/s, ~72,000 tokens per hour, ~1.7 M per day**. It generates about
**5 tok/s, ~18,000 per hour**. While it runs it holds ~71 GB of RAM: no rate cell, S1 session or other long job can
run beside it (rule: one long job at a time).

## The options

| Option | Signal | Volume per day of machine | Storage | Money | Main constraint |
| --- | --- | --- | --- | --- | --- |
| **A. Local CPU, top-k logits** from F0 | per-token top-k distribution on our own text, exact tokenizer | ~1.7 M tokens | top-32 at 6 B per entry (uint32 id + f16): 192 B/token, **0.19 GB per M tokens** (top-64: 0.38 GB) | none | occupies the benchmark machine; the teacher is the Q4_K_M without imatrix, not BF16 |
| **B. Hosted endpoint, sequence-level** | the teacher's generated text (plus top logprobs on its own tokens where offered) | set by the provider's rate limits `[unverified]` | text: ~4-5 B per token, ~5 GB per G tokens | Ling-flash-2.0 on SiliconFlow: $0.14 in / $0.57 out per M tokens `[unverified]`; Ling-3.1-flash on OpenRouter: free tier and $0.30 / $0.90 both reported `[unverified]` | terms of use for training on outputs `[unverified]`; a chat API returns logprobs only for tokens it generates, so it cannot score our text teacher-forced |
| **C. Id-compatible Ling-3.x teacher** | as A or B | locally: does not fit (below) | as A or B | as B | Ling-3.0-flash is 127.5B stored (A0): at Ling-mini's 4.87 bits per weight that is ~78 GB, above the 71.7 GB free at idle. Only hosted (= B) or at ~3 bits |
| **D. Offline memory-bank values** (teacher once per bank row) | values for S4 arm (b), built once and frozen | rows x tokens per row / 72,000 per hour | rows x row bytes (set by S4's design) | none if local | the bank size is S4's design choice; e.g. 100,000 rows of 64 scored tokens = 6.4 M tokens = **~89 h** locally |

- **Id compatibility.** Ling-flash-2.0's tokenizer is the same parsed object as Ling-mini's. Ling-3.x differs only in
  nine special tokens (ids 156900-156908), so ordinary text tokenizes the same (A0).
- **What the volume buys (from S2).** The donor-gap panel has 409 items and 71 of 72 documents, and it is held out:
  none of it may enter training or bank construction. The teacher's volume goes to the matched training data of S4,
  whose size S4's protocol fixes.

## Reading the table

- **A is the only option that gives per-token distributions on our own text, and it is free, but it is slow:** at
  ~1.7 M tokens per day, 10 M teacher tokens take ~6 days of the machine, during which nothing else runs.
- **B is cheap in money at the reported prices** (10 M generated tokens would cost ~$6 at $0.57 per M) but gives
  sequence-level signal only, and every number in its row needs verifying first, terms of use included.
- **C adds nothing locally**: the newer teacher does not fit at 4 bits.
- **D is the cheapest per unit of use** if S4 chooses arm (b): the teacher runs once per row, not once per training
  token. It scales with the bank, so its cost is fixed when S4's protocol fixes the bank.

## Recommendation (the owner decides)

**D for arm (b), sized in S4's protocol to fit about 2-4 nights of the local machine, plus A on a small matched set
for arms (a) and (c).** Both are local, exact-tokenizer and free, and need no terms of use. B stays as the fallback
for volume if S4 needs more than ~10 M teacher tokens, after its prices, rate limits and terms are read on the
provider's own pages.

Sources (aggregator pages, read 19 October 2026, unverified):
[Artificial Analysis, Ling-flash-2.0 providers](https://artificialanalysis.ai/models/ling-flash-2-0/providers),
[modelcompare.dev, SiliconFlow Ling-flash-2.0](https://modelcompare.dev/models/siliconflow/ling-flash-2-0),
[OpenRouter, Ling-3.1-flash](https://openrouter.ai/inclusionai/ling-3.1-flash),
[llmreference, Ling-3.1-flash on OpenRouter](https://www.llmreference.com/model/ling-3.1-flash/openrouter),
[CloudPrice, Ling-3.1-flash](https://cloudprice.net/models/inclusionAI/Ling-3.1-flash).
