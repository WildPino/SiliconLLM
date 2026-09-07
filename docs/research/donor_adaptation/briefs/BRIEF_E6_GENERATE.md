# E6 — the donor speaks: a real pretrained model generating text on our runtime

**Pre-registered. Pushed before any generation was run.** Sections are added the same way E5's
were: each one before the run it governs, never after.

---

## 1. Why this exists

The INDEX's one-line state has said, since the beginning of this programme:

> "the road exists end to end — safetensors → export → ternary runtime → **generated tokens**,
> with a parity gate at the seam."

**That sentence is not backed.** `donor_engine.c` has exactly three modes:

    --logits <ids.bin> <n> <out>   dump fp32 logits          (parity)
    --bpb   <ids.bin>              nats over next-token      (quality)
    --bench <n>                    time n decode steps       (speed)

None of them produces a token. `--bench` feeds the engine `1+(i%100)` — a counter, not a
prediction — precisely so that the timing does not depend on what the model says. Every quality
number in this programme is a **teacher-forced** number: the next token always came from the
corpus, never from the model. **No donor has ever emitted a token of its own on this engine.**

E1–E5 measured that the runtime computes the right numbers (parity `1.5e-05` against PyTorch) and
how fast it computes them. Neither is the same claim as *the model works*.

## 2. A correction that has to be on the record first

`1000/f` is a **ceiling**: the tok/s the engine would reach if the weight path took zero time.
It is not, and has never been, a measured rate.

| quantity | value | what it is |
|---|---|---|
| E4 `1000/f`, `T10` @800 | 78.5 tok/s | ceiling with a free weight path |
| E5 run 6 `1000/f`, same cell | 68.4 tok/s | the same ceiling, read on a sweep that ran 15.5% hot |
| **measured throughput, `T10` @800** | **3.09–3.14 tok/s** | `results/e5/run6.log`, four reads |
| measured throughput, `S05` @800 | 52.4–52.9 tok/s | same log |

**The 10.6 B shape runs at ~3.1 tok/s — about 16× short of the 50 tok/s goal.** Any reading of
78.5 or 68.4 as "we exceed 50 tok/s" is a category error, and the ledger says so in §14.2-bis.
E6 does not change this. E6 changes whether there is a *model* behind the number at all.

## 3. The question

> Does a real pretrained donor, converted by our exporter and executed by `donor_engine.c`,
> generate text — and is that text the **same** text PyTorch generates from the same prompt?

## 4. The apparatus

A fourth engine mode, `--generate <prompt_ids.bin> <n_new> <out_prefix>`:

- prefill the prompt at positions `0..P-1`;
- then `n_new` steps of **greedy argmax** — deterministic on purpose, so the run is a gate and
  not a demo (no temperature, no top-p, no seed);
- write `<out_prefix>.ids.bin` (int32, prompt followed by the generated ids) and
  `<out_prefix>.prefill.bin` (the fp32 logits of every prompt position).

Greedy first because a sampler makes disagreement unfalsifiable: with sampling, two models that
differ and two models that agree both produce different text.

## 5. The arms, fixed before the run

| arm | weights | what it tests |
|---|---|---|
| **A1** | `D:/_ktmp/e1/qwen25-05b_f32.bin` | our runtime, fp32 — the **known-positive** |
| **A2** | `D:/_ktmp/e1/qwen25-05b_tqh.bin` | ternary weights + ternary head — the format we consume |
| **A3** | `D:/_ktmp/e1/qwen25-15b_tqh.bin` | the same at 1.5 B |
| **REF** | HF `Qwen/Qwen2.5-0.5B` / `-1.5B`, fp32, `do_sample=False` | PyTorch, greedy |

**Prompts, frozen here so none can be shopped for after seeing output.** Five, plain, no chat
template (these are base models, not instruct):

1. `The capital of France is`
2. `def fibonacci(n):`
3. `Water boils at`
4. `The three laws of motion were formulated by`
5. `import numpy as np`

`n_new = 32` for every arm and prompt. `--threads 6`.

## 6. The gates

| gate | test | threshold, fixed now |
|---|---|---|
| **G-P** | the prefill logits from `--generate` vs the logits from `--logits`, same ids, same weights | **byte-identical**. Anything else means generation is a second forward pass, and E1's parity does not cover it |
| **G-D** | the same arm run twice | **byte-identical** output ids |
| **G-A** | A1 vs REF, greedy token agreement over 5 prompts x 32 tokens | **≥ 90% of positions agree**, and any divergence must be at a position whose REF top-2 logit gap is **< 1e-2** — a tie-break, not a different model |
| **G-C** | **planted control**: A2 vs REF, same measurement | must **NOT** pass G-A. The ternary conversion costs **+2.466 BPB** (T2b, as folded by E2); if a model that damaged still reproduced PyTorch token for token, G-A would be measuring the predictability of the prompts and nothing else, and **E6 is VOID** |
| **G-T** | throughput | **reported, not gated.** The machine carries chronic background load; a contended timing is not a timing. Any tok/s here is a witness with its dispersion, and is never compared to 50 without that caveat |

G-A carries the load and G-C is what makes it able to fail. Order matters: **G-P and G-D are run
before any text is read**, so the transcript cannot be talked into meaning something it does not.

## 7. What E6 will not claim

- **Not that the ternary model is good.** A2 and A3 exist to make G-C fire and to put the
  known +2.466 BPB in front of a reader as prose instead of a number. Whatever they emit is an
  illustration, not a quality result — the quality result is the BPB, already measured.
- **Not a speed result.** §2 stands; E6 adds a decode rate at 0.5 B and 1.5 B, which are not the
  target shape, and does not touch the 3.1 tok/s at 10.6 B.
- **Not that a 10 B donor generates.** No real 10 B donor exists on this disk; `T10` is a
  synthetic shape file. E6 closes the demonstration gap at the sizes that have real weights and
  says plainly that it does not close it at the target size.
