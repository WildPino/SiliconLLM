# E6 — the donor speaks: a pretrained model generating its own tokens on our runtime

**Status: CLOSED, `GENERATION-CONFIRMED`.** Brief: `briefs/BRIEF_E6_GENERATE.md`, pushed at
`ecb37e3` before the engine could generate anything. Every gate passed.

| | |
|---|---|
| what ran | `donor_engine.c --generate`, a fourth mode: greedy argmax, no temperature, no seed |
| donors | `qwen25-05b_f32.bin` (fp32), `qwen25-05b_tqh.bin`, `qwen25-15b_tqh.bin` — real converted Qwen2.5 weights, not synthetic shapes |
| reference | HF `Qwen/Qwen2.5-0.5B` and `-1.5B`, fp32, CPU, greedy |
| prompts | five, frozen in brief §5 before the first run; `n_new = 32`; `--threads 6` |
| data | `engine/results/e6/{engine.json, ref.json, summary.json, score.txt}` |
| runner | `engine/e6_generate.py` (`--stage engine | ref | score`) |

---

## 1. The result

**Our runtime, given a real pretrained donor in fp32, reproduces PyTorch's greedy continuation
token for token: 160 of 160 positions, across five prompts, zero divergences.**

    prompt : "The capital of France is"
    ours   : " Paris. It is the largest city in Europe and the second largest in the world. It is
              also the capital of France, the second largest country in Europe,"
    pytorch: " Paris. It is the largest city in Europe and the second largest in the world. It is
              also the capital of France, the second largest country in Europe,"

    prompt : "import numpy as np"
    ours   : "\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\nfrom
              sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import
              Linear"
    pytorch: identical

The other three prompts are identical too; the full transcript is `results/e6/score.txt`.

**Why this was not already known.** E1 proved the engine computes the same *logits* as PyTorch
(`1.5e-05`) and the same BPB. Every mode this engine had was **teacher-forced** — the next token
always came from the corpus. `--bench` does not even use the model's output: it feeds
`1+(i%100)`, a counter, precisely so that timing cannot depend on what the model says. So no donor
had ever chosen a token here, and 32 greedy steps is a different claim from one logit vector:
errors that a single position hides compound once the model's own output becomes its input.
**They do not compound. The trajectory is identical.**

## 2. The planted control, and what +2.466 BPB looks like as prose

The same instrument, pointed at the ternary donors — the format this programme actually consumes:

| arm | weights | agreement with PyTorch greedy | first divergence |
|---|---|---|---|
| **A1** | `qwen25-05b_f32.bin` | **160/160 = 100.0%** | none |
| A2 | `qwen25-05b_tqh.bin` | 3/160 = **1.9%** | position 0 of prompt 0 |
| A3 | `qwen25-15b_tqh.bin` | 10/160 = **6.2%** | position 0 of prompt 0 |

A2, at the same prompt A1 answered with *Paris*:

    " the, for a on, A for the, V BaseEntity sond,3---</,3---</, in-:)\n0,3 (,3',...\n the,"

A3, 1.5 B, is degenerate in the other direction — it repeats ` the` and newlines.

**This is the planted control doing its job, and it is also the most legible statement of the
quality gap this programme has produced.** T2b measured the whole runnable model at **+2.466 BPB**
over the donor. That number has been on the INDEX for weeks. Here is what it *is*: the converted
model does not write text. Nothing in E6 is new information about the conversion — it is the same
finding in a form that cannot be misread as "large but survivable".

Had A2 also reproduced PyTorch, G-A would have been measuring how predictable five short prompts
are rather than whether the runtime is faithful, and **E6 would have been void**. It was not.

## 3. Speed, and a correction that had to be made first

**`1000/f` is a ceiling — the rate the engine would reach if the weight path took zero time. It is
not a measured rate and never was.** Two figures in this programme's record are ceilings and have
been read as throughput:

| figure | what it is |
|---|---|
| E4, `1000/f` = **78.5 tok/s** at `T10` @800 | ceiling with a free weight path |
| E5 run 6, `1000/f` = **68.4 tok/s**, same cell | the same ceiling on a sweep that ran 15.5% hot |
| **measured, `T10` @800 (10.6 B)** | **3.09–3.14 tok/s** (`results/e5/run6.log`) |
| measured, `S05` @800 (0.5 B) | 52.4–52.9 tok/s (same log) |

**At the target shape the engine decodes at about 3.1 tok/s, roughly 16× short of the 50 tok/s
goal.** E6 does not move that.

What E6 adds is the first **generation** rate — a decode loop whose next token comes from the
model, on real weights:

| arm | decode | prefill | context |
|---|---|---|---|
| A1 `05b_f32` | 18.80–18.89 tok/s | 17.7–18.0 | 0.5 B, fp32, short prompts |
| A2 `05b_tqh` | 59.80–61.58 tok/s | 52.3–55.0 | 0.5 B, ternary |
| A3 `15b_tqh` | 20.49–20.94 tok/s | 11.7–20.1 | 1.5 B, ternary |

**These are contended timings and are not offered as speed results.** The machine carries chronic
background load; the standing rule is that a timing needs an idle machine, ≥3 repetitions and a
reported dispersion. They are witnesses that generation is not pathologically slower than the
teacher-forced path, nothing more. In particular **A2's 61 tok/s is not "50 tok/s reached"**: it is
a 0.5 B model at a context of 3–8 tokens producing text that section 2 shows is unusable.

## 4. What E6 does not claim

- **Not that a 10 B donor generates.** There is no real 10 B donor on this disk. `T10` is a
  synthetic shape file — correct dimensions, no trained weights — which is why E3/E4/E5 could
  measure speed at that shape and nothing else. The demonstration closes at 1.5 B.
- **Not that the ternary model works.** Section 2 says it does not.
- **Not a new speed number.** Section 3 stands.
- **Not that sampling works.** Greedy only, on purpose: with a sampler, two models that differ and
  two models that agree both produce different text, and the gate could not fail.

## 5. The gates

| gate | test | result |
|---|---|---|
| **G-P** | the prefill logits from `--generate` vs those from `--logits`, same ids and weights | **PASS, byte-identical**, all 15 arm×prompt cells — so generation is the forward pass E1 gated, not a second implementation |
| **G-D** | the same arm run twice | **PASS, byte-identical ids**, all 15 |
| **G-A** | A1 vs PyTorch greedy, ≥90% of 5×32 positions, any divergence a REF top-2 tie-break | **PASS at 100.0%**, no divergence to excuse |
| **G-C** | planted control: the ternary arms must **not** also pass | **PASS** — 1.9% and 6.2% |
| **G-T** | throughput | reported, not gated (section 3) |

G-P and G-D were computed and printed **before any generated text was decoded**, per brief §6, so
the transcript could not be read into meaning something it does not.

## 6. The sentence this fixes

The INDEX has said, since this programme opened:

> "the road exists end to end — safetensors → export → ternary runtime → **generated tokens**, with
> a parity gate at the seam."

The last clause was not backed by anything: no mode of this engine emitted a token. It is now
backed **for the fp32 runtime**, at the strongest form available — identity with PyTorch's own
greedy trajectory. It is **still not backed for the ternary runtime**, which generates tokens but
not language, and the INDEX now says so.

## 7. Owed

1. **A 10 B donor that is real.** Everything at target scale in E3–E5 is a synthetic shape. Until
   a real 10 B is exported, "10 B at 50 tok/s" has a measured speed and no measured model.
2. **The quality gap is now the only thing between here and a working deliverable at 0.5–1.5 B**,
   and section 2 is the cheapest possible statement of it.
3. **Sampling and a stop condition**, if generation is ever to be shown to a human rather than to
   a gate.
4. Longer generations: 32 tokens exercises no context beyond the prompt. E4's owed item 3 —
   `f` beyond 800 tokens — is the same hole seen from the speed side.
