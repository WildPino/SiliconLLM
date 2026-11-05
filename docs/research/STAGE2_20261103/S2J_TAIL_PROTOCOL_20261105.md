# Stage 2, part 1, test 2b-bis: the tail of the two-pass head, on new text. PROTOCOL

2026-11-05. Ling-mini, quality only. No training, no timing. The speed line and test 2b's line stay where they are.

## Question

Test 2b's cell 1 failed on one line: dBPB +0.0114 against +0.005, all from words outside the candidates. A tail of a × preview + b,
two constants from dev, read +0.0024 after the verdict. The 72 documents saw that choice among four tails, so they cannot judge
it. Does the tail hold on text nothing has touched?

## Thesis line (rule 4)

The tail adds no milliseconds, so the desk's 25.40 tok/s stays, with its selection and gather still unmeasured. A PASS gives the
head a cell within the line; it says nothing of the experts. It runs on mini; flash would pay for the preview's training.

## The engine's rule

**Sampling happens among the K candidates only, in every mode. The tail never produces a word. It exists only to score.**
So generated text comes from the exact distribution, truncated to K and renormalized. The price of the truncation is the exact
tail's mass, measured here on every position (mean, p99, maximum).

**What changes for a caller who asks for the probability of a word outside the list:**

| Request | Answer | Quality |
| --- | --- | --- |
| The top-n words, n ≤ K | the candidates | exact logits; the sum holds the tail's estimate |
| The probability of a given word outside the list (perplexity, scoring a multiple choice, a forced continuation) | `a × preview + b`, marked as estimated | an estimate: this is the number the dBPB line judges |
| The same, with its row read | exact logit, one row more (3.4 KB on flash) | numerator exact; arms `row_*` measure what is left |
| A constraint that leaves no candidate (grammar, logit bias) | no answer from this pass | needs the whole head once, 16.5 ms on flash's desk |

## Literature (read in the papers)

From test 2b's table: SVD-softmax Eq. 4 keeps preview outputs in the tail, with no correction; Learning to Screen sets the tail to
zero and gives a target outside the list no probability; HiRE does not keep the tail. The affine tail is ours. Its nearest relative
is temperature scaling, "a single-parameter variant of Platt Scaling" (Guo et al., ICML 2017, arXiv 1706.04599; abstract only
read). Our own worry: constants fitted on one distribution need not transfer to another, which is why the text is new.

## Design (`s2u_run.py`, `s2u_verdict.py`, `s2t_two_pass.h`)

**New text:** the 96 documents of `c50_holdout_v1`'s reserved split, 32 per category, 219,666 tokens. The panel's 72 come from the
fit split of the same builder, so they are the same kind of text: same sources, same 8,192-byte span, no shared source file. No
tool has read the reserved split. `sets` rebuilds the panel's 72 lines from the same rendering, 72 of 72. No command is needed.

**Overlap:** 3 of the 96 share a run of 32 tokens with the preview's corpus or the panel. They stay; the result reports the
outcome without them.

**The cell:** tern_tr, K = 1,024, exact candidates, tail a × preview + b.
- a = 1.46626, b = 1.37953 from dev's first 20,000 rows, as test 2b's diagnosis found them (`results/s2u_fit.json`).

**Panels:** a new reference on the new text (P0's records), then the two-pass tool on it, and chat. `--tail-affine` is off by default.

**Float, all on hidden states, reported:**

| Arm | What |
| --- | --- |
| offset | the registered tail |
| mass | same a, b = −0.69153, so the tail's mass is the true one on dev (0.00244; the affine tail holds 0.00862) |
| row_affine, row_mass | the target's exact logit over each tail's sum |
| plant | the affine tail with b − 3 |

On the new text, the 72 documents and ternary-expert states.

## Line (registered; `s2u_verdict.py`, selftest PASS, 10 cases)

Test 2b's own, on the 96 documents: miss ≤ 1 per 1,000 (tool and float), KL top-32 ≤ 0.02, KL over the vocabulary ≤ 0.02, dBPB
CI95 upper ≤ +0.005, chat ≥ 27, degenerate ≤ 1.

**PASS: THE HEAD READS IN TWO PASSES ON NEW TEXT**, **FAIL: THE TAIL DOES NOT HOLD ON NEW TEXT**, or **FAIL: THE WORD STAYS OUT OF
THE CANDIDATES**. The 72 are reported, not judged. Test 2b's outcome stays recorded as it is. Top-1 is reported beside dBPB.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Replay | the new binary, tail off, equals test 2b's records on the first 2 documents | VOID |
| Chat | the 40 answers equal test 2b's, token for token: greedy never reads the tail | VOID |
| Parity | float dBPB within 0.001 of the tool's | VOID |
| Plant | the tail with b − 3 leaves the line | VOID |
| Trivial | the offset arm reads more than 0.0005 from the affine one: the tail does something here | VOID |

## Predictions

| | Predicted |
| --- | --- |
| Outcome | PASS 80%, FAIL (tail) 19%, miss 1% |
| dBPB, affine | +0.0012 to +0.0040 |
| dBPB, offset | +0.005 to +0.013 |
| dBPB, mass | +0.006 to +0.012: outside the line |
| dBPB, row_affine / row_mass | under +0.001 / under +0.0005, can be negative |
| KL over the vocabulary | 0.004-0.008 |
| Miss | 0-0.2 per 1,000 |
| Tail's mass, exact, mean / p99 | 0.002-0.003 / 0.02-0.05 |

## What a result changes (rule 5)

- **PASS:** the head has a cell. Test 3b's verdict opens, judged arm x15, and the ceiling stays 25.40 until the idle session.
- **FAIL:** the head has no cell within the line, and the decision moves to the speed line.

## Parts specific to Ling

a and b belong to this preview on this model. The vocabulary sets how far K sits from V. The rule and the mechanism need only a softmax
head and a preview.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Sampling among K in the panels | greedy only | panels are greedy; the truncation's price is the exact tail's mass, measured |
| New text after the model's release | reserved split, same age as the panel's | "of the same kind" as asked; the post-release set is another kind of text |
| Mass and offset arms in the panels | float only | float gives back the tool's numbers (+0.0114, +0.0071); the parity control checks it here |
| The tail on flash | not measured | a and b are the preview's; flash's preview does not exist |
| Ternary experts in the panels | float only, reported | as test 2b |

## Cost

About 1 h for the reference, 2 h for the panels, on the CPU beside the chains; minutes of GPU for the float arms. About 3 h.
