# Stage 2, part 1, test 2b: the head in two passes. RESULT

2026-11-05. Protocol: `S2D_TWO_PASS_HEAD_PROTOCOL_20261104.md`. Verdict: `benchmarks/stage2/results/s2t_verdict.json`.

**Outcome (registered, cell 1): FAIL, THE PANELS LEAVE THE LINE. Cell 2: ONLY THE TRAINED PREVIEW IS WITHIN THE LINE.**
Every control passes. The word never stays out of the candidates: cell 1 fails on one line, dBPB, and that line is about the
tail. The line stays where it was registered.

## Numbers

| Check | Line | Cell 1: tern_tr K=1,024 | Cell 2: wsvd256 K=4,096 | Reported: tern_im K=4,096, offset off |
| --- | --- | ---: | ---: | ---: |
| Miss per 1,000, documents, tool | ≤ 1 | **0** | 6.77 [4.98, 8.78] | 0.41 |
| Miss, generated, at 4,096 | ≤ 1 | **0** | 1.57 | 0.27 |
| Miss, ternary-expert states, at 4,096 | ≤ 1 | **0** | 1.96 | 0.25 |
| KL top-32, nats | ≤ 0.02 | **0.00009** | 0.144 | 0.0048 |
| KL whole vocabulary, nats | ≤ 0.02 | **0.0034** | 0.148 | 0.0065 |
| dBPB, CI95 | upper ≤ +0.005 | **+0.0114 [0.0090, 0.0139]: out** | +0.0737 | +0.0072 [0.0055, 0.0091] |
| Chat, degenerate | ≥ 27, ≤ 1 | **28, 0** | 28, 0 | 28, 0 |
| Top-1 agreement | reported | 1.0000 | 0.9932 | 0.9996 |
| Items touched: nq, tqa, gsm8k, chat | bound | 0, 0, 0, 0 | 5, 11, 23, 9 | 5, 2, 0, 3 |
| Desk ceiling, tok/s (line 24.7) | | 25.40 | 26.18 | 25.19 |

Controls: planted 976.6, frequency 402.6, wsvd128 17.3 misses at 4,096; parity 0 on all three sets; replay equal, plant top-1
0.00145; identity holds.

Curve, misses per 1,000 on the documents, float, at K = 4,096, with the preview's MB on mini: tern_tr 0 (81.1), tern_im 0.33
(81.1), wsvd512 2.15 (83.2), wsvd256 6.77 (41.9), wsvd128 17.3 (21.3). The curves at K = 16-4,096 on each set are in
`results/s2t_sweep_*.json`.

## Predictions against the outcome

| | Predicted | Outcome |
| --- | --- | --- |
| Cell 1 | THE HEAD READS IN TWO PASSES, 85% | **wrong**: panels out |
| Cell 1 miss at 4,096 | documents 0-0.2, generated 0-0.5 | 0, 0 |
| Cell 1 panels | top-1 ≥ 0.999, KL top-32 0.001-0.006, \|dBPB\| ≤ 0.002, chat 27-29 | 1.000, 0.00009, **+0.0114**, 28 |
| Cell 2 label | UNTRAINED PASSES TOO, 45% | fails |
| Cell 2 miss at 4,096 | documents 0.4-1.5 | **6.77**: the whitening fitted on the corpus does not reach the documents (dev 0.48) |
| Items touched, cell 1 | 0-1 | 0 |

**A correction to the protocol.** It says the KL and BPB lines "clear both cells" on dev. Only KL had been read there; dBPB
never was. On dev's unused rows cell 1 reads +0.0055, already outside.

## Where the dBPB comes from (after the verdict; a diagnosis, it moves no line)

Real text picks words the model finds unlikely. Targets outside ref's top-32 are 7.1% of positions and carry all of it: +0.0283
nats per token from them, −0.0002 from the rest. So KL (weighted by the model's own belief) is small and dBPB (weighted by the
text) is not. Float gives back the panels: +0.0114 and +0.0071 (`s2t_tail_probe.py`).

The registered tail shifts the preview by the candidates' mean. In the tail the trained preview's scale is compressed: exact ≈
1.47 × preview + 1.38, mass-weighted, on 20,000 dev rows. Keeping the same candidates and changing only the tail:

| tern_tr K=1,024 | Offset (registered) | Affine, a and b from dev |
| --- | ---: | ---: |
| dBPB, documents | +0.0114 | +0.0024 [0.0017, 0.0031] |
| dBPB, dev's 54,867 rows that took no part | +0.0055 | +0.0017 |
| KL whole vocabulary, documents | 0.0034 | 0.0057 |
| Tail's mass, composed (exact 0.0021) | 0.0017 | **0.0066** |

- **The price:** the affine tail holds three times the true mass, so a sampler at temperature 1 lands in the tail three times as
  often. It hedges more than it corrects. Greedy decoding is the same in both.
- **Selection:** the documents saw the choice among four tails. Two scalars and the unused dev rows limit that; they do not
  remove it.
- **Rule 3:** a trivial transform of the tail takes 79% of this score, so dBPB on its own does not decide here. That reads the
  line; it does not change the outcome.
- **tern_im does not follow:** affine is +0.0034 on the documents but +0.0064 on the unused dev rows, with KL 0.0155.
- **More candidates do not pay:** tern_tr K=4,096 with the registered tail is +0.0058, still out, and K=16,384 (+0.0024) costs
  about 1.7 ms of rows against 1.12 ms of room. An int4 preview is within (+0.0012) but reads 22.1 tok/s on the desk.

## What it says for the thesis

The two-pass mechanism works on Ling-mini: with the trained preview no word is lost on any set, and greedy answers are ref's
byte for byte. What fails is the probability given to words outside the list. No untrained preview is within the line, so flash
would pay for a head training. The desk's 25.40 tok/s still rests on selection and gather times not yet measured at idle;
contended, they read 1.2-1.9 ms and 0.34 ms against 0.2 and 0.11, which could take the 1.12 ms of room.

## Parts specific to Ling

The vocabulary (157,184), and a and b, which belong to this preview on this model. Not specific: the mechanism, and the finding
that a preview trained on the top of the distribution compresses its tail.

## Recommendation (rule 7)

**Do not move the speed line yet. Run one test, 2b-bis:** cell 1 with the affine tail, a and b from dev, registered and pushed
first; documents and chat panels with the same line, plus the miss-free ternary-expert states read for dBPB. About 3 h of CPU,
no training, no milliseconds added.

**What would refute it:** dBPB's upper bound over +0.005 on the panels, KL over 0.02, or chat under 27. Then the head has no cell
within the line, and the decision moves to the speed line as registered.
