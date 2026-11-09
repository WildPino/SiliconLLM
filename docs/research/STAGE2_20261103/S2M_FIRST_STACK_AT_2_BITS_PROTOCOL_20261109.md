# Stage 2, part 2, second test: the first stack at 2 bits but the head. PROTOCOL

2026-11-09. Ling-mini. CPU only, no training. Code: `s2z_ks.py`, `s2z_export.py`, `s2z_run.py`, `s2z_chain.py`, `s2z_verdict.py` (selftest PASS, 11 cases); S2L's `s2y_export.compose` takes values written before quantizing (`pre`).

## Question

S2L found IQ2_KS, post hoc, the best start for the organs (top-1 0.795, chat 30), and found the first stack worse than its parts by 5.70 points. Three readings: what ks_org costs in points; what the stack of bal's experts and IQ2_KS organs costs, and whether it compounds as stk4 did; and whether bal's training carries the experts, or IQ2_KS post hoc does as well.

**Judged on points, chat and top-1, never on block loss.**

## Thesis line (rule 4)

On the desk, with ik's attention: **ks_stk4 43.05 ms, 23.23 tok/s** (41.29 + 0.25 for bal's correction + 11% of the organs' 13.74 ms); **ks_all4 44.32 ms, 22.56 tok/s** (11% of the routed experts' 13.77 ms too, no correction). Sums of cells, not a run on flash. Neither reaches 40.49 ms: speed stays stage 3's matter. If ks_stk4 is within budget, flash's organs need an imatrix and no training, and only the experts take bal's training (S2E: 86 h on the 3060 at 16.8 M positions).

## Arms

| Arm | What | Panels |
| --- | --- | --- |
| **ks_org** | S2L's file, unchanged (sha `db9d8c3a`): the organs at IQ2_KS with the imatrix, in BF16 | adds nq, tqa, gsm8k, arc, mmlu at k = 8; documents and chat as S2L ran them |
| **ks_stk4, judged** | bal-q4's experts and router (Q4_0, bytes copied), the organs at ks_org's values, the head stored | all seven, generation at k = 4 |
| ks_all4, control | the organs as ks_stk4, the routed experts at IQ2_KS with the imatrix, post hoc, in Q8_0 | all seven, generation at k = 4 |

- **The experts' imatrix** is fit24's mean per expert, in_sum2 / counts. 9 of 14,592 expert slots were never routed in calibration; each takes its tensor's mean over the others.
- **Q8_0 holds IQ2_KS's levels to within a rounding:** it is checked as a container, not bit for bit. Its activations are Q8_0, as bal's Q4_0.

## Lines (registered; `s2z_verdict.py`)

**ks_org:** stage 1's label and chat gate; its margin over as_both with a paired CI95, reported.

**Stack, judged:** stage 1's label and chat gate on ks_stk4. Excess = ks_stk4 − bal4 − ks_org, CI95: **IT ADDS UP**, **WORSE THAN ITS PARTS** (upper < 0), **BETTER THAN ITS PARTS** (lower > 0). S2L's disproof, as written there: **THE COMPOUNDING, NOT THE FORMAT** if WORSE and the excess is at most stk4's plus S2L's seed gap, −5.70 + 1.90 = −3.80; else **THE FORMAT KEEPS ITS GAIN IN THE STACK**.

**Control:** margin = ks_stk4 − ks_all4, paired CI95: **BAL'S TRAINING PAYS** (lower > 0 and margin > 1.90), **IQ2_KS BEATS BAL** (the mirror), **NOT SHOWN** otherwise.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Exports | the organs read back bit for bit; every other tensor equals its reference's bytes (bal-q4 for ks_stk4, P0 for ks_all4) | VOID |
| Container | per expert tensor, Q8_0's squared error against the IQ2_KS values under 1% of theirs against the original; selftest: Q8_0 passes, the original weights and a 5% scale fail | VOID |
| Imatrix arrived | ik's log loads 176 entries and names no organ or expert without weights; planted: S2L's log, which left the experts out, fails it | VOID |
| Organs unchanged | every organ tensor of the new IQ2_KS file equals S2L's byte for byte | VOID |
| 2-bit | at most 4 values per block of 32 in the experts; relative weight error reported | VOID |
| Tool | S2L's tool check holds for these binaries | |
| Trivial | ks_all4 for bal's training; as_both for ks_org | |

## Predictions

| | Predicted |
| --- | --- |
| ks_org | Δ −5 to −1, chat gate passes; WITHIN 35%, UNDECIDED 60%, OVER 5% |
| ks_stk4 | Δ −13 to −6, chat 15-25; OVER 55%, UNDECIDED 43%, WITHIN 2% |
| Excess | −5 to −1; WORSE 55%, ADDS UP 45%; THE COMPOUNDING 30% |
| ks_all4 | Δ −14 to −6; NOT SHOWN 55%, BAL PAYS 25%, IQ2_KS BEATS 20% |

## What a result changes (rule 5)

- **ks_stk4 WITHIN BUDGET:** it is the configuration to beat, and the stage goes to its delivery with the speed gap as what is left.
- **UNDECIDED or OVER:** part 2 trains this stack together, the organs starting from IQ2_KS's values and the experts in place, at k = 4. Its protocol passes the reviewer (rule 6).
- **THE COMPOUNDING, NOT THE FORMAT:** the same training, but the organs' start is not counted on to carry it, and the budget goes to the whole stack.
- **IQ2_KS BEATS BAL:** the experts go IQ2_KS post hoc, at +1.51 ms; bal's writer leaves the engine's plan. **PAYS or NOT SHOWN:** bal stays, at 0.25 ms.
- **ks_org's Δ** places the organs' share of the 5 points.

## Literature (rule 2)

IQ2_KS has no paper; read in ik_llama.cpp's `iqk_quantize.cpp` and `ggml-common.h` for S2L. Its known way to fail is the imatrix: on experts, an expert seen by few calibration tokens gets a thin imatrix. Here the median slot saw 3,192 tokens and 9 saw none. Bal and its training: S2I, S2K.

## What this takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| Head at 2 bits | stored (Q6_K) | the two-pass head ran apart and passed (+0.0032); one change at a time |
| IQ2_KS's own kernel | experts in Q8_0, organs in BF16 | llama.cpp's panel tools do not read IQ2_KS; containers moved under 0.3 points in S2L |
| The stack trained together | not trained | this is its start; the training follows if it is needed |
| Seeds | none | no training: the arms are deterministic |
| Imatrix at k = 4 | fit24, routed at k = 8 | the calibration in hand; generation runs at k = 4 |
| Excess at one k | ks_org at k = 8 inside it | as stk4's, so the two excesses compare |
| lr of the shared expert | closed | by decision of 2026-11-09 |

## Rule 6

No GPU, no new primary metric: the reviewer is not required. The next protocol, if it trains, takes one.

## Parts specific to Ling

The numbers, the 9 slots without calibration, the shared expert, one dense layer, the fused QKV. Not specific: IQ2_KS, the imatrix per expert, the container check, bal.

## Cost

CPU about 13 h: ks_org's five panels 2.5 h; ks_stk4 1 h of export and 3.5 h of panels; IQ2_KS on the experts about 2 h beside it; ks_all4 1.5 h and 4.5 h. About 90 GB on E:.
