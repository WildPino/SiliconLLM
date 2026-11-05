# Stage 2: the fourth code, step 1. RESULT

2026-11-05. Protocol `S2I_FOURTH_CODE_PROTOCOL_20261105.md` (`2fae58e`). Mini, three layers, no training. A test of mechanism.
Records: `benchmarks/stage2/results/s2q_run.json`, `s2q_parity.json`, `s2q_verdict.json`.

## Outcome: THE FOURTH CODE PAYS ON SOME

At equal bytes, the fourth value cuts the routed experts' block loss by 38%. On attention and the shared expert it does not reach the line.

| Organ | Best form | G (line 0.25) | Other form | Predicted | Passes |
| --- | --- | ---: | ---: | --- | --- |
| Routed experts | bal | **0.380** | asym 0.260 | 0.30-0.42 | yes |
| Shared expert | asym | 0.185 | bal 0.177 | 0.10-0.25 | no |
| Attention | asym | 0.154 | bal **−0.291** | 0.12-0.22 | no |

Block loss by layer (test 2c's loss, one organ at a time):

| Layer | Attention tern / asym / bal | Shared tern / asym / bal | Experts tern / asym / bal |
| --- | --- | --- | --- |
| 5 | 0.329 / 0.270 / 0.431 | 0.094 / 0.072 / 0.074 | 0.149 / 0.109 / 0.090 |
| 10 | 0.531 / 0.435 / 0.705 | 0.308 / 0.257 / 0.279 | 0.290 / 0.215 / 0.178 |
| 15 | 0.698 / 0.624 / 0.862 | 0.151 / 0.128 / 0.117 | 0.469 / 0.352 / 0.303 |

## What the controls say

- **Kernel:** code 3 reads +2. On 30 cases (IQ2_BN and IQ2_BN_R4, five shapes, batches 1, 4, 9) relative error is 0.0063-0.0096 for four, neg and
  bal, against tern's 0.0064-0.0085. The planted miss reads 0.51-0.63. It holds.
- **Planted:** each form's quantizer recovers 100% of weights on its own grid, at the noise's error. Identity passes.
- **Trivial, and it matters.** The experts' output gain with bal is 0.377-0.384; the Gaussian twin's is 0.365-0.386. So the gain on experts is
  what any bell shape gives at equal bytes, no more. It is real, and it carries no finding about Ling. Attention and shared stay under their twin
  (asym 0.29-0.33 and 0.27-0.31 against 0.37-0.39): their inputs have large channels, which plain row error ignores.
- **Bal on attention is the worst form,** with the least weight error (0.128 against 0.200). The dry run's reading holds: no zero, so small weights on
  large input channels become ±0.5·s. Weight error does not rank forms here.

## Other numbers

| | Measured | Predicted |
| --- | --- | --- |
| Rotation between V and o, block gain | tern 0.024, asym 0.029, bal −0.025 | 0.02-0.10: at the low edge, and it does not pay |
| Flash, weight gain of bal | experts 0.374-0.375, attention 0.365-0.370, shared 0.296-0.336 | 0.30-0.40: shared's layer 5 is just under |
| Flash, weight gain of asym | 0.216-0.233 | not predicted |

Flash reads weights only. Mini's attention shows that weight gain need not reach the output.

## What this does not say

- Post-hoc block loss is not panel quality, and test 3 showed a large raw error that the norms remove.
- After training the order can change: ParetoQ's 2-bit against 1.58-bit figures follow training, and nothing here does.
- Bal needs −0.5·s·sum(x) per row. Parity covers its value, not its time. Test 5's cell 7 times the four codes, not that correction.
- The plain row quantizer may be what sinks bal on attention. An input-weighted one was left out by the protocol.

## Thesis line

The bytes do not change, so the desk's speed does not either. This brings the thesis no closer by itself. It names the form to try on the organ that
holds most of the weights: 15.3 B on mini.

## Recommendation

**Step 2 trains the fourth code on the routed experts only, form bal, against ternary at equal budget.** Attention and the shared expert stay ternary,
and the rotation is dropped. Step 2 waits for the go, and comes after test 2c's result, which it needs as a reference.

**What would disprove it:** if trained ternary and trained bal end within test 2c's seed noise on block loss, the gain was only the untrained
quantizer's, and part 2 stays ternary with the kernel as it is.

## Parts specific to Ling

- **Specific:** the numbers, attention's large input channels, and the 4 KV heads of the rotation.
- **Not specific:** the experts' gain (a Gaussian gives the same), the grids, the kernel's reading of code 3.
