# E5 - Decomposing `R`, the 84% of the attention organ E4 could not see inside

**Status: CLOSED, `OVERHEAD-DOMINATED`.** Brief: `briefs/BRIEF_E5_DECOMPOSE_R.md` (14 sections,
every one pushed before the run it governs). Verdict from run 6.

| | |
|---|---|
| shapes | `S05` (0.5B: 896, 4864, 24, 14/2, 64, 151936), `T10` (10.6B: 4096, 14336, 48, 32/8, 128, 32768) |
| contexts | `--bench 300`, `--bench 800`, 6 threads, `HIGH_PRIORITY_CLASS` |
| build | `clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp` - never `-ffast-math` |
| runs | 6 sweeps. **1-4 VOID** (see section 5 below), 5 partial, **6 is the verdict** |
| data | `results/e5/{sweep6.json, analysis6.txt, parity6.txt, run6.log}` |
| analyser | `engine/e5_analyse6.py` (run 5's is frozen as `e5_analyse5.py`, run 3's as `e5_analyse3.py`) |

---

## 1. The result

At `T10` @800 - 10.6 B active parameters, 800 tokens of context - the attention organ is
`X + R`, where `X` is the `Q.K` dot loop E4 optimised and `R` is everything after it.
Median over five runs, each solved independently:

| term | ms/token | share of `R` | band across runs |
|---|---|---|---|
| `X` - the `Q.K` dot loop | **2.253** | - | 2.153 - 2.625 |
| `S` - the softmax pass | **2.953** | **25.3%** | 24.0 - 26.0% |
| `Y` - the `A.V` loop | **2.406** | **20.6%** | 19.1 - 22.0% |
| `P` - everything that is neither loop | **6.317** | **54.0%** | 52.0 - 56.9% |
| `R` | **11.641** | 100% | 11.306 - 11.697 |

**More than half of `R` is not a loop.** The two loops together - the softmax and `A.V`, the only
two things in the per-head body that anyone would think to optimise - are 46% of it.

Because the absolute organ drifted between sweeps (section 4), **the shares are the result** and the
ledger takes them applied to E4's own `R` = 9.816 ms:

    S = 2.485 ms      Y = 2.020 ms      P = 5.301 ms        (E4's sweep, E5's shares)

## 2. The shape changes with scale

| cell | `S`/`R` | `Y`/`R` | `P`/`R` | label |
|---|---|---|---|---|
| `T10` @800 (10.6B) | 25.3% | 20.6% | **54.0%** | OVERHEAD-DOMINATED |
| `T10` @300 | 25.5% | 19.0% | **54.6%** | OVERHEAD-DOMINATED |
| `S05` @800 (0.5B) | **47.7%** | 13.4% | 39.2% | SOFTMAX-DOMINATED |

A softmax fix measured on the small model would look like a 48%-of-`R` win and arrive at target
scale worth half that. **The donor-adaptation programme has now been bitten from both sides by
measuring at the wrong scale; this row is the cheapest available warning.**

## 3. What `P` is not

| candidate | priced at | share of `P` |
|---|---|---|
| the OpenMP fork/join (one extra region per layer, `fork2`, run 5) | +0.087 ms | **1.4%** |

`P` is **not** the fork. What remains inside it - the 32-heads-over-6-threads imbalance, per-head
address arithmetic, `out[]` initialisation, the `mx` reduction - is **unattributed**. E5 measured
`P`'s size and deliberately does not name its parts. **That is the next experiment.**

## 4. What E5 does NOT claim

- **Not a new ceiling.** Run 6's organ `none` is 13.931 ms against E4's 12.058 for the same cell,
  +15.5%, and `f` follows. `1000/f` = 68.4 tok/s is **not** a correction to E4's **78.5** and must
  not be quoted as one. Ratios travel between sweeps; absolutes do not.
- **Not a cause for `S`.** `S` came in 48% above the band pre-registered for it. `expf` on this
  toolchain is `vcvtss2sd -> callq exp -> vcvtsd2ss` (`results/e5/asm_notes.txt`, written before any
  arm was timed) - 615,168 double-precision calls per token at this cell. That is a **candidate**,
  not a finding. Testing it means an arm that is not value-preserving, with its own parity.
- **Not a result at `S05` @300.** That cell never survived G0 in any run.

## 5. How the instrument was built, and what it cost

Four sweeps were thrown away, each for a reason the next one fixed. This is the useful part of E5
for anyone measuring anything on this machine.

| run | design | outcome |
|---|---|---|
| 1 | arm-major, separate processes | **VOID** - the untouched weight path `W` moved +30.4% across one cell; three arms returned NEGATIVE components |
| 2 | rep-major | **VOID** - interleaving passes worked, but only one pass in three was clean |
| 3 | rep-major, 6 passes, G0 at 5% | **VOID** - G0 passed everywhere and G1 still failed: `(av2-none)` and `(av3-av2)` are both one extra pass and disagreed by 1.28x, because `none` was a *different code shape* from the wrapped arms |
| 4 | + `sm1`/`av1` (1x inside the wrapped path), G0 at 1%, high priority | **VOID** - all four cells. 1% of a 301 ms weight path is 3.0 ms and the component is 1.9 ms: **no between-process tolerance can be both protective and passable** |
| 5 | **all arms interleaved per token inside ONE process** | `S` and `Y` measured. `X` came out **negative** at @300 - `R` was built across two code families and absorbed their cold-start difference |
| 6 | `X` moved inside the family (`qk1/qk2/qk3`); `d` measured | **VERDICT** |

Five laws came out of it, all of them cheap to reuse:

1. **A between-process difference cannot resolve a component smaller than the gate's tolerance times
   the untouched path.** Write that product down before designing the sweep.
2. **Interleave per token, and balance the schedule by construction.** The palindrome
   `arm = idx < n ? idx : 2n-1-idx` gives every arm exactly equal mean position; a round-robin gives
   arm 0 systematically shorter contexts, a bias of order 1/n on the thing being decomposed. Inside
   one process the ten arms agreed on `W` to **0.41%**, against 8.5% worst between processes.
3. **Never subtract across code families.** `S` and `Y` are differences inside one family and passed
   every test; `R` and `X` crossed families in run 5 and produced a negative duration.
4. **The 1x point must live inside the same code shape as the 2x and 3x points**, or the difference
   is "one extra pass plus a change of shape" and the 3x test will catch you.
5. **A duration must be positive** (gate G5). Obvious, ungated for five runs, and the thing that
   finally exposed run 5.

## 6. The gates, and that they could have failed

| gate | what it tests | run 6 |
|---|---|---|
| G2 | every arm is value-preserving, **while mixed**: `--sweep6`, `--sweepd`, pure `qk3` | PASS, all `166667.1361128952` |
| G0 | `W`, which no arm touches, agrees across the arms of one process, within 1% | PASS on 3 of 4 cells |
| G5 | every component is a positive duration | PASS |
| G1 | **nine** 3x predictions the components were not fitted to, tolerance 3% | PASS, worst **-1.15%** |
| G4 | a leading share within 3 points of 0.50 decides nothing | 54.3%, clear - the label issues |
| `d` | the price of switching arms, from identical code in different neighbourhoods | **+0.1%** of the organ |

## 7. Corroboration that was not arranged

`X` = **2.253 ms** here, by doubling the `Q.K` loop inside the `avx4` family in one process.
E4 measured **2.242 ms** by a between-process subtraction across the `serial` family, on another
day, at another priority. **0.5% apart, no shared arm, neither fitted to the other.**

## 8. Owed

1. **Decompose `P`.** 54% of `R`, unattributed, and the fork is only 1.4% of it.
2. **`f` beyond 800 tokens of context** - E4's item 3, still unmeasured, still bounding nothing.
3. **A single-precision `expf` arm**, if `S` is worth attacking - not value-preserving, so it needs a
   quality gate E5's arms never needed.
