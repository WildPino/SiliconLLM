# BRIEF E51 — THE SOFTMAX WAS PAYING FOR `errno`

**Registered before any E51 cell exists.** No speed number here may be quoted until `G-E51a`
(BPB parity) and `G-E51b` (the discrete greedy test) both pass. E50 is the precedent for why both:
a scalar average over 12,264 positions does not compose to a discrete argmax.

## 1. The finding, read out of the compiler and not assumed

The baseline softmax is three lines of `donor_engine.c`:

    float sum=0.0f;
    for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }

Compiled at the project's own flags (`clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp`), that
inner loop is **eleven instructions per element**, `.LBB12_145`:

    vmovss   4(%r14,%rsi,4), %xmm0      # load a[t]
    vsubss   %xmm10, %xmm0, %xmm0       # - mx
    vcvtss2sd %xmm0, %xmm0, %xmm0       # float -> DOUBLE
    vzeroupper                          # AVX state flush, PER ELEMENT
    callq    exp                        # libm, DOUBLE precision, PER ELEMENT
    vcvtsd2ss %xmm0, %xmm0, %xmm0       # double -> float
    vmovss   %xmm0, 4(%r14,%rsi,4)      # store a[t]
    vaddss   %xmm0, %xmm11, %xmm11      # sum += ...  (serial dependency chain)
    movslq   (%rbx), %r15               # RELOAD the loop bound FROM MEMORY, per element
    incq %rsi / cmpq %r15, %rsi / jl

Four separate costs, none of them the exponential itself:

1. **`expf` is promoted to double `exp`.** The source says `expf`; the object calls `exp`, with a
   `vcvtss2sd` in and a `vcvtsd2ss` out.
2. **A `vzeroupper` per element**, because the call may clobber the upper YMM state.
3. **The loop bound is reloaded from memory on every iteration.**
4. **The sum is a serial `vaddss` chain** — the same shape E8 found in the weight matvec.

### 1.1 Why, exactly — and it is one cause, not four

**`exp` is specified to set `errno`.** That makes it a memory-writing call, so the compiler may
not treat it as pure: it cannot keep the loop bound in a register across it, cannot unroll, cannot
vectorise, and must assume the AVX state is clobbered. **One promise nobody in this programme has
ever wanted is buying all four costs.**

Recompiling with **`-fno-math-errno`** and nothing else, the same loop becomes:

    vmovss   -12(%r12,%r14,4), %xmm0
    vmovss   -8(%r12,%r14,4), %xmm12
    vsubss   %xmm10, %xmm0, %xmm0
    vzeroupper
    callq    expf                        # SINGLE precision -- both vcvt gone
    vaddss   %xmm0, %xmm11, %xmm11
    ... three more expf, unrolled 4x ...
    addq     $4, %r14
    cmpq     %r14, 336(%rsp)             # bound HOISTED, compared once per FOUR elements
    jne      .LBB12_154

`exp` → **`expf`**, the two conversions gone, the loop **unrolled 4×**, the bound **hoisted**, and
**one `vzeroupper` per four elements** instead of per element. The FFN's SiLU
(`x/(1+expf(-x))`, `.LBB15_3`) has the identical defect and is fixed by the same flag.

## 2. This is NOT `-ffast-math`, and the difference is the whole point

Phase 35 forbids `-ffast-math` and that law is not being bent. `-fno-math-errno` says exactly one
thing: **math functions do not have to set `errno`.** It does not enable reassociation, does not
enable flush-to-zero or denormals-are-zero, does not permit reciprocal or rsqrt substitution, does
not change `-ffp-contract=on`, and does not license any algebraic rewrite. Every arithmetic
operation the engine performs is still performed in the same order with the same rounding.

`-ffast-math` implies `-fno-math-errno` **plus** all of the above. E51 takes the one component
that is a side-effect promise and leaves every component that is a numerics promise.

## 3. What DOES change numerically, stated up front

**`expf(x)` and `(float)exp((double)x)` are different functions** and may differ in the last ulp.
So this change is **not bit-exact** and must not be gated as if it were. Nothing else changes: the
subtraction, the store, the accumulation order and the division are untouched.

That is the same situation E49 was in with `avx4`, and it gets the same treatment — plus E50's
addition, because E50 is the experiment that showed a scalar BPB parity does not settle a discrete
claim.

## 4. The gates

### `G-E51a` — BPB parity, with a planted control that must fire

`--bpb` on S15 (`e37_carved_nf.bin`, real weights, real corpus) at `--carve-k 256`, both builds.

* Bar **`|ΔBPB| < 1e-4`**, i.e. `2e-5` relative on `NATS_TOTAL` (the mapping E49 §4 wrote down).
* **PLANTED CONTROL, and the null does not count without it**: the same harness, unchanged, must
  FIRE on `--carve-k 3` against `--carve-k 256` on the same artifact — E37 measured those
  0.5537 BPB apart, **5,537× the bar**. This is E49 addendum A's control, reused because it is
  the one in this programme proven capable of firing.

### `G-E51b` — the discrete test, because E50 showed parity does not imply it

E6's engine stage, re-run on the new build, scored against E6's **frozen** `results/e6/ref.json`.

* **A1 (fp32 0.5B) must read 160/160.** Anything less and the flag does not ship.
* Controls **A2 at 3/160 and A3 at 10/160**, each within **±2**, or the gate is VOID rather than
  passed — the scorer must be shown to see disagreement.
* Results to `results/e51/`; **E6's own results are not overwritten.**

### `G-E51c` — the speed, judged ONLY where the dispersion permits

**This gate is written under E49 addendum C's rule and is the first to implement it.** Windows
`{40, 160, 640, 1280, 2560}`, 5 reps, both builds interleaved at rep level, `e40_r128.bin
--carve-k 3`.

* **A window is judged only if the observed spread of BOTH arms is at or below the gate's own
  tolerance.** E49 measured 2.3–5.9% at `{640, 1280, 2560}` and 7.9–26.1% at `{40, 160}`.
* **The runner must print `UNRESOLVABLE` instead of a verdict** for any window whose measured
  spread exceeds the tolerance, and must do so from the data of *this* run, not from E49's.
* `b` is refit per build and `C50` recomputed. Absolutes carry the standing ±5%; ratios do not.

### `G-E51d` — attribution: is the win in the softmax, or somewhere else?

A faster engine is not evidence that the softmax got faster. Under `--profile`, with `--attnr sm1`
and `sm2` on both builds, solve S as `(sm2 − sm1)` the way E48 did and compare S between builds.

* **S must fall**, and the drop must account for the bulk of any total improvement.
* **If the total improves while S does not fall, the story in §1 is wrong** even if the number is
  good — the win would then be the FFN's SiLU or something unnamed, and E51 reports that instead.

## 5. My prediction, registered

| quantity | prediction |
|---|---|
| `G-E51a` | **passes**, `ΔBPB` below 1e-5 — a last-ulp difference in one elementwise function |
| `G-E51a` control | fires, ~0.55 BPB |
| `G-E51b` A1 | **160/160 survives.** The fp32 0.5B's top-2 gaps are order 0.1–1.5 nats; a last-ulp move in softmax should not flip an argmax |
| `G-E51c` | `b` falls **10–30%**; `C50` rises from 1443 to **1600–2050** |
| `G-E51c` small windows | **`UNRESOLVABLE` at 40 and 160**, by the rule, not by my choosing |
| `G-E51d` | S falls and accounts for **most** of the total improvement |

The `b` range is wide on purpose. E48 put S at **33.1%** of the `avx4` attention organ, but that
is a SCORE with no RANK partner (`G-E48d` is VOID), the flag removes overhead rather than the
exponential itself, and it helps the FFN's SiLU too — which is a *constant* cost, not a context
cost, so it would move the intercept and not `b`. **A drop in the intercept with `b` unchanged is
a perfectly possible outcome and would still be a win, just a different one.**

## 6. What E51 does not claim

It does not vectorise the softmax — an explicit polynomial `exp2` kernel is a larger,
non-value-preserving change and belongs to its own experiment, which this one is partly meant to
price. It does not touch `-ffast-math`, reassociation or the accumulation order. It does not
revisit `C50` for any shape but R128 `--carve-k 3`. And it does not re-open the headline at short
context: E49 addendum C established that this instrument cannot resolve that window, and E51
inherits that limit rather than pretending otherwise.
