# E25 — what the rank actually costs, in the engine

**Pre-registration. Nothing in §§4–8 has been measured.** Part A's implementation exists and its
gate is reported in §3; the numbers this brief registers are part B's.

---

## 0. Why this exists, in one paragraph

E21 §8, E22 §8, E23 §9 and the T4 proposal §6 all end with the same owed item: **`engine.c` has
no factored matvec.** E21 and E22 measured the rank cut by computing `A·B` and installing a
DENSE matrix, so every rank result this programme published priced the cut's **quality** and
never its **cost** — the tok/s consequence of the rank axis has been arithmetic from the day the
axis was opened. The budget table in E18 §31, the `k = 139…156` re-derivation in E23 §7 and the
whole reason `q/o` at rank 512 is in the plan all rest on `2·D·r < D²`, which nothing has
executed. **E25 builds the path and measures it.** It makes no quality claim of any kind.

## 1. What was built (part A)

A **factored matrix kind** in `donor_engine.c` and a **tagged container** to carry it.

- `mat_t` gains `rank`, `fa`, `fb`, `fs`. `matvec` dispatches on `rank`: `y = A·(s ⊙ (B·x)) + b`,
  two calls to the SAME kernels with an intermediate of size `r` in its own buffer. `rank == 0`
  is every matrix written before this existed, and that path is untouched.
- **`quant == 3`, the tagged layout**: every matrix carries its own `int32` kind — `0` packed,
  `1` factored, `2` fp32 — so one file can hold packed FFNs, fp32 `k/v` and factored `q/o`.
  That mix is not a convenience: it is exactly the object E21/E22/E23 measured (`k/v` stay fp32)
  and the object H0 trains, and a single global `quant` flag cannot express it. `quant` 0/1/2
  stay untagged and byte-identical.
- `--fuse` refuses a factored matrix (its two factors share no output row space); `--lut`
  already requires `quant == 2` and therefore declines it. Both refuse at load, not at run.

`s` multiplies the **intermediate**, never a factor: folding it into `B` would change `B`'s
per-row ternarization and folding it into `A` would change `A`'s, and the point of carrying it
is that H0's trained scale is fp32 while the factors are not.

## 2. Gates on the construction

| gate | what it asserts |
|---|---|
| **`G-E25a`** | the patched engine is **bit-identical** to the unpatched one on an existing `quant=2` model (sha256 of 10×151936 logits). Legacy paths cannot have moved. |
| **`G-E25b`** | `GATE V3` / `GATE E25-L`: the written file's byte count equals `e1_bpb_through_engine`'s independent size model for the tagged layout, which is derived from the format and knows nothing about the writer. |
| **`G-E25c`** | the engine's own `consumed exactly N bytes` layout check. |
| **`G-E25P`** | **end-to-end parity**: the engine on the exported artifact vs PyTorch with `h0_qat.TernaryLowRank` installed on the same factors — worst relative l2 `< 2e-3` and top-1 agreement `1.0`, `parity_gate.py`'s bar unchanged. |

`G-E25P` is the one that matters. Phase 60's law — *kernel-bit-exact does not compose to
system-correctness* — means a new kernel is gated end to end or not at all.

## 3. Result of part A

*(filled in from the runs; §§4–8 were written before any timing existed)*

## 4. Part B — the arms

Timing uses `synth_export.py`, whose weights are **noise**. That is deliberate and it is the
only honest way to price a shape: E3 §4's Gate V1 planted `--codes zero` against `--codes dense`
on one shape and found no timing difference, so time here depends on shape and format and not on
values. **No arm below says anything about quality. Not one.**

Two shapes: **`S15`** = Qwen2.5-1.5B's shape, where the real factors exist, and **`T10`** = the
goal's "es 10B" shape, where the lever is worth something. `--head ternary` throughout, which is
the runnable configuration the ledger's rates were measured on.

| arm | shape | layout | `q/o` | active weights/token |
|---|---|---|---|---|
| `S15-PACKED` | S15 | `quant=2` | dense | 1.5436 G |
| `S15-TAG-R0` | S15 | `quant=3` | dense | 1.5436 G |
| `S15-R768` | S15 | `quant=3` | **r = 768** | **1.5436 G** |
| `S15-R512` | S15 | `quant=3` | r = 512 | 1.4995 G (−2.85%) |
| `S15-R256` | S15 | `quant=3` | r = 256 | 1.4555 G (−5.71%) |
| `T10-TAG-R0` | T10 | `quant=3` | dense | 10.6032 G |
| `T10-R2048` | T10 | `quant=3` | **r = 2048** | **10.6032 G** |
| `T10-R512` | T10 | `quant=3` | r = 512 | 9.3952 G (−11.39%) |
| `T10-R256` | T10 | `quant=3` | r = 256 | 9.1936 G (−13.29%) |

## 5. The two controls, and why the bold rows are the point

- **`S15-PACKED` vs `S15-TAG-R0`** — the same weights in the untagged and tagged containers.
  The tag is 4 bytes read once at load. **If these two differ by more than the dispersion, the
  container is not free and every tagged number carries that offset.**
- **`S15-R768` and `T10-R2048` — THE PLANTED CONTROL.** At `r = out·in/(out+in) = D/2` for a
  square projection the factored form moves **exactly as many weights** as the dense one:
  `2·D·(D/2) = D²`. These arms are byte-neutral by arithmetic. **Whatever they differ from the
  `R0` arm by IS the factored path's own overhead — the second matvec call, the extra OpenMP
  region, the intermediate — and it must be subtracted from every other rank's gain before that
  gain is called a saving.** Without this arm a rank speedup and a container speedup are the
  same number.

`--rank R` doubles the attention projection calls: `matvec` calls per token go `L·7+1` → `L·9+1`
(S15: 169 → 225; T10: 337 → 433). `donor_engine.c`'s own `--fuse` comment records why that
matters: `k_proj` and `v_proj` are 128 output rows, which across 6 threads is 21 rows and
**~4.6 µs of real work per fork** — so the region is not large compared with the cost of opening
it. The per-fork overhead itself has never been measured here and the 5 µs used in §7 is an
ESTIMATE of that order, marked as one. **The planted control is what makes the estimate
unnecessary**: it measures the overhead instead of assuming it.

`--fuse` is OFF in every arm, including the `R0` controls. It has to be: a factored `q_proj`
cannot be fused, so leaving it on for the dense arms would compare 97 regions against 225.

## 6. Protocol

**A contended timing is not a timing.** Idle machine, nothing else running, `--threads 6`,
`--bench` with a fixed token count, **≥ 3 repetitions per arm**, and the **dispersion reported
with every rate**. Every absolute tok/s carries ±5%; the **ratios inside a shape do not**, which
is why the verdict below is stated as a ratio. Arms are run interleaved by repetition, not
blocked by arm, so a thermal drift hits every arm equally.

## 7. Predictions — fixed here, before the run

1. **`S15-PACKED` and `S15-TAG-R0` agree within dispersion.** The tag is read at load.
2. **`S15-R768` and `T10-R2048` are SLOWER than their `R0` arms by 0–2%.** Byte-neutral, plus
   56 (S15) / 96 (T10) extra OpenMP regions per token. At ~5 µs a fork that is 0.28 ms and
   0.48 ms per token, against token times of order 32 ms and 221 ms — so 0.9% and 0.2%.
3. **`T10-R512` is FASTER than `T10-TAG-R0` by 10–13%**, against the byte prediction of
   `1/(1−0.1139) = +12.85%` minus prediction 2's overhead.
4. **`S15-R512`'s gain is NOT RESOLVABLE.** The byte prediction is `+2.93%` and this box's
   dispersion on the engine has been 4.0%. Registering that in advance is the point: a 3%
   effect under a 4% spread must be reported as unresolved, not as a small win.
5. **The ordering `R256 > R512 > R0` holds at T10 and does not resolve at S15.** The byte
   predictions are T10 `+15.33%` / `+12.86%` / `0`, and S15 `+6.05%` / `+2.94%` / `0`; only
   T10's three are separated by more than this box's dispersion.

Every "byte prediction" above is `active(R0) / active(r)` computed by `e25_rank_cost.active()`,
which mirrors `synth_export.active_weights`; the planted controls come out byte-identical to
the last weight (`1,543,569,408` at S15, `10,603,200,512` at T10), which is what makes them
controls rather than approximations.

**THE REGISTERED ALTERNATIVE.** *If `T10-R512` fails to beat `T10-TAG-R0` by at least 8% — i.e.
the measured gain falls short of two thirds of the byte prediction — then the factored form is
paying a per-call cost the byte ledger does not see, the rank axis is cheaper on paper than in
the engine, and **every budget table that charges a rank-r projection at `2·D·r` is overstating
what it buys** — E18 §31, E21 §8, E23 §7 and the T4 proposal's depth arithmetic all inherit the
correction. Written before the data exists.*

## 8. What E25 will NOT be able to claim

- **No quality claim.** Part B's weights are noise. The rank that is *good* is E21/E22/E23's
  question and remains `r/D = 1/3` validated at `D = 1536` only (E21 §8).
- **No claim that `T10` is a model.** It is a shape with the goal's dimensions and random
  weights. A 10 B that is worth running does not exist here.
- **No revision of `6.79 tok/s`.** That is the real 7.072 B packed donor, a different artifact
  and a different shape; nothing here touches it.
- **`k/v` and the head are untouched** in every arm, as in E21/E22/E23.
- **One box.** AMD Ryzen 5 3600X, Zen 2, 6c/12t, L3 32 MB, `--threads 6`. The portability law
  applies: this is the reference floor, not the class.
