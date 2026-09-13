# E58 — the 1.75× has no address. Where is a trained token's time actually spent, on the engine as it is today?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 1. The question, and why it is the next one

E57 addendum D measured that **`05b_tqh` moves 249.1 MB per token at 84.88 tok/s = 21.1 GB/s**,
against a proj-GEMV streamed floor of **37.0 GB/s** — **1.75× of headroom on the ternary path**,
while the two fp32 arms sit at 38.5/38.7 GB/s with none. D.3 concluded that the remaining engine
work is on the ternary path only and that it **transfers to any healed artifact with the same
byte count**: a healed 1.5B at `15b_tqh`'s bytes, at the floor, reads **47.71 tok/s**.

**That number is worth an experiment and currently has no address.** 1.75× is not a kernel
result; it is a gap between a rate and a bound. E58 asks which organ it is in.

## 2. Checked, not assumed — what the record already holds

Per addendum C.7's amended rule, every source below was **opened**, not listed:

* **`SPEED_LEDGER` §11.4/§12** already holds an organ table for **this exact artifact**:
  `qkv` 12.4 MB/token, `o_proj` 9.6, `ffn` 156.9, `head` 68.1, **total 247.0 MB at 19.942 ms =
  12.4 GB/s aggregate**, with the head the fastest organ at 17.6 GB/s and a registered ceiling:
  *"if every organ merely reached the head's own 17.6 GB/s: 14.03 ms, 65.7 tok/s — 1.37×"*.
  It carries its own correction: **the `qkv` row is an artefact** (both `rope()` calls sat inside
  `T_QKV`, charging 1.89 ms of double-precision `pow()` to qkv's weights); qkv is **14.2 GB/s,
  not 4.1**, and the "threads are being woken" inference that followed is withdrawn.
* **`probes/E5_DECOMPOSE_R.md`** and `--attnr {none,sm2,sm3,av2,av3,fork2}` are the instrument
  built to split the attention remainder, and `donor_engine.c:48` still carries
  `T_QKV, T_ROPE, T_ATTN, T_O, T_FFN, T_HEAD, T_NORM` behind `--profile`, with E8's FFN
  sub-timers under it. **Nothing needs to be built.**
* **`probes/E30_IS_THE_ENGINE_AT_THE_WALL.md`** returns `AT-THE-WALL` for `T10-LUTBLK` at
  33.01/36.30 GB/s, corrected to 0.639 of that ceiling for the operative packed path — **on a
  synthetic shape**, and with no trained rate to ask it with.

**What is new, and it is the whole justification:** that organ table reads **19.942 ms/token =
50.1 tok/s**, and E57 measures this artifact at **84.88 tok/s = 11.78 ms**. **The table predates
roughly 1.69× of engine work** — E8, E9, E13, E49, E50, E51, E53 all landed after it — so its
*structure* (which organ, how many calls, how many bytes) survives and **every one of its times
is stale**. The registered 1.37× ceiling was computed from those stale times and cannot be
assumed to still hold in either direction.

## 3. Design

One artifact, `qwen25-05b_tqh.bin`, on `donor_engine_e53.exe`, `--threads 6`, `--bench 160`,
`--profile`. **Timing discipline as standing law**: idle box, foreign occupancy recorded per
cell, **k = 9 reps**, quantiles and interquartile width reported, **no `G-E55a2`** (malformed,
E57 A.5) — dispersion is reported as data with a bootstrap interval.

**Bytes per organ are derived from the artifact's shape and checked against E1's stored code
count**, not read off the old table: with `D = 896`, `L = 24`, `F = 4864`, `V = 151936`,
2 KV heads of 64 —

| organ | weights/token | MB/token at 0.5 B/weight |
|---|---|---|
| qkv | 24,772,608 | 12.4 |
| o_proj | 19,267,584 | 9.6 |
| ffn | 313,786,368 | 156.9 |
| head | 136,134,656 | 68.1 |
| **total** | **493,961,216** | **247.0** |

`493,961,216` is exactly E1's stored `n_codes` for this artifact, so the shape arithmetic is
checked against a file rather than against the table it is replacing.

## 4. Gates, registered before the run

### `G-E58a` — the profile must close, or nothing below counts

> The sum of the per-organ timers plus `T_NORM` and `T_ROPE` must account for the measured wall
> time of a decode step to within **±8%**. If the profile does not close, the timers are
> measuring something other than the token and no organ attribution may be read.

This is the planted control: a decomposition that does not sum to the thing it decomposes is not
a decomposition. **±8%** is registered against the 2.62% interquartile width E57 measured on
this arm plus the known cost of the timers themselves; it is looser than the dispersion on the
axis it watches, per the standing law.

### `G-E58b` — the laggard, named on bytes, not on time

> For each weight-bearing organ report **GB/s = (MB/token) / (ms)**. The laggard is the organ
> with the lowest GB/s. **Report the ceiling if every organ reached the FASTEST organ's rate**,
> which is the same arithmetic §12 used, recomputed on current times.

### `G-E58c` — `Rem`, the non-weight remainder, which is the actual suspect

> `Rem = wall − Σ(weight-organ times)` — attention proper, rope, norms, sampling, the OpenMP
> glue. **Report `Rem` in ms and as a fraction of the token.** At 84.88 tok/s the token is
> 11.78 ms and the weights at the 37.0 GB/s floor would be **6.73 ms**, so **if `Rem` is near
> 5.05 ms the 1.75× is not in the kernels at all** and no kernel work will collect it.

### `G-E58d` — nothing is promoted

> E58 is a **MAP, not a change.** No flag becomes a default, no rate is published, and the
> engine is not modified in this experiment. Anything it suggests is registered as its own
> experiment with its own control.

## 5. Predictions — on the record, before the run

| | prediction |
|---|---|
| `G-E58a` | the profile closes inside ±8% |
| laggard organ | **`qkv`** — 72 calls of 128–896 output rows across 6 threads, the shape §12 flagged before its rope confound was found |
| `head` | still the fastest organ, ≥ 25 GB/s on the current engine |
| `ffn` | 20–30 GB/s; it is 63.5% of the bytes and E8/E9/E13 all targeted it |
| **`Rem`** | **3.0–4.5 ms, i.e. 25–38% of the token** — large, but not the whole 5.05 ms |
| resulting ceiling | if every organ reached the head's rate: **1.25–1.45×**, below D.3's 1.75× |

**I am predicting that the 1.75× is split**, roughly half in a slow organ and half in `Rem`, and
therefore that **neither a kernel nor a glue fix alone collects it.** If `Rem` comes back above
4.5 ms the kernel story is over on this arm and the attention/glue path is the only lever left;
if it comes back under 2 ms, one organ owns the gap and E5's `--attnr` ladder should be pointed
at it.

## 6. What E58 cannot claim

* **Nothing about quality.** This artifact scores 3/160 and 4.531 BPB — above chance. E58 times
  a non-model on purpose, because addendum D.3's transfer argument is about **bytes**, and a
  healed artifact of the same shape has the same bytes. Any rate here is quotable **only** with
  that clause.
* **Nothing about 10B.** One shape, one artifact, one context (`--bench 160`, mean position 80).
  E48 already measured that the context slope lives elsewhere.
* **No comparison to the old table's absolute times.** Different engine, different session. Only
  the *structure* (bytes, call counts) is inherited, and only that is compared.
