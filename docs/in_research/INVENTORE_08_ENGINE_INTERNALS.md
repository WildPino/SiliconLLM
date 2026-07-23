# INVENTORE_08 — Reading the engine: creative levers found in the actual code

**Status: Inventor pass 3, code-grounded (2026-07-19). Read in full: `benchmarks/phase60/engine.c` (957 lines,
the consolidated product) + `benchmarks/phase55/phase55_ssm.py` (the model ArchA). No engine code touched.**

Pass 1 read the organs (docs). Pass 2 read the runtime (docs, one probe). **Pass 3 read the C.** The rule
from the owner stands: the product is the engine's tok/s on a commodity CPU; weights are a fixed input;
training is out. Everything below is anchored to a line in the real source, and every lever is either
already measured, or falsifiable at $0 / by a bounded engine change — flagged per item. Where a move needs
editing `engine.c` (shared code), it is routed to the Architect, not done here.

---

## 1. The per-token control flow, as actually written (`forward_token`, engine.c:324)

```
emb gather (327, 1 row = 1 KB)
for each of 6 layers:
  rmsnorm (330)                                              ── norms bucket
  SSM layer (345–372):
    in_proj  matvec  (348)  fp32  2·DN·D = 1 MB  ┐
    conv4 + silu (351–352)  silu = scalar libm    │
    x_proj   matvec  (353)  fp32  (DTR+2N)·DN     │ all in the
    dt_proj  + softplus (355) softplus scalar libm│ "proj-GEMV" /
    selective scan (359–368) exp256_ps (fast) ────┤ scan_other
    z-gate  y*=silu(z) (370)  silu scalar libm     │ bucket = 52.7%
    out_proj matvec  (371)  fp32  D·DN = 512 KB   ┘
  rmsnorm (375)                                              ── norms bucket
  MLP: mlp_moe (288) or mlp_dense (254)  ternary LUT         ── LUT-MLP 23%
head matvec (384)  fp32  V·D = 1 MB (grows with vocab)       ── head 3% (at V=1024)
```

**Measured shares (E3.5 full engine):** proj-GEMV/scan_other **52.7%** · LUT-MLP 23% · scan-exp 13.7% ·
SWA 7% · head 3%. Two facts jump out of the *code* that the profile label hides:

1. The "proj-GEMV 52.7%" bucket is **not only GEMVs** — it also contains **7,680 scalar `libm` transcendental
   calls per token** (conv-`silu` 512 + `softplus` 512 + z-`silu` 512, ×5 SSM layers; engine.c:352/355/370).
   E3.5's fast-exp replaced *only the scan's inner `exp`* (line 361→365); the glue `silu`/`softplus` were left
   on scalar `libm`. **This is unattacked, and it is visible only in the source.**
2. The projection **weights are fp32** (`rd()`, engine.c:205–208, 214), streamed through `matvec`→`dotf`
   (engine.c:100/92) — a plain fp32 GEMV. The int8 machinery the MLP uses (`quant_i8`, AQ=63, engine.c:167)
   is **right there**, unused by the projections.

---

## 2. The findings (each anchored, ranked in §3)

### F1 — int8 / dequant-on-read projection weights ★ (the code is 90% built)

**Where:** `matvec` (engine.c:100) does `y[o]=dotf(W+o*in, x, in)` over **fp32** `W`. The seven fp32
projection tensors — `in_proj, x_proj, dt_proj, out_proj` (engine.c:205–208), `swa.qkv, swa.o` (203),
`head` (214) — are the whole 52.7% bucket. The comment at engine.c:95–99 records that a *blocked-GEMV*
(compute-lever) was reverted because the loop is **memory-bandwidth-bound** — which is exactly why the
**byte-lever** is the right one: on a weight-bandwidth-bound GEMV, time ∝ stored bytes.

**Move:** store these tensors int8 with a per-row scale (the MLP is already exported packed; the exporter
learns one more path), and either (a) **dequant-on-read** (load int8 row → cast fp32 → existing `dotf`;
streamed bytes ÷4, accumulate stays fp32 = numerically gentle) or (b) a true int8×int8 `matvec_i8`
(`_mm256_maddubs_epi16`+`madd`, VNNI-free, ~2× compute *and* ÷4 bytes). **(a) is the clean first cut** —
minimal numeric change, 4× fewer streamed/resident bytes on the dominant component.

**Quality: already proven free.** `t1_proj_precision.py` (INVENTORE_07, this branch): int8-per-row on all
four SSM projections = **−0.0000 BPB, top-1 99.79%** at 100K/seq512; fp16 = −0.0000/99.99%. The cliff is
below int8 (ternary, P61, was +0.018). **The gate that matters — quality — is passed.**

**Leverage:** ÷4 bytes on 52.7% → memory-bound ⇒ order **~1.6× end-to-end** (Amdahl), *and* the resident
footprint of the fp32 organs (the heavy tenant of the ≤16 MB keystone) drops 4× → more model stays
compute-bound at scale-up. **Status:** quality done; the engine kernel + exporter path is an Architect
change (touches `engine.c`/`e1_export.py`). This is the headline of the whole exploration.

### F2 — the head is fp32 and **grows with vocab** — F1's value scales with it

**Where:** `head=rd(f, V*D)` (engine.c:214), `matvec(head, xn, logits, V, D)` every token (engine.c:384).
At V=1024 the head is 1 MB streamed/token (= `in_proj`'s size); the blueprint (§3.7) already flags "at
target vocab the head can eat the L3 budget." **Code fact:** `emb` (engine.c:192) and `head` (214) are
loaded as **separate** tensors — the model does **not** tie them (`phase55_ssm.py:116` `emb` vs `:118`
`head`, distinct Parameters). So the resident cost is **2× V·D fp32**, and the head is streamed in full
every step. **Move:** int8 head (F1 applied to the output projection) is the *direct* answer to §3.7's
"head eats L3" — no adaptive-softmax / VQ / factorization needed. At V=4096 it turns a 4 MB/token fp32
stream into 1 MB. Tying emb↔head would also halve the resident copy but needs a retrain (out of scope) —
**int8 is the inference-only lever.** **Status:** rides on F1; its payoff *increases* with the product's
real vocab. Quality re-check: extend the t1 probe to the head tensor ($0, minutes).

### F3 — vectorize the `silu`/`softplus` glue with the exp the engine already has ★ (unattacked, code-only)

**Where:** `silu` (engine.c:101) = `x/(1+expf(-x))`, `softplus` (engine.c:102) = `log1pf(expf(x))` — both
**scalar `libm`**. Called in DN=512 loops: conv-`silu` (engine.c:352), `softplus` on dt (355), z-gate
`silu` (370) — **7,680 scalar transcendental evaluations/token**, sitting inside the 52.7% bucket. E3.5
built `exp256_ps` (engine.c:78) — a deterministic 8-wide poly-exp, parity-proven ≤2e-6 (engine.c:887) —
and used it **only for the scan's inner exp**. The glue was never converted.

**Move:** `silu8`/`softplus8` from `exp256_ps` (silu(x)=x·σ(x) needs one exp; softplus(x)=x for x>20 else
log1p(exp x) — a poly-log1p over the observed domain, range-characterized like E3.5). Vectorize the three
DN-loops. **Same lawful discipline as E3.5** (versioned deterministic approximation, pre-registered parity
gate: BPB Δ ≤ +0.001, top-1 ≥ 99.9%). **Leverage:** bounded by the scalar-transcendental share of the
52.7% bucket — desk bound = 7,680 `libm` calls/token removed; the honest number is the in-engine
before/after (Architect). This is the cleanest *new* speed lever in the code: it reuses an
already-shipped, already-parity-proven kernel on hot loops that were simply missed. **Status:** needs an
`engine.c` change → Architect; the falsification protocol is E3.5's, verbatim.

### F4 — the MoE dispatch overhead has a specific, code-visible shape

**Where:** `mlp_moe` (engine.c:288). The gate/up LUT is built **once** from `xn` (engine.c:296), but the
**down** projection rebuilds a LUT **per selected expert** — `quant_i8(he,…); build_lut_t3(…)` inside the
`KTOP=8` loop (engine.c:307), because `he` differs per expert. That per-expert quant+build + three small
windowed kernel calls (engine.c:303/305/308) is the ~8.4 µs/expert dispatch overhead measured in 64.0
(the MoE-MLP is *gather/kernel-bound*, thread-flat ×1.06, not bandwidth-bound). **Moves:** (i) the queued
fusion — gather the 8 selected experts into one contiguous padded kernel (one dispatch); (ii) a
code-specific micro-fix — `egate_cd`/`eup_cd` are separate blocks read with the *same* LUT, so
concatenating gate+up rows per expert makes one kernel call do both `2·HID_E` rows (halves the gate/up
call count). **Status:** (i) already in the engine-v2 queue; (ii) is a small new observation. Both are
Architect engine work; noted for the map, not claimed.

### F5 — the bottleneck **shifts** after F1: the lever sequence

This is the insight only the code + profile together give. Today: proj-GEMV 52.7% (fp32). Apply F1 (÷4 the
proj bytes, memory-bound) and that bucket drops toward ~15–20%; the new top becomes the **LUT-MLP (23%)**
and the **scan (13.7%+)**. So the *ordered* engine roadmap is: **F1 (proj int8) → F3 (silu/softplus glue,
which lives in the same bucket and is freed alongside) → then F4 (MoE dispatch) / scan**. Sequencing
matters because each lever's Amdahl weight is set by what the previous one left standing — a map you cannot
draw from the docs' single 52.7% number.

### F6 — recorded absences (the teardown must also note what's correctly missing / out of reach here)

- The **recall tier is not in `engine.c`** at all — it is a standalone artifact (64.3, 29 µs/token). Its
  integration into the main loop (and the F1 treatment of its IVF/ADC codes) is a separate, real
  opportunity, but it needs the recall C-side merged first — Architect track.
- `emb`/`head` untying → retrain → out of scope (noted under F2 as inference-int8 instead).
- The scan's `exp256_ps` (engine.c:365) and the LUT kernels (111–157) are already near-optimal; no lever.

---

## 3. Ranked (engine, fixed weights)

| # | finding | where | leverage | status |
|---|---------|-------|----------|--------|
| **1** | **F1 int8/dequant-on-read projections** | matvec 100; loads 205–214 | ÷4 bytes on 52.7%; ~1.6× + keystone | **quality PROVEN free** (t1); kernel = Architect |
| **2** | **F3 vectorize silu/softplus glue** | 101/102, 352/355/370 | frees 7,680 libm calls/tok in the 52.7% bucket | new; reuses `exp256_ps`; parity-gated (E3.5 protocol) |
| 3 | F2 int8 head (scales with vocab) | 214/384 | 4 MB→1 MB/tok at V=4096 | rides F1; re-probe head, $0 |
| 4 | F4 MoE dispatch shape (fuse gate+up / gather) | mlp_moe 288–309 | attacks the ~8.4 µs/expert | (i) queued; (ii) new micro-fix |
| 5 | F5 lever-sequence after F1 | — | roadmap correctness | analysis |

## 4. Scope honesty

- **What this pass produced:** a proven-free quality result for F1 (int8 projections) and a *new*,
  code-only speed lever (F3, the un-vectorized glue) — plus the map of how the levers sequence.
- **What it does NOT claim:** in-engine tok/s. F1/F2/F3/F4 all require editing `engine.c` (shared code) to
  turn the byte/call reductions into measured wall-clock — that is the Architect's hand, with the E2/E3.5
  parity gates. The side-lab's job ends at "quality is safe + here is exactly where and why."
- **The one thing an owner could act on immediately:** F1 + F3 are the same bucket (52.7%), the same
  parity discipline, and the quality half of F1 is already banked. They are the natural next engine stage.
