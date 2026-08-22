# Controller Audit — Donor Stage-1 Arithmetic Pass

**Role:** Controller (independent adversarial reviewer), §7.1 two-key rule.
**Target:** `docs/research/donor_adaptation/probes/DONOR_STAGE1_ARITHMETIC.md`, `benchmarks/donor_adaptation/donor_inventory.py`,
`benchmarks/donor_adaptation/configs/`.
**Posture:** READ-ONLY on all code. The Builder's tool was RUN, never edited. No commit, no push.
**Date:** 2026-08-20. Branch `research/donor-adaptation`.

---

## VERDICT

> **"Zero donors pass the sealed gates" is NOT a sound conclusion. It is an artefact of the tool —
> specifically of a one-sided bracket collapse in the column the document nominates as "the verdict".**

The tool's arithmetic is *correct*. Its parameter formula reproduces `Qwen2.5-1.5B` to the
**exact byte** (1,543,714,304 = advertised, zero error, re-derived by hand below). Its planted control
is **genuine, not fitted**. Its cross-check containment **holds**. The document is unusually honest
about its own failures.

The defect is not in any number. It is in **which end of each measured bracket was reported as the
result.** The MEASURED-ONLY column — the one §1.3 tells the reader to "read as the verdict" — is
constructed in code (`donor_inventory.py:669`) as

```python
meas = t_proj_pess + t_lut_meas + glue_us + (t_kv_pess or 0.0)
```

i.e. **the pessimistic corner of every bracket simultaneously**: proj at 34.0 GB/s (the low end),
KV at 40.0 GB/s (the low end). There is no MEASURED-ONLY-high anywhere in the document. A bracket was
silently collapsed to a point at its worst edge, and that point was reported as a gate verdict.

**Corrected result: `Qwen/Qwen2.5-1.5B` PASSES the ≥10 tok/s gate at SKU-A / 32K native on
measured-only rates, at every reading of the bracket except its extreme floor.** The honest statement
is `9.77 – 10.65 tok/s`, gate **UNDECIDED**, not FAIL. See F1 — where this is confirmed by
re-running the Builder's own tool with three rate constants moved to the other end of their brackets.

This does **not** rescue the route — see F7 (the sealed 128K contract) and F8 (4-bit KV is unbuilt).
But **"no donor passes, NO-GO" must not go to the owner in its present form.** The defensible headline
is: *one donor straddles the gate; every other donor fails by 2–15×, and those eliminations are sound.*

---

## F1 — BLOCK. The headline rests on the bottom of a bracket the budget document itself opens wider.

**Claimed** (§0): *"Priced on rates that rest on nothing contested, not one of the 18 reachable donors
passes the sealed ≥10 tok/s decode gate."* (§5.1): *"9.76 measured-only … → ~102 ms → 9.76 tok/s."*

**Independent re-derivation** — from `configs/Qwen__Qwen2.5-1.5B.json` only, by hand, no Builder code.
`D=1536, L=28, V=151936, ffn=8960, n_head=12, n_kv=2, head_dim=128, tied`.

| organ | derivation | params |
|---|---|---|
| q,k,v,o proj / layer | 1536² + 256·1536 + 256·1536 + 1536² | 5,505,024 |
| qkv biases / layer | 1536 + 256 + 256 | 2,048 |
| MLP / layer (gated, ×3) | 3 · 8960 · 1536 | 41,287,680 |
| norms / layer | 2 · 1536 | 3,072 |
| **× 28 layers** | | **1,310,339,072** |
| embedding (tied, counted once) | 151936 · 1536 | 233,373,696 |
| final norm | | 1,536 |
| **total** | | **1,543,714,304** |

Advertised safetensors total: **1,543,714,304**. **Delta = 0 parameters, exactly.** The formula is not
approximately right, it is right. PASS on the parameter model.

Per-token bytes, re-derived:
- proj+head fp32 = (5,505,024·28 + 233,373,696) · 4 B = **1,550,057,472 B = 1.4436 GiB** — matches "1.44 GB".
- ternary MLP = 41,287,680·28·0.5 B + (8960+8960+1536)·28·4 B of per-row scales = 578,027,520 + 2,179,072
  = **580,206,592 B = 553.33 MiB** — matches "553.3 MB". No 32-padding owed (all `in_cols` ≡ 0 mod 32).
- KV @32K 4-bit = 2·2·128 · 28 layers · 32768 · 0.5 B = **234,881,024 B = 224 MiB** — matches.
- glue = 7 µs · 28/6 = 32.7 µs — matches.

**Times.** Builder: 1.550e9/34e9 = 45.59 ms; 0.5802e9/11.398e9 = 50.90 ms; 0.2349e9/40e9 = 5.87 ms;
Σ = 102.40 ms → **9.766 tok/s**. **Arithmetically reproduced, term for term.** The number is not wrong.

**The rate choices are.** `PHASE64_BUDGET.md` §1b and §2 both state the proj-GEMV bracket as
**"the fully-streamed floor is [34-40 GB/s]"**. The tool's own source records this and then discards
the top:

```python
PROJ_STREAMED_FLOOR = 34.0   # sec.1b caveat: fully-streamed floor [34-40], pollution-capped
```

The ENV table further reports the asymptote as `[34.0, 36.0]`, narrowing the budget's [34-40] with no
stated justification — while §1's *measured, engine-integrated* proj-GEMV datum is **11 MB / 272 µs ≈
40 GB/s**, described there as *"the fp32 projections already run at aggregate BW when threaded"*.
40 GB/s is the more measured of the two, not the less. The DRAM aggregate ceiling is 40–44 GB/s, so a
1.5 GB sequential fp32 stream running *below* 34 GB/s would need its own justification.

Recomputed — same organ decomposition, same tool, only the bracket read honestly:

| proj GB/s | KV GB/s | proj ms | MLP ms | KV ms | total ms | **tok/s** | gate |
|---|---|---|---|---|---|---|---|
| 34 (Builder) | 40 | 45.59 | 50.90 | 5.87 | 102.40 | **9.77** | FAIL |
| 36 (top of tool's own asymptote) | 40 | 43.06 | 50.90 | 5.87 | 99.86 | **10.01** | **PASS** |
| 36 | 44 | 43.06 | 50.90 | 5.34 | 99.33 | **10.07** | **PASS** |
| 40 (budget §1/§2 top) | 44 | 38.75 | 50.90 | 5.34 | 95.02 | **10.52** | **PASS** |
| 40, **+ F2 scales fix** | 44 | 38.75 | 49.74 | 5.34 | 93.86 | **10.65** | **PASS** |

**At the top of the tool's *own* narrowed bracket the donor already passes.** The FAIL verdict is
produced by, and only by, standing at 34.0 GB/s.

**Corroborated with the Builder's own instrument.** A copy of `donor_inventory.py` was placed in a
scratch directory (the repo copy was never touched), with three constants changed and nothing else:
`PROJ_STREAMED_FLOOR 34.0 -> 40.0`, `RCURVE_ASYMPTOTE 34/36 -> 40/40`,
`DENSE_LUT_GBS 2359296/207e-6 -> 2414592/207e-6` (F2), and the MEASURED-ONLY line's KV term moved from
`t_kv_pess` to `t_kv_opt`. Re-running `analyze` over the same cached configs reproduces the whole table
with **exactly one row changed**:

| donor | Builder meas-only @32K | upper-bracket meas-only @32K | gate @32K |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 9.76 FAIL | **10.65** | **PASS** |
| `allenai/OLMoE-1B-7B-0924` | 5.98 | 6.32 | FAIL |
| `deepseek-ai/DeepSeek-V2-Lite` | 3.41 | 3.54 | FAIL |
| *(all 15 others)* | — | +2–10% | FAIL |

`Qwen2.5-1.5B` @128K goes 8.33 -> 9.10, still FAIL. **One donor, one gate, one bracket. Nothing else
in the document moves.** That is both the size of the defect and the size of the correction.

**Corrected value: `Qwen2.5-1.5B` = 9.77 – 10.65 tok/s at SKU-A/32K on measured-only rates. Gate:
UNDECIDED, straddled.** Everything downstream that says "zero survivors" must be restated.

## F2 — FLAG. The dense-LUT rate is derived codes-only but applied codes-plus-scales. Systematic ~2.3% understatement.

`DENSE_LUT_GBS = 11.398` comes from 2,359,296 B / 207 µs. But `engine.c:227-229` shows the dense LUT
path also carries per-row fp32 scales: `gate_sc[MLP_HID=1024]` + `up_sc[1024]` + `down_sc[D=256]`
= 2,304 rows × 4 B × 6 layers = **55,296 B** additionally streamed per token. True dense rate =
2,414,592 / 207 µs = **11.66 GB/s**. The tool charges *donors* codes **and** scales (F1's 553.3 MB
includes 2.18 MB of scales) while deriving the rate from codes only — an internal inconsistency, in the
conservative direction. Same defect on the routed side: §1's 4.2 GB/s is 2304 KiB / 543 µs codes-only;
the engine streams 2,457,600 B (see F4), giving **4.53 GB/s**.

Effect on the headline: MLP 50.90 → 49.74 ms. At 36/44 GB/s the best donor reaches **10.19 tok/s**;
at 40/44 GB/s, **10.65**.
Independent of F1 and in the same direction.

## F3 — PASS. The Zamba2 formula defect is contained. Route (iv) eliminations stand.

**Claimed:** the +164.11% miss is Zamba2-specific shared-block reuse; that row alone is unusable.

**Verified two independent ways.**

(a) *Key-level.* Only `Zyphra__Zamba2-2.7B.json` carries the block-reuse vocabulary — `num_mem_blocks: 2`,
`adapter_rank: 128`, `use_shared_mlp_adapter: true`, `use_shared_attention_adapter: false`. Grepped
across `granite-4.0-h-small`, `Nemotron-H-8B-Base-8K`, `Falcon-H1-7B-Base`, `Qwen3-Next-80B`: **no such
key in any of them.** granite's `shared_intermediate_size: 1536` and Qwen3-Next's
`shared_expert_intermediate_size: 512` are *shared experts* (an always-active MoE branch), a different
mechanism which the tool counts explicitly and which the Builder reports in its F6.

(b) *Outcome-level.* The safetensors cross-check — an artefact independent of the Builder's formula —
returns **-0.00%** for granite, Nemotron-H and Falcon-H1. Layer-block reuse of the Zamba kind produces
a *large positive* delta (parameters counted L times that are physically stored twice). A silent
2–14× reuse defect cannot hide behind a -0.00% total.

**The cross-check is doing exactly the job §6.3 requires of an instrument: it fired on the one row
where the formula broke and stayed silent where it did not.** Route (iv) is closed on sound arithmetic.

*Residual FLAG:* the cross-check validates **total** parameters, not the **active-per-token** split.
For granite (8.80B active of 32.21B) and Qwen3-Next (3.46B of 79.57B) the active figure is unguarded by
any independent artefact. Both fail by 6–10×, so no verdict turns on it — but the guard should not be
described as covering it. Note also that Zamba2's *time* row is probably closer to valid than its
footprint row (block reuse inflates resident bytes far more than per-token traffic, since every layer
still executes the shared block); it fails at 1.84–2.17 tok/s on either reading.

## F4 — PASS. The planted control is derived, not fitted. The 0.0000% is earned.

**Claimed:** 2,457,600 B/tok = 2400.0 KB, error +0.0000% vs `docs/SIZING.md`; the 2048 B/expert scales
term is what turns 2304 into 2400.

**Re-derived from `engine.c` without reference to the target.** `V=1024, D=256, L=6, E=32, HID_E=128,
KTOP=8` (`engine.c:51-72`). Weight load, `engine.c:218-220`:
`egate_sc[l]=rd(f,GH)` with `GH = E·HID_E = 4096`; `eup_sc[l]=rd(f,GH)`; `eWd_sc[l]=rd(f,E·D=8192)`.
Per expert that is `128 + 128 + 256 = 512` fp32 rows = **2,048 B**, and `engine.c:304/306/309` index all
three per *selected* expert (`egate_sc[l][e*HID_E+i]`, `eWd_sc[l][e*D+d]`), so they are streamed per
token, not resident-shared. Codes: `3·D·HID_E·0.5 = 49,152 B`. Expert = **51,200 B**.
× top-8 × 6 layers = **2,457,600 B = 2400 KiB.**

`SIZING.md:21-22` states the target as *"expert = 98,304 ternary weights **+ per-row scales** ≈ 48 KB
… = 2400 KB/token (counted in-engine)"*. The scales term is **named in the target's own sentence** and
its size is **forced by three lines of `engine.c`**. It was not reverse-engineered to close a gap.
The 0.0000% is the right kind of zero.

*Small FLAG, against SIZING.md rather than the Builder:* that document is internally inconsistent —
"≈48 KB" × 8 × 6 = 2304, not 2400. The Builder resolved the ambiguity correctly (2400 is the
"counted in-engine" authority; 48 KB is the codes-only rounding). Worth fixing at source. Note that
`CONTROLLER2_REPLICATION.md` reads the same expert as codes-only 49,152 B — the two Controllers'
unit conventions differ by exactly the scales term, and F2 is the downstream consequence.

**C2 and C3 were RUN, not taken on trust** (`python donor_inventory.py control`, this session):
- **C2:** 2,457,600 → 2,150,400 B, −307,200 B, **−12.500%**, exactly 7/8. Right direction, right
  magnitude, no over- or under-shoot. Fires as claimed.
- **C3:** deleting each of `num_hidden_layers` / `hidden_size` / `vocab_size` raised
  `MissingConfigField` naming the key *and* the figure that needed it; the unperturbed config raised
  nothing. **Guard exercised in both directions**, as project law §4 requires.

**All three controls PASS exactly as reported.**

## F5 — FLAG. Footprint DOES include KV, but at 32K, not at the sealed SKU-A contract.

**Claimed** (§0.1): *"14 of 18 donors fit inside SKU-A's 16 GB with KV and a declared 1 GB margin."*

KV **is** in the footprint, at the stated context — verified: the Qwen2.5 SKU-A cell = 741.8 MB weights
+ 224 MB KV@32K + 1 GiB margin = 1.943 GB ✓, matching T3. The omission the mandate feared did not occur,
and S2's inventory terms (packed weights, per-row scales, 32-padding, non-ternary organs,
embeddings/head, KV, logits, declared margin) are all present in the tool's formula.

But S3 seals SKU-A at a **128K user-visible contract**. Re-reading T3's own 128K column against 16 GiB:
OLMo-2 (20.41), Qwen3-30B (18.31), Mixtral (26.79), granite (16.58), Qwen3-Next (39.10) go over →
**13/18, not 14/18** — and that is still with 4-bit KV assumed. At fp16 KV @128K, SmolLM2's cache alone
is 24.00 GB, larger than the whole SKU-A budget. **"Footprint is not where the trouble is" is true at
32K and 4-bit, and is not established at the sealed contract.** The claim must carry its conditions.

## F6 — PASS (both legs of the Principal-facing claim), with one correction to the reasoning.

*Dispatch term.* `Qwen3-30B-A3B`: 384 calls × 8.4 µs = 3.23 ms against a 362 ms total = **0.89%**;
`Qwen3-Next`: 528 × 8.4 = 4.44 ms of 415 ms = **1.07%**. Both ≤ 1.1%, and these are the largest call
counts in the set. **Confirmed:** no donor's verdict moves for any value of this constant in [0, 8.4] µs.

*ρ-safe granularity.* T8 recomputed independently, codes + per-row scales:
- OLMoE: `3·2048·1024·0.5 + (1024+1024+2048)·4` = 3,145,728 + 16,384 = **3,088.0 KB** ✓
- Mixtral: `3·4096·14336·0.5 + (14336·2+4096)·4` = 88,080,384 + 131,072 = **86,144 KB** ✓
- Qwen3-Next: `3·2048·512·0.5 + (512·2+2048)·4` = 1,572,864 + 12,288 = **1,548.0 KB** ✓

Minimum over all seven MoE donors = 1,548 KB = **32× the 48 KB threshold**; maximum 1,795×.
**Confirmed — the measurement the Principal has requested from the owner can be retired.**

*Correction to the stated mechanism:* the safety does not follow from 48 KB being "ρ-safe". Per
`CONTROLLER2_REPLICATION.md` §Q3.2 the LUT path is **compute/kernel-bound by ~16×**, not
bandwidth-bound, so the ρ-law is not the operative constraint on this path at all. Right conclusion,
wrong reason. Under the project's verify-before-publishing rule the mechanism must be restated
correctly before this reaches the owner, since the owner cannot audit it.

## F7 — BLOCK (presentation). No donor passes at the sealed 128K SKU-A contract, and the tool never checks native context at all.

Every reported pass is at **32K native attention**, never 128K. Under S3 a 32K native window *is* the
permitted SKU-A configuration — but only *"plus the recall tier … it must pass the long-context
retention gate in S4/§8.I at 128K"*, which is unmeasured and entirely unpriced here (the recall query
itself is cheap, ~29 µs at 128K entries, so the cost is quality, not time). §5.1's heading
*"SKU-A (16 GB, 32K native attention window, 4-bit KV)"* is honest; §0's *"at any SKU, at any context"*
and T6's bare `gate @32K` column are not, and the elimination ledger's `SKU-A @32K best-case tok/s` is
the number a reader will quote out of context.

**At SKU-B (128K native, sealed, no recall path permitted): no donor passes on any column.**
`Qwen2.5-1.5B` = 8.33 measured-only / 10.14 opt; corrected for F1+F2 at 40 + 44 GB/s it reaches
**9.01 tok/s — still FAIL.** That half of the headline survives audit intact and is safe to report.

**Additional defect, unflagged by the Builder:** the tool never reads `max_position_embeddings`.
`allenai/OLMoE-1B-7B-0924` is a **4096**-position model and is scored at 32K and 128K;
`Qwen/Qwen3-1.7B` is 40960 and scored at 128K; `nvidia/Nemotron-H-8B-Base-8K` is an 8K model scored at
both. These rows price contexts the donors cannot natively serve. The direction is *favourable* to
elimination (they would need RoPE extension, which costs quality under S4), so no verdict flips — but
the **#2-ranked shortlist entry `OLMoE` is a 4K model presented as a 32K SKU-A candidate**, and that
must be labelled before the owner reads the shortlist.

## F8 — FLAG. The headline stands at the pessimistic corner on rates and the optimistic corner on everything else.

Three optimistic assumptions sit inside the "conservative" number and are never netted against F1:

1. **4-bit KV is assumed throughout and does not exist in this engine.** At fp16 — the only KV precision
   built — `Qwen2.5-1.5B` @32K reads 896 MB/token → 22.4 ms → **8.4 tok/s**. The dequantisation compute
   for 470M elements per token is unpriced in both cases.
2. **Organ times are summed with zero overlap** — no prefetch, no compute/stream overlap, no
   interleaving of KV read with weight stream (the Builder declares this, §7.3).
3. **The per-position compute floor is a linearly-scaled 7 µs glue term** for models 6× deeper and 36×
   wider than the anchor whose measured floor was 1.14 ms/token (declared, §7.4).

These do not restore the FAIL verdict — (1) is the largest and it lands on a term worth 5.9 ms against
the 16.8 ms swing from F1+F2 — but a document must not present one corner as conservatism while
standing on the other. **Recommendation: report every donor as a bracket with both ends, or give the
tool a `--measured-only-high` column. The single-point MEASURED-ONLY column should be withdrawn.**

## F9 — PASS. Everything else audited reproduces.

Parameter formula exact to the byte on `Qwen2.5-1.5B` (F1). Cross-check disagreements are reported
loudly and diagnosed correctly (`Qwen3-1.7B` tied-duplication = exactly one `V·D` = 311.2M ✓;
`gpt-oss-20b` packed mxfp4 storage, `U8` blocks counted as parameters ✓). Provenance rule honoured —
every figure traced to a named config key, `config.json` sha256 + resolved revision recorded per donor
(T9), refusal by named exception rather than silent defaulting. The `1 GiB` footprint / decimal
`1e9 B/s` rate convention is declared and applied consistently. §9's list of what the pass does *not*
establish is accurate and unusually complete. **F1 is a reporting defect in an otherwise well-built
instrument, not a broken instrument.**

---

## What the owner should be told

1. **Not "zero donors pass".** One donor — `Qwen/Qwen2.5-1.5B` — **straddles the ≥10 tok/s gate at
   SKU-A / 32K native on measured-only rates: 9.77 – 10.65 tok/s**, confirmed by re-running the
   Builder's own tool at the other end of its own brackets. The gate sits inside the bracket.
   That is UNDECIDED, and it is decidable cheaply.
2. **The other 17 eliminations are sound** and survive every correction in this audit. The second-best
   donor reaches 5.98–6.32 even at the corrected upper bracket; the gap to the gate is 40%+, far outside any bracket. **Route (iv) / the
   hybrids is genuinely closed** (F3). The negative result at stage 1 is real for 17 of 18 donors.
3. **No donor passes SKU-B at the sealed 128K native contract** — `Qwen2.5-1.5B` reaches 9.01 tok/s
   even corrected. That conclusion is safe to act on.
4. **This is not a NO-GO. It is a one-donor decision that turns on one unmeasured constant.**
   The entire verdict hangs on the fp32 streamed-projection rate somewhere in [34, 40] GB/s. That is a
   **CPU-only, zero-GPU, sub-hour microbench** of the existing proj-GEMV kernel at donor projection
   sizes (~1.5 GB/token). **Measure it before anyone amends a sealed gate or declares a NO-GO.** The
   Builder's §7 assumption 1 asks for the adjacent measurement (ternary projections); this one is
   cheaper and decides more.

## Not audited (stated rather than guessed)

- `inventory.json`'s per-figure provenance map was spot-checked, not walked donor by donor.
- The 32-padding term and the `architecture metadata` line of S2's inventory formula were verified for
  `Qwen2.5-1.5B` only.
- The MLA KV formula for `DeepSeek-V2-Lite` was not re-derived from first principles (that row fails
  by 1.3–2.9× and no correction of plausible size reaches the gate).
- No attempt was made to reach the two gated repos, per the read-only posture.
