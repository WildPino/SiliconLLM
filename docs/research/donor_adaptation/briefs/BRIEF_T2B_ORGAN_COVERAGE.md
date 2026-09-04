# BRIEF T2b — Does the winning rule survive outside the FFN?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-04.**
**Depends on: `probes/T2_TERNARIZATION_RULE.md` (in writing), `briefs/BRIEF_T2_TERNARIZATION_RULE.md`.**

---

## 1. The gap this closes

T2 measured every arm on the **FFN organs only** — 84 tensors, 28 layers × `{gate, up, down}`.
Its own §4 forbids extrapolating that result:

> *"No extrapolation. Whatever wins on the FFN may not be assumed to transfer to attention or to
> the head without measuring those separately."*

But **a model that runs needs every organ converted.** `qwen_export.py` applies the chosen rule to
`q/k/v/o` and, with `--head-ternary`, to `lm_head` as well. So the file the runtime loads is
quantized over organs whose cost has never been measured, and the sidecar now records that
difference explicitly (`rule_applied_to` vs `t2_measured_organs`) rather than leaving it implicit.

**Until T2b runs, the BPB of the runnable model is unknown.** T2's number is a lower bound on the
damage, not an estimate of it.

## 2. Why this is not a formality

Two reasons to expect attention and the head to behave differently, both already on this
programme's record:

- **`PHASE61_PROJQUANT` measured SSM projections as precision-hungry**: ternarizing them cost
  +0.018–0.022 BPB where the FFN tolerated far more, and they were left fp32 as a result. The
  attention projections are the closest analogue a transformer has.
- **The head is 15.1% of this donor's non-embedding weights** (27.6% on the 0.5B the runtime
  actually executes, where `D` is smaller against the same 151,936-token vocabulary) and up to
  97% of a 10B speed budget
  (`donor_speed_budget.py`). If the head must stay fp32 for quality, the speed ledger's headline
  changes — R1 §4.1 bought 23.5 → 38.0 tok/s precisely by ternarizing it.

## 3. Arms — same rule, widening organ sets

Fixed: donor Qwen2.5-1.5B rev `8faed761…`, shared eval slice (`heldout` 24×512 seed 1234,
`ids_sha256 = a1a48dc9…`), calibration 32×512 seed 42424 from the disjoint half, paired
sequence-bootstrap SEs, **rule = T2's winner**.

| arm | organs converted | purpose |
|---|---|---|
| **base** | none | replication gate: must reproduce `0.767594958` |
| **F** | `gate, up, down` | **replication gate: must reproduce T2's winning Δ to <0.01** |
| **FA** | F + `q, k, v, o` | the attention increment |
| **FAH** | FA + `lm_head` | the increment that makes the model runnable as exported |
| **A-only** | `q, k, v, o` | isolates attention from the FFN's damage |
| **H-only** | `lm_head` | isolates the head; the organ the speed ledger most wants ternary |
| **I** | identity substitution over the FAH code path | **instrument control — must reproduce base bit-exactly** |

Arm **F** is the load-bearing control: it re-derives T2's number through *this* script. If F does
not replicate, nothing else in the run is read.

## 4. Pre-registered decision rule — fixed before any result

Let `Δ(arm)` = BPB(arm) − BPB(base). Let `w` = T2's winning FFN Δ, re-measured as `Δ(F)`.

| outcome | condition | what it means |
|---|---|---|
| **UNIFORM** | `Δ(FAH) ≤ 1.60 × Δ(F)` | the rule transfers; the runnable model costs what the FFN measurement said, and T2's number may be quoted for the whole model |
| **ATTENTION-BOUND** | `Δ(A-only) > Δ(F)` | the attention projections are the precision-hungry organ, as P61 found for the SSM. They come out of the ternary set and the speed ledger is re-priced without them |
| **HEAD-BOUND** | `Δ(H-only) > 0.5 × Δ(F)` | the head cannot be ternarized at this rule. R1 §4.1's 23.5 → 38.0 tok/s is withdrawn and the head needs its own treatment |
| **DIFFUSE** | none of the above, and `Δ(FAH) > 1.60 × Δ(F)` | the damage is spread; no single organ is the culprit and the next lever is width or healing, not organ selection |
| **VOID** | arm I does not reproduce base exactly, or arm F misses T2's Δ by >0.01 | instrument broken; report nothing else |

These thresholds are fixed here, before the run, and the bar is derived rather than chosen.
Counted from the config actually on disk (`Qwen2.5-1.5B`, `D=1536 F=8960 L=28 V=151936`,
`heads=12/2 hd=128`):

| organ set | weights | share of non-embedding |
|---|---|---|
| attention `q,k,v,o` | 154.1 M | 10.0% |
| FFN `gate,up,down` | 1156.1 M | 74.9% |
| head | 233.4 M | 15.1% |

**A rule that damaged every weight equally would therefore put `Δ(FAH)/Δ(F)` at 1543.6/1156.1 =
1.335.** An earlier draft of this brief set the bar at 1.3× from a remembered 36%/64% split; that
split was wrong, and 1.3× would have failed a perfectly uniform rule. The bar is **1.60×**:
comfortably above the uniform expectation, and below the ~1.9× that a rule twice as damaging
outside the FFN would produce.

## 5. What this brief does NOT test

- It does **not** test another rule. T2 chose the rule; this asks only where it applies.
- It does **not** test activation quantization. That is the `--lut` path's cost, measured
  separately in `donor_engine.c` and **not** composed with these numbers.
- It does **not** sweep the calibration budget. `D4b` is still unrun and every R3-family number in
  this programme remains a floor for that reason.
- It does **not** measure through the runtime. That is the separate `--bpb` job, and until it runs
  these are PyTorch numbers about a model the runtime executes.

## 6. Cost

One BPB pass per arm (~140 s at 6 threads on 1.5B, measured in T1/T2) plus one calibration pass
shared across arms. **CPU only, no gradients, well under an hour.**

## 7. Reporting

`probes/T2B_ORGAN_COVERAGE.md`. Report Δ and paired SE for every arm, the arm-I and arm-F
controls, the §4 label verbatim, and — if ATTENTION-BOUND or HEAD-BOUND — the explicit re-pricing
of `SPEED_LEDGER.md` §10 that follows from taking those organs out of the ternary set.
