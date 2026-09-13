# BRIEF E60 — the rung that was never built: one byte per weight

**Status: PRE-REGISTERED.** Written and pushed before the runner exists, before `donor_engine.c`
is touched, and before any artifact at this format has been exported. Nothing below is a result.

---

## 1. The question

**On this engine, with these trained donors, what does a weight cost at ONE byte — in speed, and
in fidelity?**

The engine's precision ladder has exactly two rungs:

| `quant` | format | bytes / weight | who uses it |
|---|---|---|---|
| `0` | fp32 | **4.0** | `qwen25-*_f32.bin` — the only faithful artifacts this programme has |
| `2` | packed base-3, 2 trits/byte | **0.5** | `qwen25-*_tq.bin`, `*_tqh.bin` — every fast artifact |

**There is nothing in between, and the jump is 8×.** E57 measured what that costs: the faithful
arm `05b_f32` runs at **19.09 tok/s** (38% of the bar) and scores **160/160**; the fast arm
`05b_tqh` runs at **83.41** and scores **3/160**, which E18 part A established is *below* the
floor of a model that emits `'\n'` and nothing else (11/160 at 0.5 B). There is no arm that is
both.

E58 then closed the engine-side question on the faithful arms: `05b_f32` moves **38.5 GB/s** and
`15b_f32` **38.7 GB/s** (ledger §57, `05b`/`15b` rows), against E30's measured ceiling of 36.30
and §1's DRAM aggregate of 40–44. **The fp32 arms are at the memory wall.** E58 §3 measured the
token to be **96.3% weight-organ time at 0.5 B and 98.0% at 1.5 B** (`Rem` = 0.427 ms and
0.668 ms). So on a faithful arm there is no kernel lever left: *the only lever is bytes per
weight*, and the ladder skips the whole interesting middle.

**This brief asks for the middle rung.** One byte per weight is a 4× cut against fp32 —
arithmetically the difference between 19.09 tok/s and something in the fifties or seventies — and
it is the precision at which post-hoc quantization is ordinarily unremarkable, unlike ternary,
which this programme has now measured as fatal at every scale and every organ (E12
`CHANCE-LINE`, E18 `CLIFF-NOT-SLOPE`).

## 2. Why this is not already answered — checked, not assumed

Searched `docs/research/donor_adaptation/**` and `*/results/` for `int8`, `INT8`, `Q8_0`,
`4-bit`, `nf4`, `unpacked`, and `B/weight`. **Every brief the search returned is named below with
what it SAID**, per the rule registered in `feedback_search_before_claiming_a_gap` and re-stated
as E57 B.3 / C.7.

| artefact | what it actually said | does it cover this question? |
|---|---|---|
| `project_cpu_bandwidth_research` (memory, 2026-06-25) | *"int8 su Zen2 dava 1.19× perché manca VNNI → la risposta non era int8, è ternario-LUT"* | **No.** The rejection is on the **compute** axis (`vpdpbusd` absent on Zen 2). It says nothing about bytes moved. |
| `project_probe1_ternary` (memory, 2026-06-29) | *"meno bit → più veloce … LA firma che lo distingue dal dequant-int8 che si pianta <4-bit"*, and, in its own words, ***"Scope onesto: valida il MECCANISMO+kernel sul nostro Zen2, NON il win di banda (microbench L1/L2-resident, lo streaming è scale-up)"*** | **No — and it disqualifies itself.** Probe-1's anti-int8 finding is an L1/L2-resident compute microbench. Phase 61's registered law is *"microbench compute-bound NON compone a engine memory-bound"*. |
| `E14_ACTIVATION_INT8_COST`, `E32_THE_OWED_PROTOCOL`, `E11`, `E13` | int8 **activations** (`quant_i8`, `AQ=63`, per-group amax). E32 re-ran E14's protocol and the sign flipped to `+0.048990745` at 1.5 B | **No.** Different tensor. Weights are untouched in all four. |
| `BRIEF_E4_ATTENTION_ACCUMULATORS`, `CONTROLLER_*_AUDIT` | int8 / 4-bit **KV cache** (unbuilt), and 4-bit as a *donor baseline precision* for fitting a model on a T4 | **No.** Neither is a weight format in this engine. |
| `BRIEF_P1_NIBBLE_PACKING`, `LUT_PACKING_PRIOR_ART`, `CONTROLLER_D5_RESULTS_AUDIT` | the engine stores a 9-state **trit pair** in a full `int8_t` byte because `vpshufb` needs a 4-bit index; bitnet.cpp TL1/TL2 do the same | **No.** That is 0.5 B/weight wearing an `int8_t` type name. |
| `E18_THE_RANKING_LADDER` §5 (part C) | the ladder's speed axis, computed at **4 B and 0.5 B only**: *"an fp32 weight costs **5.87×** a ternary one"*, `base` 1.23 → `FAH` 7.21 tok/s | **No — this is the gap, in the record's own table.** Every rung is a *mix* of the two existing formats. No rung changes the bytes of a weight itself. |
| `E30`, `E34`, `E35`, `E36`, `E58`, ledger §§23/56/57 | every rate is in **charged weights at the packed convention, 0.500000 B/weight** | **No.** The whole rate ledger is written at one byte-per-weight value. |
| `grep unpacked` over all of `docs/` | two hits, both prose about the *packed* layout | **The `quant=1` path has never been timed.** |
| all six `D:/_ktmp/e1/*.bin` headers, read this session | `f32` → `quant=0`; `tq`, `tqh` → **`quant=2`** at both scales | **No `quant=1` artifact exists for any donor.** |

**Conclusion of the search: the 1 B/weight rung has never been built, never been timed, and never
been scored, on this engine or on these donors. The one place the record rejects int8 rejects it
on an axis that is not the binding one, and says so itself.**

## 3. Why it might work with no kernel written at all

`donor_engine.c:611-634`, the final fallback of `matvec_sel` — the `quant==1` branch:

```c
for(int o=0;o<n_out;o++){
    const int8_t* c=m->code+(size_t)o*n_in;
    __m256 acc=_mm256_setzero_ps(); int i=0;
    for(;i+8<=n_in;i+=8){
        __m128i c8=_mm_loadl_epi64((const __m128i*)(c+i));
        __m256i ci=_mm256_cvtepi8_epi32(c8);
        acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(ci),_mm256_loadu_ps(x+i),acc);
    }
    ...
    t*=m->scale[o];
    y[o]=bias?t+bias[o]:t;
}
```

**This is a general `int8 × fp32` dot product with a per-row fp32 scale.** It is "ternary" only
because of the values that happen to be in the file — `_mm256_cvtepi8_epi32` widens the full
signed byte range, and nothing downstream of it assumes `|c| ≤ 1`. `read_mat` (`:903`) reads
`out*in` bytes plus `out` floats; `fuse_mats` (`:1039`) already computes the row stride as
`in` rather than `in/2` when the matrix is not packed.

**So an int8-weight artifact is an exporter question, not a kernel question.** The exporter
already has the writer: `qwen_export.py --quant ternary` (`w_tern`) emits exactly `int8 [out,in]`
codes plus `fp32 [out]` scales. Only the *quantizer* that fills it is ternary; swapping in
round-to-nearest at `amax/127` changes the values and nothing else.

Two consequences worth stating before the measurement, because they are the reason this is cheap:

1. **Speed is a function of bytes, not of values** on this branch — the same instruction stream
   runs whatever is in the bytes. That is testable (`G-E60c`) and is the control that separates
   part A from part B.
2. **`--lut` / `--lutblk` are structurally excluded.** `donor_engine.c:1553` dies with *"`--lut`
   requires a `--quant packed` model"*. E59's ×1.253/×1.333 kernel cannot be applied to this rung,
   and this brief does not claim it. The scope of every number here is **the engine's
   convert-based `quant==1` kernel** — not "the engine".

## 4. The one C change, and why it is not optional

An int8-valued file and a ternary-valued file are **both `quant=1`**, and the engine prints
`quant=ternary` for both (`:957`, `:1581`). Two arms that differ in the thing being measured would
print the same `CONFIG` line. That is exactly the defect
`feedback_config_must_appear_in_output` was registered for — *the attention kernel sat on the slow
default for twenty-two experiments because the `BENCH` line did not print the arm*.

So E60 adds **`quant=5` = "int8, 1 B/weight"**, which reads byte-for-byte like `quant=1` and
differs only in what the engine calls it:

1. `read_mat`: `else if(quant==1||quant==5)` — same payload.
2. `const int tagq = (M->quant==4) ? 3 : M->quant;` — unchanged; 5 is not tagged.
3. the stderr shape footer (`:953-957`) and the stdout `CONFIG` line (`:1577-1581`) gain an
   `int8(1 B/weight)` / `int8` case.

Nothing else reads `M->quant`. `m->packed=(quant==2)` stays false; `--lut`'s guard at `:1553`
still refuses. **A new binary `donor_engine_e60.exe` is built and EVERY arm of E60 — fp32, packed
and int8 — runs on it**, so no comparison crosses binaries.

## 5. Arms

Artifacts to export (all with `--head-ternary`'s untied layout, i.e. the `*h` family E57/E58/E59
measured, so the comparison is against arms that exist):

| id | file | quant | rule | bytes/weight | purpose |
|---|---|---|---|---|---|
| `F32` | `qwen25-{05b,15b}_f32.bin` | 0 | — | 4.0 | **exists.** faithful control |
| `PACKED` | `qwen25-{05b,15b}_tqh.bin` | 2 | R0 | 0.5 | **exists.** fast control |
| `I8` | `qwen25-{05b,15b}_i8h.bin` | **5** | **R8 (new)** | **1.0** | **the treatment** |
| `T1` | `qwen25-05b_t1h.bin` | **5** | R0 | **1.0** | 0.5 B only. the SAME trits at 1 B/weight |

`R8` is round-to-nearest with one scale per output row: `s = amax(w_row)/127`,
`q = clip(round(w/s), -127, 127)`. It goes in `ternary/t2_rules.py` beside R0–R3 so the exporter
and any later probe cannot drift apart, exactly as the module's own docstring requires.

`T1` exists for one reason: it holds the *values* of `PACKED` at the *bytes* of `I8`. If `I8` and
`T1` time the same, speed on this rung is a pure byte effect and part B's rate needs no separate
defence. If they differ, something value-dependent is happening and the speed claim is scoped to
int8 values only.

## 6. Gates — registered now, wired into the runner before it runs

**`G-E60a` — PLANTED CONTROL ON THE BINARY (must FIRE before anything else counts).**
`donor_engine_e60.exe` and the frozen `donor_engine_e53.exe` must emit **token-identical**
generations on `F32` and on `PACKED`, both scales, same prompts, greedy, 32 tokens.
`FIRES` = 4/4 arms identical. Anything else → **STOP, the C change is not inert**, no other number
in E60 is reported.

**`G-E60b` — PLANTED CONTROL ON THE FIDELITY INSTRUMENT (must FIRE).** Scored against the stored
HuggingFace references, the instrument must reproduce E57: `F32` at **160/160** at both scales,
`PACKED` at **3/160** (`05b`) and **10/160** (`15b`). `FIRES` = all four within ±1.
If it does not fire, the instrument is broken and no fidelity verdict is issued.

*Why HuggingFace is the right reference here and was the wrong one in E59:* E59's treatment was a
kernel applied on top of an already-broken artifact reading 3/160, so the HF counter was **on its
floor and could not show damage** — the second face of
`feedback_gate_is_not_a_progress_meter`, registered yesterday. Here the control arm reads
**160/160**. The counter has the full 160 points of range in the direction that fails. The floor
rule is satisfied by construction, and this is written down before the run precisely so it can be
checked afterwards.

**`G-E60c` — SPEED IS A BYTE EFFECT.** At `05b`, `|rate(I8) − rate(T1)|` must be **within the
dispersion measured on the same sweep** (max half-width of the two arms' bootstrap intervals). Not
a tolerance picked in advance: the bar is the dispersion the axis actually shows, per
`feedback_gate_vs_measured_dispersion`. Prints `SEPARATED` / `WITHIN` / `UNRESOLVABLE`.

**`G-E60d` — THE SPEED QUESTION.** Bootstrap interval of `I8` at `05b` against the **50.0 tok/s**
bar: `ABOVE` if the whole interval exceeds 50.0, `BELOW` if the whole interval is under,
`STRADDLES` otherwise. Reported for `15b` the same way without a bar claim, since nobody expects
1.5 B at 1 B/weight to clear 50 (§7 says why).

**`G-E60e` — THE FIDELITY QUESTION.** `I8` vs the HF reference, greedy 160, at both scales, with
`dBPB` vs `F32` on the standard 24×512 slice (chance = `4.069819`):

| label | condition |
|---|---|
| `FAITHFUL` | ≥ **150/160** at BOTH scales **and** `dBPB ≤ +0.020` at both |
| `DEGRADED` | above E18's floor (`> 14/160`) but failing either clause |
| `AT-FLOOR` | `≤ 14/160` at either scale — the E18 part A floor, `11–12` plus E17's uninterpretable margin of 2 |

**`G-E60f` — the RANK partner.** `dBPB` is a SCORE; E14 §3 forbids a SCORE without a RANK partner.
The partner is the greedy agreement already in `G-E60e`, and it is named here so the pairing is
registered rather than assembled afterwards (E14 §6 forbids promoting a post-hoc metric to a gate).

**Conduct.** All speed arms **interleaved inside one sweep** (`for r in reps: for arm in arms`),
k ≥ 7, `--threads 6`, `--bench 160`. Foreign occupancy witnessed per cell against
`OCC_BAR = 4.39` (E52) and **reported, not cleaned**. A contended timing is reported as contended
with the direction of the bias stated. Every absolute tok/s carries ±5%; ratios do not.

## 7. Predictions — written before any number exists

Desk arithmetic, from E58's measured organ bytes (`05b` **247.0 MB/tok**, `15b` **771.8 MB/tok**
at exactly 0.500000 B/weight, so **494.0** and **1543.6 MB/tok** at 1.0) and E58's measured `Rem`
(0.427 / 0.668 ms). Per-row scales add < 1% and are charged in the runner, not here.

| if the `quant==1` kernel streams at… | `05b` | `15b` |
|---|---|---|
| **38.5 GB/s** (what the fp32 arm achieves) | **75.4 tok/s** | **24.5 tok/s** |
| 25.49 GB/s (E10's packed grand median) | 50.5 | 16.3 |

The kernel's own throughput ceiling is the open variable, and it is *not* obviously above the
wall: this branch has **a single accumulator chain** (E13 §8 named it owed, it is still owed),
one FMA per 8 weights at ~4-cycle latency ⇒ ~2 weights/cycle/thread ⇒ **~45 G-weights/s** on six
threads, only ~18% above the 38.5 GB/s the memory system delivers at 1 B/weight. It is a close
call by construction, which is why it is being measured instead of asserted.

1. `G-E60a` **FIRES** 4/4 — the `quant=5` alias is inert on existing paths.
2. `G-E60b` **FIRES** 4/4 — E57 reproduces.
3. `G-E60c` **WITHIN** — `I8` and `T1` are the same instruction stream.
4. `I8` at `05b` lands in **55–80 tok/s**; `G-E60d` reads **ABOVE**.
5. `I8` at `15b` lands in **18–26 tok/s**.
6. `G-E60e` reads **`FAITHFUL`**: **158–160/160** at both scales, `dBPB ≤ +0.005`.

**Falsification, stated now:** if `I8` at `05b` reads below **50 tok/s**, the "first faithful arm
above the bar" claim dies and the rung is a 2.6× improvement over fp32 that still misses. If
`G-E60e` reads below **150/160**, then post-hoc weight quantization is damaging on this donor at
1 byte too, and the programme's "post-hoc is fatal, training is the only route" finding extends
one rung further up — which would be a *stronger* result than a pass, and is the outcome this
brief would least like and most needs to be able to accept.

## 8. What E60 cannot claim, whatever it reads

- **It does not reach the goal's 10 B.** E34 measured the attention+head floor of the 10 B target
  shape at **2.147 G charged weights = 1.079 G moved bytes/token** at 0.5 B/weight. At 1 B/weight
  that floor **doubles**. A dense 10 B at one byte per weight moves ~10 GB/token ⇒ ~4 tok/s.
  **On the 10 B shape, int8 is strictly worse than ternary.** What it can do is set the *faithful*
  ceiling at that shape: E36's registered 10 B cell at 1.17% FFN activation reads 49.96 tok/s at
  0.5 B/weight, so the same shape at 1 B/weight reads **~25 tok/s** — against today's faithful
  ceiling of ~6. That is a halving of the remaining gap, not a crossing of it, and it will be
  written that way.
- **It says nothing about `--lutblk`.** That kernel requires `quant=2` (`:1553`) and E59 measured
  what it costs. The two levers do not compose on this engine as written.
- **It is post-hoc.** Every number will be a quantizer applied to a model that never saw it —
  the same class the programme keeps measuring as fatal at 0.5 B and training out at H0. If
  `G-E60e` reads `FAITHFUL`, the finding is *"the cliff is between 1 B and 0.5 B, not between 4 B
  and 1 B"*, which locates the cliff; it does not remove it.
- **Every absolute rate is ±5%** and the artifacts live on a USB 3.1 HDD; load time is excluded
  from `--bench` by construction and will be reported separately.

## 9. Owed / touched

- Adds `R8` to `ternary/t2_rules.py`; every existing rule untouched, and the parity gate that
  shares the module is unaffected because no existing caller can select `R8`.
- Adds `quant=5` to `donor_engine.c` (four sites, §4). `G-E60a` is the parity gate for it.
- Does **not** pay E13 §8's multi-accumulator debt on the non-packed path. If `G-E60d` reads
  `BELOW` and the measured GB/s lands near 25, that debt becomes the next experiment rather than
  a footnote, because it would be the thing standing between this rung and the bar.
