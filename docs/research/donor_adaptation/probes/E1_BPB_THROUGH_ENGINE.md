# E1 — Does the model the ENGINE executes score the BPB that PyTorch says it does?

**Brief: `briefs/BRIEF_E1_BPB_THROUGH_ENGINE.md`, pre-registered at `f92af8a`, before any run.**
**Runner: `benchmarks/donor_adaptation/engine/e1_bpb_through_engine.py` (`cdb7119`).**
**Results: `density/results/e1_bpb_through_engine_qwen25-05b.json`,
`…_qwen25-15b_tqtqh_s24.json`, `…_qwen25-15b_f32_s4.json`.**
**Logs: `engine/e1_05b.log`, `engine/e1_15b_tq.log`, `engine/e1_15b_f32.log`,
`engine/rebuild_control.log`.**

---

## 0. Verdict

**Mechanical label, 0.5B, verbatim from the JSON: `LOOP-CLOSED`.**
**Mechanical label, 1.5B, verbatim from both JSONs: `INCOMPLETE`** — because brief §3
pre-registers the 1.5B fp32 arm as a *separate* 4-sequence subset, so neither of the two 1.5B
runs contains all the arms its own decision function needs. **Assembled across the two runs, the
1.5B meets every condition brief §4 lists for `LOOP-CLOSED`** (§2.3 below shows the assembly
term by term). The mechanical labels are reported as returned and were not rewritten, per T3 §1.3.

> **The engine scores what PyTorch scores.** Across two donors and five arms the largest
> disagreement is **`+1.53e-05` BPB** — `0.003 σ_seed`, and **130× inside** the tightest
> pre-registered gate. Every quality number this programme owns was a claim about a
> `transformers` module; on both donors it is now a claim about the deliverable.

**Two things were found on the way that matter more than the agreement itself:**

1. **The engine could not load a model over 2 GB, and blamed the file for it** (§4.4). The stated
   goal is a 10B model; at ternary packed that is ~5 GB. **The target model was not slow, it was
   unloadable**, and no speed probe could ever have found that.
2. **No R3 model had ever been exported.** The artifact behind every speed number in
   `SPEED_LEDGER.md` is R0/BitLinear158 — the *worst* of T2's five rules (§4.6).

---

## 1. The gates, which were checked before any BPB was compared

**Gate A — the weights must be the same weights.** The runner re-reads the exported binary,
unpacks it, and compares `codes × scale` against the tensor `t2_rules.r3_actsearch` produced in
the same process. Reported as registered (product bit-identical), and additionally split into
**A1 codes** (exact) and **A2 scales** (ulp) as a diagnostic — the one additive departure, §6.1.

| run | arm | tensors | codes compared | codes differing | scales differing | worst ulp | as registered |
|---|---|---|---|---|---|---|---|
| 0.5B | F32 | 290 | — (fp32 round-trip) | — | — | — | **PASS**, worst \|diff\| `0.000e+00` |
| 0.5B | TQ | 168 | 357,826,560 | **0** | **0** | `0.00` | **PASS** `0.000e+00` |
| 0.5B | TQH | 169 | 493,961,216 | **0** | **0** | `0.00` | **PASS** `0.000e+00` |
| 1.5B | TQ | 196 | 1,310,195,712 | **0** | **0** | `0.00` | **PASS** `0.000e+00` |
| 1.5B | TQH | 197 | 1,543,569,408 | **0** | **0** | `0.00` | **PASS** `0.000e+00` |
| 1.5B | F32 | 338 | — (fp32 round-trip) | — | — | — | **PASS**, worst \|diff\| `0.000e+00` |

`493,961,216` is exactly the 0.5B's active weights per token, so on arm TQH the gate covers
**every weight the engine touches**. The fp32 arms verify every tensor including norms and biases.

**Gate A did not pass on the first attempt, and was not loosened to make it pass.** The smoke
fired it at `2.980e-08` — exactly `2^-25`, one ulp — and the diagnosis took three measurements and
found two real bugs in the exporter; that history is §4.5 and commit `cdb7119`.

## 2. Results

Shared `heldout` slice, `ids_sha256 = a1a48dc9…`, 24×512, **51,870 scored bytes**; calibration
`calib` 32×512 seed 42424 at `--calib-seqs 32`; `--threads 6`, recorded. `Δ = BPB(engine) −
BPB(PyTorch)` on weights Gate A proved identical.

### 2.1 Qwen2.5-0.5B — all three arms, full slice

| arm | converts | BPB PyTorch | BPB engine | Δ | per-seq max \|Δ\| | zero frac | gate |
|---|---|---|---|---|---|---|---|
| **F32** | 0 | `0.871795113818918` | `0.871810461083424` | **`+1.5347e-05`** | `2.262e-05` | — | PASS (bar `0.002`) |
| **TQ** | 168 | `4.509151256008091` | `4.509163909048454` | **`+1.2653e-05`** | `4.659e-05` | `0.4914` | PASS (bar `0.01`) |
| **TQH** | 169 | `4.531219311358794` | `4.531233733626962` | **`+1.4422e-05`** | `5.552e-05` | `0.4915` | PASS (bar `0.01`) |

`OUTCOME_LABEL: LOOP-CLOSED`.

### 2.2 Qwen2.5-1.5B rev `8faed761…` — where every quality number lives

| arm | converts | BPB PyTorch | BPB engine | Δ | per-seq max \|Δ\| | zero frac | gate |
|---|---|---|---|---|---|---|---|
| **F32** (4 seqs) | 0 | `0.702308545920147` | `0.702318880488138` | **`+1.0335e-05`** | `1.336e-05` | — | PASS (bar `0.002`) |
| **TQ** | 196 | `3.484251280665358` | `3.484253077409102` | **`+1.7967e-06`** | `1.389e-05` | `0.4714` | PASS (bar `0.01`) |
| **TQH** | 197 | `3.475705978857720` | `3.475706691632780` | **`+7.1278e-07`** | `1.486e-05` | `0.4713` | PASS (bar `0.01`) |

The fp32 arm is the pre-registered 4-sequence subset (9,203 scored bytes); it is an *agreement*
test and its BPB is not a number this programme quotes.

### 2.3 The 1.5B label, assembled term by term

Brief §4's `LOOP-CLOSED` requires: Gate A passes; `|Δ_F32| ≤ 0.002`; `|Δ_TQ| ≤ 0.01`;
`|Δ_TQH| ≤ 0.01`.

| term | value | bar | met |
|---|---|---|---|
| Gate A, all arms | `0.000e+00` | bit-identical | ✓ |
| `\|Δ_F32\|` | `1.0335e-05` | `0.002` | ✓ (194×) |
| `\|Δ_TQ\|` | `1.7967e-06` | `0.01` | ✓ (5,566×) |
| `\|Δ_TQH\|` | `7.1278e-07` | `0.01` | ✓ (14,030×) |

Every term is met. The `INCOMPLETE` labels are a consequence of the brief splitting the 1.5B
across two invocations, not of any term failing.

## 3. Replication against T2b — the part that could have failed

The exporter and T2b's runner share **only** `t2_rules`. The calibration capture, the packing,
the file format, the read-back and the BPB harness are different code, and E1's PyTorch reference
is built by reading the weights back **out of the exported binary**.

Comparison at full precision against `density/results/t2b_organs.json`, which is the file, the
arm and the organ set named together per T3 §6.1:

| E1 arm | T2b arm | organ set | E1 `BPB(PyTorch)` | T2b `bpb` | residue |
|---|---|---|---|---|---|
| **TQ** | **FA** | 196 tensors, FFN + attention | `3.4842512806653585` | `3.4842512670844652` | `1.3581e-08` |
| **TQH** | **FAH** | 197 tensors, + `lm_head` | `3.4757059788577203` | `3.4757059788577203` | **`0.0` — bit-identical** |

As Δ against the standing baseline `0.7675949584171732`: `+2.7166563222481854` vs
`+2.7166563086672921`, and `+2.7081110204405472` vs `+2.7081110204405472`.

Mean ternary zero fraction also matches to every printed digit: `0.4714445757622622` (E1 TQ vs
T2b FA) and `0.47132473926858853` (E1 TQH vs T2b FAH).

> The JSON's own `replication_vs_t2b` block reports larger residues (`3.22e-07`, `2.04e-08`)
> **because the runner stores the constants rounded to six decimals** (`T2B_FA = 2.716656`).
> The table above is computed in this report from the file, at full precision. The runner's
> rounded copies are a convenience; the file is the source.

## 4. What the numbers say

### 4.1 The delta does not grow with the arm, and that is the informative part

Not the size of the deltas — their **flatness**. On the 0.5B: `1.53e-05` (fp32), `1.27e-05`
(ternary), `1.44e-05` (ternary + ternary head). The ternary path costs the *same* as the fp32
path. A bug in unpacking, in the per-row scale, in the tile-major permutation or in the ternary
kernel would grow with the number of converted tensors; this does not. It is the signature of a
fixed fp32 accumulation-order difference between two implementations that are both correct.

Per-sequence maxima (`2.26e-05`, `4.66e-05`, `5.55e-05` on the 0.5B; `1.39e-05`, `1.49e-05` on the
1.5B) confirm no single sequence carries the aggregate: a mean can hide one divergent sequence,
and none is hiding.

### 4.2 The 1.5B agrees more closely than the 0.5B, and no mechanism is claimed

`7.1e-07` and `1.8e-06` against `1.3e-05`–`1.5e-05`: a factor of ~7–20 in the *better* direction
on the *larger* model. **Reported as a fact and not explained.** Two donors do not distinguish a
trend from a coincidence, and this programme has three retractions on record from explaining a
two-point pattern. If it matters later it needs a third donor, not a paragraph.

### 4.3 "The head is free" is a 1.5B finding, and it does not reproduce on the 0.5B

| donor | ternary FFN+attn | + ternary head | head's marginal cost |
|---|---|---|---|
| **1.5B** (T2b, replicated here) | `+2.716656` | `+2.708111` | **`−0.008545`** — free, and slightly negative |
| **0.5B** (this probe, new) | `4.509151` (Δ `+3.637`) | `4.531219` (Δ `+3.659`) | **`+0.022068`** |

Ternarizing FFN+attention also costs far more on the small donor: `+3.637` against `+2.717`,
measured against each donor's own fp32 arm. Both facts point the same way — the smaller model has
less redundancy to spend — which is the direction the scale-up argument needs. **It is one point,
not a trend**, and it is stated here so that T2b §3's "the head is free" is not quoted as a
property of donors in general. `0.022` is 4.4 σ_seed, so it is not noise on the 0.5B.

### 4.4 The engine could not load a model over 2 GB — the finding that outranks the rest

The 1.5B fp32 arm died with `FATAL: bad magic -- not a QWENDON1 file` on a **6,174,857,268-byte**
export whose magic was intact: E1's own reader had just walked all 338 tensors of it with a
bit-exact round-trip. Measured with a scratch probe against the same file:

```
sizeof(long)=4  LONG_MAX=2147483647
fseek(f,0,SEEK_END) returned -1   ftell = 0
read 8 magic bytes: 'QWENDON1'
```

`long` is 32 bits on Windows even in an x64 build. Above 2 GB `fseek(SEEK_END)` **fails** and
`ftell` reports `0`; `load()` allocated a zero-byte blob, read zero bytes — which equals the zero
it asked for, **so the `short read` guard passed** — and `memcmp` compared the magic against an
empty buffer. **The error message accused the artifact of a fault in the reader.**

**Why it had never fired:** the largest file the engine had ever been given was the 0.5B fp32
export at **1.84 GB**, just under the ceiling. Every speed number was taken on the 0.8 GB packed
model. Nothing had ever handed it a large model.

**Why it outranks the agreement result:** the goal is a 10B model, which at ternary packed is
~5 GB. The target was **unloadable**, and speed probing could not have revealed it, because speed
is measured on small files.

Fixed at `33f0add`: 64-bit offsets chosen by platform (`_fseeki64`/`_ftelli64` on Windows,
`fseeko`/`ftello` elsewhere — the class, not this machine), a real error when the seek fails, and
the read done in 1 GB chunks because not every C runtime honours a single `fread` above 2 GB.

**Planted control, run before the rebuilt binary was allowed to produce any new number:** it
re-scored the already-measured 0.5B TQ artifact and returned `NATS_TOTAL 162120.4241599279`,
`N_PREDICTED 12264` — bit-identical to the pre-rebuild value. The arithmetic did not move.
**Known-positive for the fix:** the 5.75 GB file then loaded and scored (§2.2), and the PyTorch
side reproduced its value from the failed run exactly, so the reference is deterministic across
processes.

### 4.5 Gate A fired first, and repairing the instrument found two more bugs

The smoke fired Gate A at `2.980e-08` = `2^-25`. The gate was not loosened. Three measurements:

| hypothesis | measurement | verdict |
|---|---|---|
| the codes are wrong | **0** differing of 357,826,560 | no — the quantization was never in doubt |
| the capture is nondeterministic | reproducible within one process: `True` | no |
| **thread count** | 6 vs 1 threads: **102,123** `act_rms` elements differ, worst `1.9e-06` | **real, but not the residual cause** — 6 vs 6 in two processes differed by **0** |
| **`attn_implementation`** | exporter used HF's default **sdpa**; every probe uses **eager**. `act_rms` differs on **142,977** elements, worst rel `8.2e-06`, moving **132,844** stored scales by up to **6 ulp** | **the cause** |

Both were fixed and both are recorded in the sidecar. Gate A then passed exactly as pre-registered.
The first partial hypothesis was reported as partial rather than as a fix that worked.

A third bug fell out of the same work: **`--quant ternary` had been dead** since `--rule` landed
(`w_tern` squeezed a scale `quantize()` already returns as a `[out]` vector). Nothing caught it
because every artifact this programme built used `--quant packed`.

### 4.6 No R3 model had ever been exported

`D:/_ktmp/qwen05b_packed.bin`'s sidecar has **no `rule` field at all** — it predates `--rule`, so
it is R0/BitLinear158, mean zero fraction `0.327`, which T2 showed to be the worst of the five
rules and *indistinguishable from random signs*. Every number in `SPEED_LEDGER.md` was measured on
that file. The format is byte-for-byte the same, so **the speed numbers stand**; but until this
probe, the artifact demonstrating this programme's best quality claim did not exist on disk.
Found by reading the sidecar of the file that is actually there, rather than the command that was
supposed to have written it.

### 4.7 T2b's `untied` field means something other than it appears to

`t2b_organs.json` reports `"untied": false` on arm FAH, an arm that ternarizes a **tied** head —
which would mean the embedding table was ternarized too, i.e. a different model from the one the
exporter writes. It is not: **E1's arm TQH, which certainly unties and keeps the embedding fp32,
reproduces FAH's BPB bit-identically** (§3), and two different models cannot do that.

The explanation is in the file: arm **`I`** (the identity control, 197 tensors) reports
`untied: true`, and every later arm reports `false`. T2b's `restore()` restores weights but never
re-ties, so from arm `I` onward the head stays untied and the field records *"did this arm perform
the untie"*, not *"is the head untied here"*. Benign — untying is what is wanted, and arm `I`
returns the base BPB exactly (`0.7675949584171732`) — but it is **state leaking across the arms of
a sweep**, and it was caught by a replication, not by the sweep itself.

## 5. What this closes, and what it licenses

- **INDEX §4 item 1 is closed on both donors.** `--bpb` has now run at scale; the parity gate is
  no longer the only bridge between the PyTorch numbers and the artifact.
- **T2b's `+2.708111` is the runtime's number**, not a simulation's, and it is reproduced here by
  a second runner through a file round-trip.
- **What it does not license:** nothing about speed. `--bpb` runs a full prefill per position and
  the machine was doing other work; brief §3 forbids quoting any timing from this probe, and the
  seconds in §2's source JSONs are bookkeeping. A contended timing is not a timing.
- **What it does not license, second:** the fp32-arm agreement does not extend to `--lut`, which
  is a different numeric object (`rel l2 3.10e-02` at G=32) and was deliberately excluded.
- **E2 is unblocked.** `briefs/BRIEF_E2_RMSNORM_FOLD.md` (`338d187`) carries a Gate E that reads
  `LOOP-CLOSED` here; its contrasts may be stated as claims about the deliverable.

## 6. Departures from the brief, and what the brief got wrong

### 6.1 One additive departure, declared in the runner's docstring before the run

Gate A is **split** into A1 (codes, exact) and A2 (scales, ulp, tol 64) as diagnostics. The
brief's single gate — dequantized product bit-identical — is still evaluated and reported as
written, and it passed on its own terms on every arm. The split adds information; it removes no
requirement. The `ULP_TOL = 64` is a ceiling, not a fit: the worst value ever observed was 6, and
after the `eager` fix it is 0.

### 6.2 The brief under-specified the 1.5B label

Splitting the 1.5B across two invocations means neither run can evaluate brief §4's decision
function, and both return `INCOMPLETE`. That is the brief's fault, not the runner's, and §2.3
assembles the label by hand rather than letting the runner rewrite it. **A pre-registration that
splits a decision across runs must say which run owns the label** — the same class as T3's
self-contradicting constant, and the second time in three probes that the brief, not the code,
produced the mechanical label.

### 6.3 A collision the brief did not foresee

The runner named its result file after the model alone, so the pre-registered 1.5B fp32 **subset**
run was about to overwrite the 55-minute TQ/TQH result written under the same name. Fixed at
`72da215`: a subset run now carries its arms and sequence count in the filename, and only the
canonical run (all three arms, full pinned 24) keeps the bare name. The 1.5B ternary JSON is
committed under the name the fixed runner would have produced; the log's `wrote` line still shows
the old path, which is what actually happened.

## 7. Reproduction

```
cd benchmarks/donor_adaptation/engine
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm

E1_THREADS=6 python e1_bpb_through_engine.py --arms F32,TQ,TQH --seqs 24 --calib-seqs 32
E1_THREADS=6 python e1_bpb_through_engine.py --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 --arms TQ,TQH --seqs 24 --calib-seqs 32
E1_THREADS=6 python e1_bpb_through_engine.py --model Qwen/Qwen2.5-1.5B \
  --revision 8faed761d45a263340a0528343f099c05c9a4323 --arms F32 --seqs 4
```

Run serially — never two heavy jobs at once. Wall clock as measured: 2,074 s (0.5B, three arms),
3,269 s (1.5B ternary), 431 s (1.5B fp32 subset). Exports land in `D:/_ktmp/e1`; `--reuse` skips
an export whose `.bin` already exists. Disk: the 1.5B fp32 export alone is 5.75 GB.
