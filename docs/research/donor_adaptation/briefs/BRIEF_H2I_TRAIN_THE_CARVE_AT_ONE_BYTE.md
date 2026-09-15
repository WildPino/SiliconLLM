# BRIEF H2I — train the carve at one byte, not the byte conversion

**Status:** Phase A pre-registered 2026-09-15, before its runner exists and before any H2I
number is measured. Phase B is contingent on Phase A and receives a separate addendum before
GPU execution.

**Cost now:** CPU only. No rate. No T4 request in Phase A.

## 0. The question

Can the selection/routing damage measured by E64 be trained away while the FFN stays in the
one-byte `R8` representation that E60/E62/E66 measured as faithful?

This is not `H2T`. That proposal trained the ternary body, paying GPU to heal a format loss
which E66 shows can be avoided. H2I keeps the byte conversion and trains the part still broken:
the carve and its router.

## 1. Why this experiment, from measured facts only

| fact already banked | consequence here |
|---|---|
| E66: 7 B `R8`, no fold, is +0.000378 BPB from fp32 and reproduces 4/5 complete trajectories | one byte is a faithful starting representation; do not train ternary-format damage |
| E64: at 1.5 B the `R8` carve costs +1.827–1.945 BPB; at `k=16`, +1.940853 and the full model is above chance | the selection hole survives, and is larger on int8 than on ternary |
| H1: training eight ternary-carved layers improves 1.096636 → 0.962593; the trained router becomes helpful | applied and trained selection are different objects; the training mechanism is live |
| E40: `R128` buys 50 tok/s at about 6% FFN activation and >100 tok/s near 1.7% | `k=16/256 = 6.25%` is the good-target structural rung; `k≈4` is the later excellent rung |

No scale transfer is made. H2I uses the 1.5 B donor because it fits one T4 and because E64 and
H1 were measured there. It is a mechanism falsification before any larger claim.

## 2. Phase A object — the matched applied baseline

Start from the exact H1 start state:

1. Qwen2.5-1.5B revision `8faed761d45a263340a0528343f099c05c9a4323`;
2. H0 run-3 trained ternary low-rank `q/o` factors installed;
3. the same eight layers `[3,6,9,12,15,18,21,24]`;
4. the same D0c `E=256` partition and E37 fitted routers;
5. hard engine-equivalent `{0,1}` gate, `k=16`;
6. **only the three FFN quantizers change:** H1's activation-aware ternary `R3` becomes
   per-output-row int8 RTN `R8`, exactly `amax/127`, round, clamp `[-127,127]`.

The comparison is therefore within one shape and one start state:

| row | FFN representation in 8 layers | selection |
|---|---|---|
| `h0-run3` | fp32 | none |
| `int8-8L` | `R8` STE forward | `k=E`, no carve |
| `applied-int8-8L` | same `R8` values | E37 router, hard `k=16` |

The published H1 `ternary-8L=0.947851138055396` and
`applied-8L=1.096636132580995` are hash-pinned and **not remeasured**. They are descriptive
cross-format partners, not thresholds imported into the new gate.

## 3. Frozen inputs

| input | bytes | sha256 |
|---|---:|---|
| H0 run-3 factors | 352,967,686 | `dd62482d65accb34129a7b7a3f8baf75b0ccf3c4fdf2481233f1b9ce823927d8` |
| D0c `labels_E256.npz` | 859,537 | `c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c` |
| H1 activation statistics | 1,189,630 | `49fd2f659a37237ead850ea86057664b74db4f2286dc7146abf87e197e6887fe` |
| E37 routers | 44,046,858 | `42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a` |
| published H1 applied result | 1,605 | `6ebc333f74f0f337567a99e527cdfa7750d092161cc5ca2d1836a510557f1b17` |
| H1 held-out slice file | 98,564 | `110a90ab358efcacdc090fdf0fce62bb74d6d06df02d1083cf4cf7e752529d35` |

The runtime slice must still report 24×512, 51,870 scored bytes and IDs sha256
`a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
Hashes are identities, not evidence that the scientific gate fires.

## 4. Phase A controls and gates

All controls fire before `applied-int8-8L` is read.

### `G-H2Ia` — provenance

The runner must be the exact committed HEAD blob. Every input in §3 must match byte count and
sha256. The published H1 JSON must contain the exact four registered BPBs, layers, `k`, `E`,
model revision and slice identity. Any mismatch refuses before model loading.

### `G-H2Ib` — the R8 implementation is the shipped rule

On deterministic toy tensors, H2I's STE forward value must be bit-identical to
`t2_rules.r8_int8_rtn`: one scale per output row, codes in `[-127,127]`. A planted
non-ternary tensor must use at least five distinct code values; otherwise a ternary rule has
been mislabeled int8. Under backward, every fp32 master must receive a finite non-zero gradient.

### `G-H2Ic` — carve wiring

On the toy and one real layer:

- hard `k=E` must be bit-identical to the uncarved int8 FFN for **any router**, zero tolerance;
- hard `k=16` must activate exactly `16 × 35 = 560` neurons in the 1.5 B shape and must differ
  from `k=E`;
- the installed labels and router shapes must match exactly.

### `G-H2Id` — inherited instrument controls

On the frozen CPU fp32 instrument:

- intact BPB reproduces `0.767594964119663` within `1e-5`;
- H0 run-3 reproduces `0.810022487699936` within `1e-5`.

Failure voids all H2I cells. These are necessary same-instrument controls, not new findings.

### `G-H2Ie` — one byte is inert enough on the matched eight layers

`abs(int8-8L − h0-run3) <= 0.01 BPB`.

This is deliberately much looser than E62/E66's observed one-byte errors and is a launch gate,
not a claim of exact equivalence. If it fails, H2I is not selection-only and Phase B may not be
described that way.

### `G-H2If` — the applied selection hole is live

`applied-int8-8L > int8-8L`, ordinal, and the difference must exceed `0.05 BPB` before GPU
training is worth launching. If it is at most 0.05, the matched object is already close enough
that training this eight-layer proxy has little information value.

`applied-int8-8L` becomes Phase B's frozen matched threshold. It is written once to a new result
file and then hash-pinned in the Phase B addendum; it is never remeasured after training.

## 5. Rank partner — measured now, never inferred from BPB

Using the five frozen E6 prompts, Phase A first records H0 run-3's own 32-token greedy
continuations. It then measures `applied-int8-8L` against those exact trajectories:

- free-running positional agreement out of 160;
- first divergence and per-prompt counts;
- teacher-forced top-1 agreement and mean target rank on the frozen H0 trajectories.

This is descriptive in Phase A: no rank threshold is invented after seeing it. Phase B will
freeze its rank gate against this exact baseline before GPU execution. The purpose is to prevent
a BPB-only improvement from being promoted as a functioning model, the failure E16/E20 exposed.

## 6. Predictions, registered before the runner

1. `G-H2Ia–d` all fire.
2. `G-H2Ie` fires and `int8-8L` is within **0.002 BPB** of H0 run-3. The formal bar remains 0.01.
3. `G-H2If` fires by more than 0.05 BPB.
4. Despite the better uncarved base, `applied-int8-8L` is **worse** than H1's ternary
   `applied-8L=1.096636`, matching E64's sign. Descriptive magnitude prediction:
   **1.15–1.50 BPB**.
5. Free-running agreement is below **150/160**. This predicts that one byte preserves weights
   but does not make the severe selection rank-neutral.

## 7. What Phase A decides

| reading | action |
|---|---|
| controls fail | repair apparatus; no scientific number |
| one-byte delta >0.01 | H2I is not selection-only; diagnose quantizer mismatch before GPU |
| carve delta <=0.05 | do not train this proxy; selection is already cheap on the matched scope |
| one-byte delta <=0.01 and carve delta >0.05 | write Phase B addendum, adapt H1's trainer to R8 and run CPU toy/real smoke |

Phase B's intended object is one continuous 11-hour T4 session, same eight layers and `k=16`,
starting from H0 run 3 rather than from H1's already-trained ternary checkpoint. The quantizer
change should reduce per-step cost because R8 has no ten-point threshold search, but no throughput
or step-count prediction is registered until the actual trainer smoke measures it.

## 8. What H2I cannot establish

- No rate: training computes the dense FFN before masking. Engine rate remains E40/E63's domain.
- No 10 B quality transfer and no width/depth scale law.
- No `R128` rank-axis healing: H0 factors are a fixed inherited base here.
- No excellent-target claim: `k=16` prices the ~50 tok/s rung, not `k≈4` for ~100 tok/s.
- No activation-int8/LUT claim. That is a separate forward quantizer and must remain a separate
  partner so weight healing cannot hide activation damage.
- No result from the T4 until CPU fp32 BPB **and** rank partners adjudicate it.
