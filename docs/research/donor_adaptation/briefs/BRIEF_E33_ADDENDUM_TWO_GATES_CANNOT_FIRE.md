# BRIEF E33 — ADDENDUM: two of my own gates cannot fire as written

**Pre-registered. Pushed before the runner runs and before any artifact is exported.** This
document exists because I checked `BRIEF_E33_IS_DOWN_THE_WHOLE_COST.md` (`2fa83e9`) against
`synth_export.py`'s accounting and `results/e26_carve_cost.json` **before** taking a measurement,
and found two gates that are wrong. Correcting them after seeing a number would be worthless, so
they are corrected here, with the reasoning, and the original brief is **not edited**.

---

## 1. `G-E33C` is arithmetically unsatisfiable — my error, and the confound it hides

The brief says:

> All `quant==4` carved artifacts … **byte-identical activation volume across `E`** by
> construction: the exporter keeps `k/E` fixed, so every arm charges the same active weights per
> token and only the *contiguity* changes.
> **`G-E33C`** — every arm's exporter JSON must report the same `active_weights_per_token`. If the
> arms do not charge identically, the comparison is a size comparison and the probe is VOID.

**That is false, and `synth_export.py:58` says why in its own docstring.** The FFN term under a
carve is

```
ffn = E*D  +  3*D*(F/E)*k          # router  +  kept neurons        (synth_export.py:66)
```

Holding `k/E` fixed holds the **second** term fixed — `(F/E)·k = F·k/E` — and leaves the
**router** term `E·D` free. **The router is CHARGED** (E26 insisted on this: leaving it out
"would flatter the carve by 50.3 M weights a token at the goal's shape"), and it *shrinks* as the
carve gets coarser. Computed from the exporter's own function, not from a table:

| arm | `GSZ` | router weights | `active_weights_per_token` | vs `E256` |
|---|---|---|---|---|
| `S15-E256` k=64 | 35 | 11,010,048 | 687,538,176 | — |
| `S15-E64` k=16 | 140 | 2,752,512 | 679,280,640 | **−1.201%** |
| `S15-E16` k=4 | 560 | 688,128 | 677,216,256 | **−1.501%** |
| `T10-E256` k=64 | 56 | 50,331,648 | 4,311,744,512 | — |
| `T10-E64` k=16 | 224 | 12,582,912 | 4,273,995,776 | −0.875% |
| `T10-E16` k=4 | 896 | 3,145,728 | 4,264,558,592 | **−1.094%** |

Exact equalisation is **not available at integer `k`**: making `S15-E16` charge exactly
`687,538,176` needs `k = 4.1429`, and `S15-E64` needs `k = 16.4571`.

**Why this matters and why it is not cosmetic.** The confound points *in the direction of my own
prediction*: the coarse arm is handed a free `1.5%` (S15) / `1.1%` (T10) head start in tok/s
before any locality effect exists. Against a predicted `1.25×` that is small — but it is exactly
the kind of "plausible artefact" this programme's planted-control law exists to catch, and it is
mine.

### The amendment, registered here

- **`G-E33C` as literally written is recorded FAILED, and the failure is reported as a defect of
  my brief, not of the machine.** It is not quietly widened and it does not VOID the probe on its
  own — a `1.5%` charged difference is not "a size comparison", which is what the gate was written
  to catch.
- **`G-E33C′` (replacement, decides VOID):** every carved arm of a shape must charge within
  **2.0%** of the `E256` arm, from the exporter JSONs. Above that the probe is VOID as a locality
  experiment. Predicted values are in the table above; they are fixed before the run.
- **The verdict is reported TWICE, and both numbers go in the result document:**
  - **raw** `S15-E16 tok/s ÷ S15-E256 tok/s` — the cell the brief registered, unchanged, and the
    one the bands in §4 of the brief are applied to;
  - **byte-normalised** `raw × (act_E16 ÷ act_E256)` = `raw × 0.98499` at S15 and
    `raw × 0.98906` at T10 — which removes the router confound exactly, in E25/E26's own charged
    convention.
  - **If the two land in different bands, the probe reports the NORMALISED band as the finding
    and says so in the first line.** Registered now, before the number exists.

## 2. `G-E33A` is anchored to an absolute E26 refused to stand behind

The brief says `S15-E256` must reproduce **E26's published carved rate** within `±10%`. From
`results/e26_carve_cost.json`, that rate is `S15-K64 = 52.557 tok/s` (mean of `54.46 / 52.72 /
50.49`).

**But E26's session was thermally slow and E26 itself ruled that only its RATIOS survive** — its
own dense S15 read `26.637` where E28, on the same box with the same dense file, read `29.30`
(`+9.99%`). E26's registered anchor "passed by 0.19 of a point". So a `±10%` bar on E26's
*absolute* carved rate is a bar on which session ran warmer, not on whether this probe is
comparable to the measurement it corrects — and it can VOID a perfectly good run, or pass a bad
one, for the wrong reason.

### The amendment, registered here

- **`G-E33A` (absolute, as written) is still computed and reported**: `S15-E256` vs `52.557`,
  `±10%`. It is informational. **It does not VOID on its own.**
- **`G-E33A′` (ratio, decides VOID):** `S15-E256 ÷ S15-DENSE` measured **in this session** must
  reproduce E26's own `ratio_vs_dense = 1.9731` within `±10%`. Same-session ratio against
  same-session dense, which is the only form E26 left quotable.
- `G-E33B` (dense vs E28's `29.30`, `±10%`) is unchanged and still decides, because it is the
  same-box, same-file, same-format comparison E28 published as an absolute.

## 3. What does NOT change

Everything else in `2fa83e9` stands exactly as pushed: the arms, `PT_BLK = 64`, the verdict cell,
the bands (`≥1.18` / `1.06–1.18` / `<1.06`), all six predictions, §6's dispositions and §7's
list of what E33 may not claim. **Prediction 2 — `LOCALITY-CONFIRMED`, ratio `1.18–1.32` — is
re-affirmed here unchanged, now against the normalised number as well as the raw one.** The desk
model says `1.25`; a `1.5%` confound does not move that prediction into a different band, and I
am not moving it.

## 4. The honest summary of this addendum

I wrote a gate that says "these must be identical" about quantities I had not computed, and an
anchor against an absolute the source probe had already disowned. Neither error would have been
visible in the result — the first would have VOIDed a good probe on a `1.5%` technicality, the
second would have decided VOID on room temperature. **Both were found by reading the exporter and
E26's JSON before running, which is the only time finding them is worth anything.**
