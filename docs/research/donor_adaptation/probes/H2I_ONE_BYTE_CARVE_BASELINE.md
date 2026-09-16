# H2I Phase A — one byte is inert; the matched carve hole is trainable-sized and rank-visible

**Brief:** `briefs/BRIEF_H2I_TRAIN_THE_CARVE_AT_ONE_BYTE.md` + addendum A

**Runner:** `s1/h2i_applied.py`, commit `c4a156f`

**Result:** `s1/results/h2i/h2i_applied_8L.json`, sha256
`c367acb45113e9ce68fb898de0b0833db61b6a1b84990b7ccc6467f82d2ef8cc`

**Donor:** Qwen2.5-1.5B rev `8faed761`; H0 run-3 q/o; layers 3,6,9,12,15,18,21,24;
E=256, hard k=16, E37 router; CPU fp32

**Cost:** 971 s CPU; no GPU and no timing claim

## Verdict

> **`PHASE_B_ELIGIBLE`.** On the exact H1 eight-layer scope, replacing only the FFN forward
> quantizer by one-byte R8 changes H0 by `−0.000016893 BPB`. Applying the hard carve then costs
> `+0.209070870 BPB`. The byte conversion is inert; selection is the live trainable object.

| arm | BPB |
|---|---:|
| intact control | 0.767594964 |
| H0 run 3 | 0.810022488 |
| R8 8L, k=E | **0.810005595** |
| R8 8L, E37 hard k=16 | **1.019076465** |

Both registered launch gates fire: one-byte absolute delta `0.000016893 <= 0.01`; carve delta
`0.209070870 > 0.05` with the required ordinal sign. The result is write-once and the published
H1 rows were hash-validated rather than remeasured.

## The rank check changes how the win must be judged

The applied R8 arm agrees with H0's own freshly captured continuations at only **9/160**
free-running positions, per prompt `[6,0,0,0,3]`. Under H0's fixed context it gives **99/160**
teacher-forced top-1, mean target rank **3.675**, and target rank <=5 at **143/160** positions.

So the target often remains nearby, but four trajectories diverge at their first token. This
is why the Phase B result must improve both score and rank; lowering BPB alone cannot be promoted
as a functioning carve.

## What changed relative to earlier evidence

E64 found int8 carve dearer than ternary over all 28 layers. On this matched eight-layer scope,
applied R8 is instead **0.077559667 BPB better** than H1's applied ternary `1.096636133`. The
registered E64-direction prediction is falsified. The two statements coexist because accumulated
post-hoc damage is not linear in layer count; neither scope licenses extrapolation to the other.

The old H2T route remains retired. Training ternary-format damage would attack an axis this test
and E66 show can be avoided. The next GPU object is R8 weights with the carve/router present
during training, starting from H0 run 3.

## Do not repeat

- Do not rerun these Phase A rows or the rank partner; their result hash is frozen.
- Do not use H1's ternary applied number as the H2I training threshold. The matched threshold is
  `1.019076465262247`.
- Do not infer rate: this PyTorch path computes the dense FFN before masking.
- Do not promote the 8-layer ordering against E64 into an all-layer law.
- Do not infer 10 B quality, or activation-int8/LUT behavior, from weight-only R8.

## Phase B handoff — READY_NOT_RUN

Phase B was preregistered in brief addendum B before its trainer/evaluator existed. The committed
apparatus is `s1/h2i_qat.py` plus `s1/h2i_eval.py`; the exact bundle is
`s1/_h2i_bundle`, with its sole launch command in `RUN.md`.

The real-donor CPU smoke passes every preregistered pre-GPU control: exact R8 rule, gradients,
hard/soft `k=E`, real `k=16` cardinality, live carve, exact E37 router initialization, one applied
update moving both depth extremes and the router, no non-finite microbatch, and exact
save/reload. Its artifact is `s1/results/h2i/h2i_qat_cpu_smoke.json`, sha256
`33ae14478a457e140b5de034fa2b9c31af448ac400308058d573fce675c9456e`.

The independently rehashed bundle contains 15 payloads / 463,452,303 bytes; manifest sha256 is
`a37d65fdc3e7833bb2b539abd43a07d958a98bc9dd65d091240716a5606abf20`. It needs one continuous
11-hour T4 session. No expected step count is inferred from the two-layer CPU smoke; the frozen
progress prediction is only `>=250` applied updates with a loadable checkpoint. After the T4,
the final artifact is adjudicated once on CPU fp32 against the exact Phase A score and rank gates.

---

## H2I Phase B v4 — terminal `SCORE-ONLY`: score learns, rank replication does not clear

**Canonical result:** `s1/results/h2i/h2i_eval_h2i_trained_s1.json`, SHA-256
`2180e1c5d01d903368b1b9119b99049ac5e0d48abbac8bbdbc460dca2151e805`.
**Canonical adjudication:** `s1/results/h2i/h2i_eval_h2i_trained_s1_adjudication.json`.

The frozen v4 guard returned `SCORE-ONLY` and the evaluator returned rc `3`: this is the
registered terminal scientific verdict, not an operational error. Training reached its time cap
at 2,286 steps / 2,282 applied updates in 39,615.08494114876 s (17.32943348256726 s/step), with
zero non-finite microbatches and manifest
`330fa237ebe3ea85c48447f52628a3f77f1f4cf0e7275ef7e122e9eb8b93ba56`.

| terminal comparison | BPB / result |
|---|---:|
| applied R8 hard k=16 | 1.0190764652622473 |
| trained hard k=16 | **0.905344219843237** |
| score delta | **−0.11373224541901028** (`TRAINING-HELPS`) |
| trained soft | 0.9117608687599223 |
| trained `k=E` | 1.6628228062405777 |
| trained experts + E37 router, hard | 0.9286241241806835 |

The experts account for `−0.09045234108156375` BPB against applied; the trained router adds
`−0.023279904337446533` over E37, and it beats STATIC at 2.654136780391849 versus
3.0665777063103286 nats/token. This establishes score learning and useful routing on the held-out
slice. It does not establish the combined branch claim: free 11/160 passes `>9` and mean rank
2.55625 passes `<3.675`, but teacher-forced top-1 95/160 fails strict `>99` (five hits short of
the minimum 100). Thus rank and combined `G-H2I` are false.

The B.6 predictions score: 1 **PASS**; 2 **PASS** (0.905344 within 0.86–0.96); 3 **FAIL only on
the teacher-forced clause**; 4 **PASS** (router beats STATIC); 5 **PASS** (0.905344 < 0.928624
with experts fixed); 6 **PASS**. For prediction 6, the downloaded periodic checkpoint is step
2250 with 2,246 applied updates; ZIP CRC reports no bad member, and
`numpy.load(..., allow_pickle=False)` opens 56 expected layer arrays.

No H2I export, engine, rate, 10B, scale, or target-donor promotion is licensed. Versions 1–3
remain `VOID_OPERATIONAL`; v4 closes the score+rank cell and authorizes no H2I rerun. The Kaggle
CLI's post-download Windows character-map error left a zero-byte raw-log placeholder; its empty
hash is recorded by the guard, which otherwise passed all final-pair, dependency, input and
metadata checks. It is an archival limitation, not a scientific gate outcome.
