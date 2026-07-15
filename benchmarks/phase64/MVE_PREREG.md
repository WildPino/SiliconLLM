# MVE — Pre-registration (Phase 64.4, stage-0, $0)

**Status: apparatus built and smoked; NOT yet launched. This file is pushed BEFORE the run (push-before-run, in
falsifiable form: the commit date is the witness). Costs re-price at the MVE; GATES DO NOT MOVE.**

Owner launches. Builder does not launch gate-bearing runs.

---

## 1. What the MVE is

The full ladder recipe, end-to-end, at pilot scale, on the pilot domain — the minimum run that can *fail* for a
reason that would also make the real ladder fail. It is not a quality run: **quality at 22-30M on TinyStories is
RECORDED, never gated** (TS anchors are for property shape, not SOTA).

| | |
|---|---|
| student | S0 ladder recipe: `D256 N96 L8 swa@5 dt16`, dReLU-gated MLP → ternary, MoE `E32×h128 top8`, recall slot. **11.0 M dense → 30.1 M total / 11.2 M active** after the MoE upcycle |
| corpus | TinyStories (P55 pinned), 64 MiB = 32.7 M student tokens; 90/10 train/val |
| tokenizer | **V=1024, `cartography.Bpe` — the production class/recipe**, retrained on TS. P55's own id stream is deliberately NOT reused: the MVE must exercise the production tokenizer path (train → encode → exp_len → anchor), not a look-alike |
| teacher | **Qwen2.5-Coder-1.5B** (the sealed production teacher — the point is to exercise the same tokenizer/span-mapping path, not to get good TS BPB). Licence verified live: **apache-2.0** |
| curriculum | A data+tokenizer → B teacher logits → C KD pre-train fp → D QAT ternary → E MoE upcycle + recall/InfoNCE → F reverse-KL. Token split 55/20/20/5 (plan §4) |

## 2. The five sealed gates (plan §6.4 — verbatim in force)

1. **D3** — cross-tokenizer KD beats plain CE-on-teacher-text. *Arms: `--arm kd` vs `--arm ce`, one variable, same
   text, same steps, same seed.* Read on **val BPB**.
2. **Pipeline** — A→F executes end-to-end **including resume-from-preemption** (Kaggle 12h cap is the real threat).
3. **D4 clause 2, first read** — the InfoNCE/recall stage does not destabilize the curriculum. *Arms: `--recall on`
   vs `--recall off`.* The recall slot's residual gate is **initialized to zero**, so the *insertion* is a
   mathematical no-op: what is on trial is what the InfoNCE pressure subsequently learns, which is the right variable.
4. **KD-then-QAT stability** — no divergence at the QAT switch. Instrumented: val BPB is measured immediately
   before and after each stage transition.
5. **Throughputs** — measured train tok/s and teacher scoring tok/s → the STOP-B re-priced table.

## 3. What the smoke ALREADY established (before launch, $0) — and the one decision it forces

**(a) The teacher's cost was mis-metered.** Offline KD needs the teacher's distribution over a corpus we already
have. That is **teacher-forced scoring — prefill only, no autoregressive decode**. The T1 `gen tok/s` column
(batch-1 decode; physically inverted 0.5B < 1.5B) was never the meter for this cost. Measured on the 3060
(fp16, ctx 2048, stride 1536, batch 4): **2283 tok/s**. Batch is *not* the lever (b2→b4: +3%); the full-vocab
softmax is — taking top-K on the logits and normalizing by the row logsumexp is what unblocks the card.

**(b) The sealed span-KD design delivers ~15% of the teacher's information.** Measured, not argued
(`kd_information.py`): the projection of the teacher's top-K onto the **first student token** of the segment is a
many-to-one collapse — `" the"`, `" then"`, `" they"` share a student first-token, so the teacher's uncertainty is
*summed away*. H(teacher top-K) = 2.09 bits → **H(q projected) = 0.32 bits (15% retained)**; the target is
near-one-hot on 74% of anchors. The map is not broken (93% of its mass sits on the true next token) — it is
**correct and nearly empty**. Raising the student vocab does not fix it and mildly worsens it (V=4096: 13%).

**(c) The information is recoverable at ZERO extra teacher compute.** By the chain rule the teacher's uncertainty
factorizes across the student tokens of the span: the choice between `" the"/" then"/" they"` is not made at `" t"`,
it is made at the *next* student token — a position the sealed design supervises with a hard CE label. Reading the
**same stored K=32 rows** at every interior position, conditioned on the bytes already emitted, retains
**83–86% of H(teacher)**. Implemented as `--kd span`; the sealed design remains the default `--kd anchor`.

> **DECISION RESOLVED BEFORE LAUNCH (Architect, 2026-07-13 — sealed in `docs/PHASE64_DECISIONS.md` §3, pushed in the
> same pre-launch session, nothing executed): `--kd span`** (chain-rule-factorized), with `anchor` recorded as the
> measured-degenerate case. Rationale: `anchor` is *known* to deliver only ~15% of the teacher's signal, and a gate
> that fails for a reason already measured teaches nothing. **D3 therefore runs as `span` vs `ce`.** The gate itself
> (cross-tokenizer KD must beat plain CE) is UNCHANGED — this fixes the arm, it does not loosen the bar.

## 4. Declared choices (they are the mechanism on trial, not hidden costs)

- **KD target**: teacher top-K = **32**, probs quantized to **uint8** (~160 B/teacher-token). Rows are indexed by
  **teacher token, not by student anchor** — anchors are a function of V (still an open A/B at rung-1), teacher
  tokens are not. This keeps V open *and* is cheaper (teacher tokens are 0.5× student tokens on this corpus).
- **α = 0.5**: at a KD position, `loss = (1-α)·CE + α·KD`; CE elsewhere. Under `--kd span` the KD touches ~97% of
  positions (vs ~45% under `anchor`), so α is doing much more work — **α is a free parameter and is frozen here at
  0.5 by declaration, not by evidence.**
- **t2s / decomp approximation**: a teacher token's student decomposition is computed on its bytes **in isolation**.
  The ground-truth student token always comes from the real corpus stream, so only the K−1 counterfactual
  alternatives carry this approximation.
- **InfoNCE positives are MINED, not labelled** (TinyStories has no MQAR ground truth): the positive for query *t*
  is the most recent past position that repeats the current bigram (same current token AND same next token). This
  is a declared rule and is itself what the D4-clause-2 read puts on trial.
- **MoE upcycle is magnitude-matched**: the seeded expert `down` weights are scaled by *k*. Without this the MoE
  emits ~1/k of the dense output it was seeded from and the curriculum takes a gratuitous hit at the switch
  (measured: **+0.49 BPB → +0.0056 BPB** at the transition).
- **Chunk-local sampling**: batches are drawn from the student range covered by the resident logit chunks (2 at a
  time, advancing, delete-behind). This is the production generate→train→delete pipeline exercised, not simulated —
  and it means sampling is chunk-local rather than globally i.i.d.
- **fp16 + loss scaling** (T4 has no bf16). **AdamW-8bit** through a single optimizer factory, so the stage
  transitions cannot silently drop back to fp32 states.

## 5. Commands (the exact registered run)

```bash
# (A) data + tokenizer + anchors + decomp        [CPU, ~1h on the full 64 MiB]
.venv/Scripts/python.exe benchmarks/phase64/mve/mve_data.py

# (B) teacher logits, chunked                    [3060, ~1.9h @ 2283 tok/s; vLLM on Linux/Kaggle]
.venv/Scripts/python.exe benchmarks/phase64/mve/mve_logits.py --tag full --backend hf --quant fp16 \
    --ctx 2048 --stride 1536 --batch 4

# (C-F) the curriculum: the D3 arms (span vs ce, resolved above), then the D4-clause-2 arms
.venv/Scripts/python.exe benchmarks/phase64/mve/mve_train.py --tag full --arm kd --kd span --recall on  --fp16 --steps 20000
.venv/Scripts/python.exe benchmarks/phase64/mve/mve_train.py --tag full --arm ce                --recall on  --fp16 --steps 20000
.venv/Scripts/python.exe benchmarks/phase64/mve/mve_train.py --tag full --arm kd --kd span --recall off --fp16 --steps 20000

# DDP validation (2xT4, Linux): loss-curve parity vs single-GPU + scaling >= 1.6x
torchrun --nproc_per_node=2 benchmarks/phase64/mve/mve_train.py --tag full --arm kd --fp16 --steps 2000 --stages C
```

**Execution note (session-management flags — do NOT change the experiment).** The three arms run on Kaggle (3 accounts,
one arm each, in parallel) with `--data-dir /kaggle/input/... --ckpt-dir /kaggle/working --resume --time-budget-min 660`
and, on the ≤16 GB T4, `--batch 8 --accum 2`. These are infrastructure only: `--resume`/`--time-budget-min` implement
gate #2 (survive the 12h session cap; the run stops cleanly at `MVE-INCOMPLETE` and continues on re-launch of the
identical cell), and `--batch 8 --accum 2` keeps the **effective batch = 16** (identical optimization target to
`--batch 16`). The arms, the loss (α=0.5, KD/CE, reverse-KL at F), the C→D→E→F curriculum, and every gate above are
unchanged. The Kaggle run-pack that wires these (dataset copies + notebook cells) is a local staging artifact, not
committed.

## 6. Known blockers handed back to the owner

- **The Stack v2 content-path smoke cannot be run by the Builder**: the dataset is gated *to the card* (HTTP 401 on
  the README). It needs an **HF token with the terms accepted** on the owner's account. Until then the brief's
  `content`-column assumption stays **unverified** — and it is load-bearing for every ingestion script.
- **torchrun cannot disable libuv on Windows** → the local DDP check runs through a manual `env://` rendezvous. The
  real DDP smoke belongs on the 2×T4 (Linux), where torchrun is fine.
- **Licences, live (2026-07-13):** Qwen2.5-Coder **1.5B = apache-2.0** ✓ (sealed teacher clean) · **3B = "other /
  qwen-research"** ✓ (the upgrade path *is* restricted, as the plan suspected) · **7B = apache-2.0** (new: the
  licence-clean upgrade is the **7B**, not the 3B) · **gemma-4-12b exists and is apache-2.0** (brief's claim
  confirmed; decision-moot) · gemma-2/3 are under the "gemma" licence and manually gated.
