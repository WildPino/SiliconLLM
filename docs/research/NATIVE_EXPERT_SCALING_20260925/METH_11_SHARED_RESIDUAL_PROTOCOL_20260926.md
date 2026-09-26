# METH-11: shared FFN plus routed residual, frozen before scoring

**Status:** prospective CPU-fp32 screen. This protocol makes no quality or speed claim.

## Question and decision

The E256/top-16 H1 carve trained on eight Qwen2.5-1.5B layers still trails its
intact donor by 0.155895 BPB after 765 steps. Does keeping a calibrated shared
FFN path make a *full 28-layer*, donor-initialized conditional transformation
close enough at step zero to justify joint training? The changed variable is a
shared path; donor attention, embeddings and head remain intact, and there is
no ternary or H0 low-rank approximation in this screen. This isolates the
conditional geometry from the two other H1 losses. The screen will decide
whether to design training and native export for this geometry, or reject it.

## Bound inputs and geometry

- Donor: `Qwen/Qwen2.5-1.5B` revision
  `8faed761d45a263340a0528343f099c05c9a4323`, safetensors SHA-256
  `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`.
- Per-layer E256 labels, 35 distinct donor FFN neurons/group, from
  `benchmarks/donor_adaptation/s1/_h1_bundle/labels_E256.npz`. Activation
  `rms_h` comes from the donor-only, disjoint calibration split in
  `_h1_bundle/h1_actstats.npz`.
- Per layer, rank each group by the sum over its neurons of
  `rms_h[i]^2 * ||down[:,i]||_2^2`. Tie goes to lower original group ID.
  The highest 64 groups are always active. The other 192 are distinct
  residual experts. Their router rows come from E37's fitted E256 router
  at `D:/_ktmp/e37/e37_routers_E256.npz` (SHA-256
  `42b12cb9d4bda2da7833cd0e179a9dab8e3f43264ce640e2b5db31d5fd043d8a`).
  Score the 192 residual rows and take the highest 16; tie goes to lower
  original group ID. Gate is exactly one for shared and selected residual
  groups, zero for all others. No output rescaling.
- The target is 64 shared ×35 + 16 routed ×35 = **2,800 active FFN
  neurons/layer**, 31.25% of the donor's 8,960. All 8,960 donor neurons
  remain stored and distinct. This is an exact source partition followed by
  an approximate selection. A new E=1,920 target at fixed group width would
  require *new* trained expert parameters; repeating donor groups would not
  count as 10× capacity.

## Fixed screen and stops

Use the **first two** windows of the previously frozen H1 heldout IDs,
512 tokens each, in CPU fp32. They are an internal pilot only: H1 has already
used this heldout, so a future success requires a fresh untouched split.
Score UTF-8 bytes with `density/common.py:get_slice` and the H1 BPB formula.
Run the intact donor and the shared+residual transform on exactly these two
windows. Also verify an all-groups-active control produces logits matching
the donor within 1e-4 absolute on the first 16 positions of window zero.
Abort if donor identity, data ID, group counts, finite logits or control fail.

**Before seeing pilot results:** if step-zero BPB exceeds donor by more than
0.20 BPB, stop this geometry before training. At or below +0.20, the next
cell may design a bounded joint-training run but cannot call the quality
gate passed. The final target remains donor-relative quality on untouched
data, useful generation and tasks, plus ≥50 accepted tok/s on the same
native artifact. No GPU work is authorized by this protocol itself.

Expected local cost: two donor/conditional 512-token fp32 passes plus a
16-token wiring check, roughly minutes on six CPU threads; stop on timeout
at 30 minutes or RAM pressure. The score is independent of the engine's
performance. A capacity projection is not a measured CPU LUT rate.

## Native and scaling prerequisites

The existing `benchmarks/donor_adaptation/engine/donor_engine.c` accepts one
carved FFN with E dividing F; it does **not** implement an always-active
shared path, and its Qwen embedding/head remains fp32. An export therefore
needs a new FFN kind, exact parity controls, low-bit head/attention support
and realistic native timing. `benchmarks/phase60/engine.c` has a ternary LUT
for its small SSM/SWA architecture, not a Qwen Q4_K kernel. No Q4_K or
50 tok/s capability may be inferred from this fp32 screen. The E×10 case
also requires measured CPU router+selection latency, RAM for stored distinct
experts, and *quality* with 10× more choices; constant top-k alone proves
none of these.
