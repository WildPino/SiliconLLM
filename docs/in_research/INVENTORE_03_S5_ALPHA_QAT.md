# The Inventor — 03: S5 proposal — α-scheduled QAT (the ε-identity law applied to stage D)

**Status: PROPOSAL IN THE POCKET — conditional, NOT applied.** Trigger: the MVE's gate (iv) reads
"KD-then-QAT stability". If the C→D switch shows a transition shock, this is the fix already designed;
if it passes clean, this stays parked (no change to a working curriculum). Any application is a
**declared deviation at a rung boundary**, owner-approved — the MVE apparatus itself is never touched
mid-run.

## 1. The law (stated once, project-wide)

The curriculum has paid twice for the same missing principle and fixed it twice locally:
- **Upcycle (stage E1):** naive seeding emitted ~1/k of the dense output → +0.49 BPB shock; the
  magnitude-matched seeding (×k) reduced it to +0.0056 — banked as a law in the plan.
- **Recall insertion (stage E2):** the slot enters behind a zero-init gate → exact identity at insertion,
  by design.

General statement: **every curriculum switch must be an ε-identity of the function at switch time** —
the optimizer should meet a continuous loss landscape, never a teleport. Stage D (QAT hot-swap) is the
one remaining switch that violates it: `qat_ternary()` swaps fp linears to BitLinear158 in one step —
the forward jumps from `W·x` to `ternary(W)·x` instantly. The plan's own §8 lists "divergence at the QAT
switch" as a risk; this proposal dissolves it *by construction* instead of hoping.

## 2. The mechanism (~15 lines, zero new trainable params)

Interpolated ternarization: `W_eff(α) = (1−α)·W + α·ternary(W)`, with α scheduled 0→1 over N_α steps
after the switch. STE unchanged (backward = identity either way); α is a scheduled scalar, deterministic,
and at α=1 the model is bit-for-bit the current BitLinear158 — the end state is IDENTICAL, only the path
is continuous.

Implementation as a subclass in `mve_model.py` (shared `phase57_ternary.py` untouched):

```python
class BitLinear158Alpha(BitLinear158):
    """W_eff = (1-alpha)*W + alpha*(ternary(W)) - the epsilon-identity form of the D-switch.
    alpha=0: exact fp32 identity at the switch; alpha=1: exactly BitLinear158. STE unchanged."""
    alpha: float = 0.0
    def forward(s, x):
        w = s.weight
        scale = (w.abs().mean(dim=1, keepdim=True) if s.per_row else w.abs().mean()).clamp_min(1e-5)
        wq = (w / scale).round().clamp(-1, 1)
        w_ste = w + (s.alpha * (wq * scale - w)).detach()
        return F.linear(x, w_ste)
```

Trainer side (stage-D loop): `for m in model.modules(): if isinstance(m, BitLinear158Alpha):
m.alpha = min(1.0, step_in_D / N_alpha)` with N_α ≈ 1-2k steps (desk; A/B-able).

## 3. Falsifiable prediction (declared now, before any trigger)

With α-scheduling, the val-BPB trace across the C→D switch is continuous (no step-jump beyond seed
noise 0.005), and the post-switch recovery period disappears; final stage-D BPB ≥ the hot-swap arm's
(same steps). If the hot-swap arm shows no shock anyway (gate iv clean), the mechanism buys nothing —
that is the parked outcome and it is fine.

## 4. Costs & risks (honest)

Zero params, zero teacher cost, one scalar schedule. Risk: during 0<α<1 the weights are NOT ternary —
the QAT regularization pressure ramps in later; if N_α is too long, stage D shortens effectively.
Mitigation: N_α ≤ 10% of stage-D steps. Not compatible with claims that stage D was "pure QAT
throughout" — the deviation must be declared in the rung's record (standard discipline).
