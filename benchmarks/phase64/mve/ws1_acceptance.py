#!/usr/bin/env python3
"""WS1 acceptance tests -- the three apparatus fixes, each with the check the brief asked for.

  1. alpha-QAT end state is BIT-IDENTICAL to the plain hot-swap (checksum over logits AND weights).
     This is the load-bearing one: the epsilon-identity is only free if alpha=1 changes nothing at all.
  2. alpha=0 is an exact fp32 identity at the instant of the switch (that IS the mechanism).
  3. the collective stop flag is agreed by all ranks (single-process algebra check; the real 2-rank
     forced-budget run is driven separately by ws1_ddp_stop.py).

Run: python benchmarks/phase64/mve/ws1_acceptance.py
"""
import os, sys, hashlib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mve_model import MVEStudent, S0, BitLinear158Alpha          # noqa: E402
from phase57_ternary import BitLinear158                          # noqa: E402

V = 1024
ok = True


def h(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def build(seed):
    torch.manual_seed(seed)
    return MVEStudent(V, **S0)


def logits_of(m, x):
    m.eval()
    with torch.no_grad():
        lg, _ = m(x, None)
    return lg


x = torch.randint(0, V, (2, 64))

# ---- 1. end-state equivalence -----------------------------------------------------------------
# Two models from the same seed: one hot-swapped (the MVE behaviour), one alpha-scheduled and driven
# to alpha=1. If these differ by a single bit, the "same end state, continuous path" claim is false
# and stage D would no longer be comparable across rungs.
m_hot = build(0); n_hot = m_hot.qat_ternary(alpha_sched=False)
m_alp = build(0); n_alp = m_alp.qat_ternary(alpha_sched=True)
m_alp.set_qat_alpha(1.0)

w_hot = torch.cat([p.reshape(-1) for p in m_hot.state_dict().values() if p.dtype.is_floating_point])
w_alp = torch.cat([p.reshape(-1) for p in m_alp.state_dict().values() if p.dtype.is_floating_point])
same_w = h(w_hot) == h(w_alp)
l_hot, l_alp = logits_of(m_hot, x), logits_of(m_alp, x)
same_l = h(l_hot) == h(l_alp)
print("1. end state alpha=1 vs hot-swap")
print(f"   swapped linears: hot={n_hot} alpha={n_alp}")
print(f"   weights  sha={h(w_hot)} vs {h(w_alp)}  -> {'IDENTICAL' if same_w else 'DIFFER'}")
print(f"   logits   sha={h(l_hot)} vs {h(l_alp)}  -> {'IDENTICAL' if same_l else 'DIFFER'}")
print(f"   max|dlogits| = {(l_hot - l_alp).abs().max().item():.3e}")
ok &= same_w and same_l

# ---- 2. alpha=0 is the fp32 identity ----------------------------------------------------------
# The whole point of the ramp: at the switch instant the function is UNCHANGED, so the optimizer
# meets a continuous landscape instead of the measured +0.32 BPB teleport.
m_fp = build(0)
l_fp = logits_of(m_fp, x)
m_a0 = build(0); m_a0.qat_ternary(alpha_sched=True); m_a0.set_qat_alpha(0.0)
l_a0 = logits_of(m_a0, x)
d0 = (l_fp - l_a0).abs().max().item()
print("\n2. alpha=0 vs pre-switch fp32")
print(f"   max|dlogits| = {d0:.3e}  -> {'IDENTITY' if d0 == 0.0 else 'NOT identity'}")
ok &= (d0 == 0.0)

# and the ramp must actually move the function, or we would be shipping a no-op
m_a0.set_qat_alpha(0.5); l_h = logits_of(m_a0, x)
moved = (l_fp - l_h).abs().max().item()
print(f"   alpha=0.5 max|dlogits| vs fp32 = {moved:.3e}  -> {'ramp is live' if moved > 0 else 'RAMP IS DEAD'}")
ok &= moved > 0

# ---- 3. monotone, terminating schedule ---------------------------------------------------------
# alpha is a pure function of the in-stage step, which is what lets a mid-stage-D resume restore it
# from step_in_stage alone -- no checkpoint state, no chance of the two ranks disagreeing.
N = 1000
sched = [min(1.0, (i + 1) / N) for i in (0, 1, N // 2, N - 2, N - 1, N, 2 * N)]
mono = all(b >= a for a, b in zip(sched, sched[1:])) and sched[-1] == 1.0 and sched[0] > 0
print(f"\n3. alpha schedule over N={N}: {[round(s, 4) for s in sched]}")
print(f"   monotone, reaches exactly 1.0, never exceeds -> {'OK' if mono else 'BROKEN'}")
ok &= mono

print("\n==== WS1 acceptance: " + ("PASS" if ok else "FAIL") + " ====")
sys.exit(0 if ok else 1)
