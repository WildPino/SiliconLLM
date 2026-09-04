"""t2_r4_controls.py -- is R4 (GPTQ) WRONG, or is it RIGHT and the objective wrong?

T2's full run measured R4 at +3.532 BPB: worse than R0 (+3.309) and worse than arm Z, which
is R0's codes with RANDOM SIGNS (+3.373). A correct GPTQ cannot be worse than the
round-to-nearest it starts from on its own objective, so before that number is reported as a
finding it has to be shown that the instrument is sound.

  CONTROL A  With H = c*I the inverse-Hessian coupling is zero and GPTQ MUST collapse onto R0
             exactly -- same codes, same scales. Run at two different c.
  CONTROL B  On a correlated H, GPTQ MUST lower its own objective ||(W - Wq)X||^2 relative to
             R0. This is the load-bearing one: A alone would pass for a GPTQ that does nothing.
  MECHANISM  Then measure what it costs per weight to buy that, on a 3-level grid.

Run:  python t2_r4_controls.py    (from benchmarks/donor_adaptation/ternary)
"""
import os, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 't2_rules.py'), encoding='utf-8').read()
src = src.split('# ===================================================================== calibration capture')[0]
ns = {'__file__': os.path.join(HERE, 't2_rules.py'), '__name__': 'not_main'}
exec(compile(src, 't2_rules.py', 'exec'), ns)
r0, r4 = ns['r0_bitlinear'], ns['r4_gptq']

torch.manual_seed(0)
n_out, n_in = 256, 512
W = torch.randn(n_out, n_in) * 0.02

print('CONTROL A -- H = c*I must make GPTQ collapse onto R0 exactly')
for c in (1.0, 7.3):
    q0, a0 = r0(W)
    q4, a4 = r4(W, torch.eye(n_in, dtype=torch.float64) * c)
    print('   c=%-5s codes identical: %-5s   scales identical: %s'
          % (c, bool((q0 == q4).all()), bool(torch.allclose(a0, a4))))

print()
print('CONTROL B -- on a CORRELATED H, GPTQ must lower its own objective ||(W-Wq)X||^2')
A = torch.randn(n_in, n_in) * 0.1 + torch.eye(n_in)
X = torch.randn(4096, n_in) @ A
X[:, :8] *= 12.0
H = (X.T @ X).double()
q0, a0 = r0(W)
q4, a4 = r4(W, H)

def loss(q, a):
    E = (W - q * a)
    return float((E @ X.T).pow(2).sum())

l0, l4 = loss(q0, a0), loss(q4, a4)
print('   R0 layer loss  %.6e' % l0)
print('   R4 layer loss  %.6e' % l4)
print('   GPTQ improves its own objective: %s  (ratio %.4f)' % (l4 < l0, l4 / l0))
print('   codes changed by GPTQ: %.2f%%' % (100.0 * float((q0 != q4).float().mean())))
print('   saturated |w/a|>1 fraction, R0: %.3f' % float((W.abs() / a0 > 1).float().mean()))

print()
print('MECHANISM -- can a 3-level grid absorb the error GPTQ pushes forward?')
print('   %-30s %10s %10s' % ('', 'R0', 'R4'))
print('   %-30s %10.4f %10.4f' % ('zero fraction',
      float((q0 == 0).float().mean()), float((q4 == 0).float().mean())))
print('   %-30s %10.4f %10.4f' % ('|code|==1 fraction',
      float((q0.abs() == 1).float().mean()), float((q4.abs() == 1).float().mean())))
r0r = (W - q0 * a0).abs() / a0
r4r = (W - q4 * a4).abs() / a4
print('   %-30s %10.4f %10.4f' % ('mean |residual| / row scale', float(r0r.mean()), float(r4r.mean())))
print('   %-30s %10.4f %10.4f' % ('frac residual > 0.5 scale', float((r0r > 0.5).float().mean()),
      float((r4r > 0.5).float().mean())))
print('   %-30s %10.4f %10.4f' % ('max |residual| / row scale', float(r0r.max()), float(r4r.max())))
