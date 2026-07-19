#!/usr/bin/env python3
"""WS2 acceptance: the active-only MoE must be the SAME MODEL as compute-all, only cheaper.

Gates, in the order they matter:
  1. GRAD-EQUIVALENCE (non-negotiable). Same params, same input, fixed seed: forward outputs and the gradients
     w.r.t. every parameter must agree with the compute-all path to float tolerance. Not "close enough to train" --
     if this fails the sparse path is a different model and every number measured with it is incomparable to the MVE.
  2. DETERMINISM. Two runs at identical seed must be BIT-identical. Bit-identical reruns are a live diagnostic in
     this project (arms A and C agreeing to the last bit is how the apparatus gets checked), so an atomics-based
     scatter that silently costs reproducibility is not an acceptable implementation.
  3. MEMORY / THROUGHPUT, reported not asserted: peak activation memory and tok/s, so the brief's floor (3860 tok/s
     at the comparability config) can be read against a measured number.

Run: python benchmarks/phase64/mve/ws2_acceptance.py [--device cuda]
"""
import argparse, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mve_model import MVEStudent, S0, MOE, SparseMoEMLP           # noqa: E402
from phase59_moe import MoEMLP                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--tol", type=float, default=2e-4)
a = ap.parse_args()
dev = a.device
V = 1024
ok = True
print(f"device={dev}  E={MOE['E']} hid_e={MOE['hid_e']} k={MOE['k']}")


def build(sparse, seed=0):
    torch.manual_seed(seed)
    m = MVEStudent(V, **S0)
    m.qat_ternary()
    m.upcycle_moe(dev_type=("cuda" if dev.startswith("cuda") else "cpu"), seed=1234, sparse=sparse)
    return m.to(dev)


# ---- 1. grad-equivalence -----------------------------------------------------------------------
# Both models are built from the same seed and the same upcycle seed, so their parameters are identical
# tensors; any difference downstream is the dispatch and nothing else.
m_dense, m_sparse = build(False), build(True)
sd_d, sd_s = m_dense.state_dict(), m_sparse.state_dict()
same_params = all(torch.equal(sd_d[k], sd_s[k]) for k in sd_d)
print(f"\n1. grad-equivalence  (params identical before the test: {same_params})")
assert same_params, "parameter mismatch -- the comparison would be meaningless"
print(f"   moe class: dense={type(m_dense.blocks[0].mlp).__name__} sparse={type(m_sparse.blocks[0].mlp).__name__}")

torch.manual_seed(7)
x = torch.randint(0, V, (2, 128), device=dev)
y = torch.randint(0, V, (2, 128), device=dev)

outs = {}
for tag, m in (("dense", m_dense), ("sparse", m_sparse)):
    m.zero_grad(set_to_none=True)
    lg, aux = m(x, y)
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V).float(), y.reshape(-1)) + aux
    loss.backward()
    outs[tag] = (lg.detach().clone(), float(loss),
                 {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})

d_out = (outs["dense"][0] - outs["sparse"][0]).abs().max().item()
d_loss = abs(outs["dense"][1] - outs["sparse"][1])
print(f"   max|d logits| = {d_out:.3e}   |d loss| = {d_loss:.3e}")
gd, gs = outs["dense"][2], outs["sparse"][2]
worst, worst_n = 0.0, ""
for n in gd:
    if n not in gs: print(f"   MISSING GRAD in sparse: {n}"); ok = False; continue
    sc = max(gd[n].abs().max().item(), 1e-8)
    r = (gd[n] - gs[n]).abs().max().item() / sc
    if r > worst: worst, worst_n = r, n
print(f"   worst relative grad diff = {worst:.3e}  on {worst_n}  ({len(gd)} tensors)")
g1 = d_out < a.tol and d_loss < a.tol and worst < a.tol
print(f"   -> {'PASS' if g1 else 'FAIL'} (tol {a.tol})")
ok &= g1

# ---- 2. determinism ----------------------------------------------------------------------------
# The failure this guards against is silent: index_add_/scatter_add_ on CUDA accumulate via float atomics, so
# reruns differ in the last bits. Here the scatter-back is a fixed-axis sum over a stable permutation instead.
def run_once(seed):
    m = build(True)
    torch.manual_seed(seed)
    xx = torch.randint(0, V, (2, 128), device=dev)
    m.zero_grad(set_to_none=True)
    lg, aux = m(xx, xx)
    (lg.float().sum() + aux).backward()
    gcat = torch.cat([p.grad.reshape(-1) for _, p in sorted(m.named_parameters()) if p.grad is not None])
    return lg.detach(), gcat


l1, g1_ = run_once(11)
l2, g2_ = run_once(11)
bit_l = torch.equal(l1, l2); bit_g = torch.equal(g1_, g2_)
print(f"\n2. determinism (two runs, identical seed)")
print(f"   logits bit-identical: {bit_l}   grads bit-identical: {bit_g}")
if not bit_g:
    print(f"   max|dgrad| = {(g1_ - g2_).abs().max().item():.3e}")
ok &= bit_l and bit_g

# ---- 3. memory + throughput --------------------------------------------------------------------
print("\n3. memory / throughput  (reported, not asserted)")


def bench(m, B, T, iters=6):
    xx = torch.randint(0, V, (B, T), device=dev)
    if dev.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(2):                                            # warmup
        m.zero_grad(set_to_none=True)
        lg, aux = m(xx, xx); (lg.float().sum() + aux).backward()
    if dev.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        m.zero_grad(set_to_none=True)
        lg, aux = m(xx, xx); (lg.float().sum() + aux).backward()
    if dev.startswith("cuda"): torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**20) if dev.startswith("cuda") else float("nan")
    return B * T * iters / dt, peak


# Only ONE model may be resident while measuring: max_memory_allocated is process-wide, so holding both made the
# first pass report a peak that included the other model's parameters and activations. The first attempt printed an
# identical 5701.5 MiB for both paths -- the tell that the number was not measuring what it claimed to.
del m_dense, m_sparse, outs, gd, gs
if dev.startswith("cuda"): torch.cuda.empty_cache()

for label, B, T in (("comparability  b8 x T512", 8, 512), ("max-memory     b16 x T512 (record-only)", 16, 512)):
    row = []
    for tag, sp in (("compute-all", False), ("sparse", True)):
        m = build(sp)
        try:
            tps, peak = bench(m, B, T)
            row.append(f"{tag:11s}: {tps:7.0f} tok/s  peak {peak:8.1f} MiB")
        except RuntimeError as e:
            row.append(f"{tag:11s}: OOM ({str(e).splitlines()[0][:52]})")
        del m
        if dev.startswith("cuda"): torch.cuda.empty_cache()
    print(f"   {label}\n      " + "\n      ".join(row))

print("\n==== WS2 acceptance: " + ("PASS" if ok else "FAIL") + " ====")
sys.exit(0 if ok else 1)
