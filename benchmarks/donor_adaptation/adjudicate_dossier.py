import math
def P(*a): print(*a)

P("== 1. Is M=N=4096, L=80 a 100B model? ==")
# dossier: M=N=4096 "typical SwiGLU intermediate for a 100B donor", L=80
d=4096; ffn=4096; L=80
attn = 4*d*d
swiglu = 3*d*ffn
P(f"  per-layer attn={attn/1e6:.1f}M swiglu={swiglu/1e6:.1f}M tot={(attn+swiglu)/1e6:.1f}M")
P(f"  x L=80 -> {(attn+swiglu)*L/1e9:.2f} B params   <-- claimed 100B")
P(f"  shortfall factor: {100e9/((attn+swiglu)*L):.1f}x")

P("\n== 2. Real 100B-class geometry (Llama-3-70B-like scaled) ==")
for name,d,ffn,L in [("Llama-3-70B",8192,28672,80),("Qwen2.5-72B",8192,29568,80),("100B target",9216,32768,88)]:
    n=(4*d*d+3*d*ffn)*L
    P(f"  {name:14s} d={d} ffn={ffn} L={L} -> {n/1e9:.1f}B")

P("\n== 3. Solver cost at dossier dims vs real dims ==")
def cost_layer(M,N,T=50):
    chol = N**3/3
    WH   = 2*M*N*N
    backsub = 2*(N*N)*M*T      # 2 triangular solves, N^2 each, M rows, T iters
    return chol, WH, backsub, chol+WH+backsub
c=cost_layer(4096,4096)
P(f"  dossier  M=N=4096: chol={c[0]:.2e} WH={c[1]:.2e} backsub={c[2]:.2e} tot={c[3]:.2e}")
# real: 7 matrices per layer with their own (M,N)
d,ffn=8192,28672
mats=[("q",d,d),("k",1024,d),("v",1024,d),("o",d,d),("gate",ffn,d),("up",ffn,d),("down",d,ffn)]
tot=0
for nm,M,N in mats:
    cc=cost_layer(M,N)
    tot+=cc[3]
    P(f"    {nm:5s} M={M:6d} N={N:6d} -> {cc[3]:.2e}")
P(f"  real per-layer (7 matrices): {tot:.2e}  ratio vs dossier per-layer: {tot/c[3]:.1f}x")

P("\n== 4. Wall-clock, anchored on the dossier's OWN measured throughput ==")
# dossier: 1.7e12 FLOPs backsub -> 26 s  => 65 GFLOP/s effective for triangular solves
eff_tri = 1.7e12/26
eff_gemm = 7e10/1.1   # their Hessian/WH rate
P(f"  dossier implied rates: triangular {eff_tri/1e9:.0f} GFLOP/s, gemm {eff_gemm/1e12:.1f} TFLOP/s")
t=0
for nm,M,N in mats:
    chol,WH,bs,_=cost_layer(M,N)
    t += chol/eff_gemm + WH/eff_gemm + bs/eff_tri
P(f"  real per-layer time: {t:.0f} s = {t/60:.1f} min")
P(f"  x L=80: {t*80/3600:.0f} h   <-- dossier claims 14 h")
P(f"  weeks of the 90 h/week quota: {t*80/3600/90:.1f}")

P("\n== 5. Table 1 'skippable fraction ~0.001' for unstructured, block-64 iid ==")
for s in [0.50,0.80,0.90,0.95]:
    P(f"   s={s}: P(all 64 zero)= {s**64:.3e}   (dossier says 0.001 for ALL s)")
P("   at B_block=22:")
for s in [0.50,0.80,0.90,0.95]:
    P(f"   s={s}: {s**22:.3e}")

P("\n== 6. Theorem 4.4 sign: E <= (1-s)/lam * Tr(WHW') ==")
P("   s = FRACTION ZEROED (their own def). s=0 -> E = full energy (max). s=1 -> E=0.")
P("   => bound says pruning EVERYTHING is lossless and pruning NOTHING is maximal. INVERTED.")
P("   Their text: 'doubling the sparsity halves the reconstruction error'. Same inversion.")

P("\n== 7. Theorem 4.5 internal consistency ==")
P(f"   unstructured at s=0.5 per (21): (1-0.5)/lam = 0.5/lam")
P(f"   2:4 per (22): 2/lam  ->  ratio = {2/0.5}x, but text+Table1 claim 2.0x. INCONSISTENT.")

P("\n== 8. Theorem 4.6 factor vs Table 1 ==")
B=22; eta=0.2
f=(1+math.log(B)/B)*(1+eta/(1-eta))
P(f"   (1+lnB/B)(1+eta/(1-eta)) = ({1+math.log(B)/B:.4f})({1+eta/(1-eta):.4f}) = {f:.3f}x")
P(f"   log2 variant: {(1+math.log2(B)/B)*(1.25):.3f}x")
P("   Table 1 claims 1.6 / 1.6 / 2.4 / 3.8 and s-DEPENDENT.")
P("   Formula (23) multiplicative factor has NO s-dependence. Table not derivable from Theorem.")
P(f"   coarsening term lnB/B at B=1 (unstructured) = {math.log(1)/1:.3f} -> 0 penalty (ok)")
P(f"   at B->inf -> 0 penalty too. Non-monotone, peaks at B=e. Not a coarsening penalty.")

P("\n== 9. Theorem 5.2 arithmetic ==")
sp=0.01; al=0.005; L=80
P(f"   sqrt(L)*sigma_probe = {math.sqrt(L)*sp:.4f}  (dossier 0.09 OK)")
P(f"   alpha*L^2*sigma^2   = {al*L*L*sp*sp:.6f}  (dossier 0.32 -> off by {0.32/(al*L*L*sp*sp):.0f}x)")
P(f"   corrected total = {math.sqrt(L)*sp + al*L*L*sp*sp + 0.04:.3f} vs dossier 0.45")

P("\n== 10. Theorem 5.1 operator ==")
P("   (26): delta_{l+1} = (I+J_l) delta_l + ...   assumption stated: ||J_l|| <= 1-beta")
P("   => ||I+J_l|| <= 2-beta  => growth up to (2-beta)^L, EXPONENTIAL, not linear.")
P("   Bound (27) requires ||I+J_l||<=1-beta i.e. a CONTRACTING residual stream.")
P("   Assumption is placed on the wrong operator; (27) does not follow.")
P(f"   Also (1-(1-b)^L)/b <= 1/b (bounded in L). 'linear in L' only as b->0.")

P("\n== 11. rho-law B_block transplanted from OUR D=1536 to donor dims ==")
P("   bytes/neuron = 3 organs * D * 0.5 B/weight = 1.5*D")
for D in [1536,4096,8192,28672]:
    bpn=1.5*D
    P(f"   D={D:6d}: {bpn:8.0f} B/neuron -> neurons per 48KB = {49152/bpn:6.2f}")
P("   dossier hardcodes B_block>=22 (our D=1536 value) into a 100B analysis.")

P("\n== 12. Sec 6.9 compression ratio ==")
d=4096;L=80
fp16 = 2*d*d*L
tern = 0.5*d*d*L
P(f"   fp16 {fp16/1e9:.2f} GB ; ternary(0.5 B/w) {tern/1e6:.0f} MB  (dossier says 670 MB, ok)")
P(f"   ratio = {fp16/tern:.1f}x   <-- dossier claims 150x")
P(f"   even with s=0.80 storing only active blocks: {fp16/(tern*0.2):.0f}x")

P("\n== 13. Activation storage: 5.6 vs 6.1 ==")
P(f"   sec 6.1: 2*N*4096 B, N=4096 -> {2*4096*4096/1e6:.1f} MB per layer")
P("   sec 5.6: 'each ~2 GB in FP16' per layer -> 60x contradiction in the same document")

P("\n== 14. 100B forward pass on ONE T4 in 30 min? ==")
toks = 4096*2048
fl = 2*100e9*toks
P(f"   calib 4096 seq x 2048 tok = {toks/1e6:.1f}M tokens ; FLOPs = {fl:.2e}")
P(f"   T4 peak fp16 65 TFLOPS -> {fl/65e12/3600:.1f} h at 100% MFU (dossier: 0.5 h)")
P(f"   at a realistic 30% MFU -> {fl/(65e12*0.3)/3600:.1f} h")
P(f"   and 100B fp16 = 200 GB of weights vs 16 GB VRAM -> offload streaming, not modelled")

P("\n== 15. K_active consistency ==")
P("   6.4: K_active = K/2 = 93  (that is s=0.5)")
P(f"   Table 3 header: s=0.80 -> K_active = 0.2*186 = {0.2*186:.0f}. INCONSISTENT.")
