import math
P=print
P("=== RETRACTION CHECK: where did my 413 h come from? ===")
P("I calibrated on the dossier's OWN implied throughput: 1.7e12 FLOP / 26 s = 65 GFLOP/s.")
P("T4 peaks: fp16 tensor 65 TFLOPS, fp32 8.1 TFLOPS.")
P(f"  65 GFLOP/s = {65e9/65e12*100:.2f}% of fp16 peak, {65e9/8.1e12*100:.2f}% of fp32 peak.")
P("  => I adopted a throughput ~100x below anything realistic, FROM the document I was auditing.")

P("\n=== Total work, my convention (7 matrices x 80 layers x T_outer=50) ===")
def cost_layer(M,N,T=50):
    return N**3/3 + 2*M*N*N + 2*(N*N)*M*T
d,ffn=8192,28672
mats=[("q",d,d),("k",1024,d),("v",1024,d),("o",d,d),("gate",ffn,d),("up",ffn,d),("down",d,ffn)]
per=sum(cost_layer(M,N) for _,M,N in mats)
tot=per*80
P(f"  per layer {per:.2e} ; x80 = {tot:.2e} FLOPs")
for r,lab in [(8.1e12,"fp32 peak"),(3e12,"realistic fp32/mixed"),(1e12,"pessimistic trsm-bound"),(65e9,"the rate I used")]:
    P(f"   at {lab:24s} {r:8.2e} FLOP/s -> {tot/r/3600:8.1f} h")

P("\n=== ANCHOR 1: SparseGPT published wall-clock ===")
P("  ~4 h for 175B on ONE A100-80GB, 128x2048 calibration, SINGLE pass. [A, sec.4]")
base=4.0*(100/175)
P(f"  scale to 100B params: {base:.2f} h on A100 (single pass)")
for lo,hi,lab in [(2.4,4.8,"A100->T4 (fp32 2.4x .. fp16 4.8x)")]:
    P(f"  {lab}: single pass on T4 = [{base*lo:.1f}, {base*hi:.1f}] h")
P("  ADMM iteration vs SparseGPT pass: BOTH O(M N^2) once L (or the inverse) is cached.")
P("  Published admm-pruning (2401.02938) uses 20 iters tuned to MATCH SparseGPT's overhead [T].")
P(f"  => 50 iters ~ 2.5x a SparseGPT pass -> [{base*2.4*2.5:.0f}, {base*4.8*2.5:.0f}] h on a T4")

P("\n=== ANCHOR 2: ADMM-Q measured overhead vs single-pass GPTQ ===")
adm,gptq=117.73,33.97
P(f"  ADMM-Q 32B = {adm} min vs GPTQ 32B = {gptq} min -> {adm/gptq:.2f}x a single-pass method [T]")
P(f"  100B single pass on A100 {base:.2f} h -> ADMM {base*adm/gptq:.1f} h on A100")
P(f"  -> on a T4: [{base*adm/gptq*2.4:.0f}, {base*adm/gptq*4.8:.0f}] h")

P("\n=== VERDICT ===")
P("  Three independent anchors bracket roughly 15-40 h.")
P("  The dossier says 14 h. It is at the OPTIMISTIC EDGE but it is NOT refuted.")
P("  MY '413 h / ~30x short' IS WRONG AND IS RETRACTED.")
P("\n  What still stands: M=N=4096,L=80 IS 9.4B not 100B (10.6x), and Table 3 costs")
P("  1 matrix per layer not 7. Those are 173x in WORK. The time estimate survives only")
P("  because it also understates THROUGHPUT by ~100x. Two errors cancelling is not a")
P("  cost model -- but the headline number is approximately right, and I said it was not.")
