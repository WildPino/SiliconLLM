# Actual native head-domain moments and CPU cost: frozen preflight

12 October 2026. All existing 4422 FIT/4386 DEV raw source labels, full65537
vocabulary. No fitting/optimization/source forward/GPU/T4/RESERVED/longalign.
Uncertainty: can exact sufficient teacher data be compiled reproducibly, and what
is the CPU cost of a full loss/gradient evaluation before selecting a search?
Prior: full independently qualified QR gauge/native scalar head export. New
head/image is actual exported F32 C/beta/epsilon promoted to F64, not old paired255.

## Arithmetic and custody

d=256; e is positive exported F32 epsilon; s=sqrt(d e).
A=[C,beta/s] in F64. Source q uses actual BF16 head/final gamma/source multiplier
decoded F64 and raw source final h with original epsilon1e-5. Compile exact-real
categorical moments m=A^T q, H=-sum q logq, and teacher ID. No truncated vocabulary.
Current source-injected F32 t promoted F64 gives r=sqrt(||t||²/d+e),
v=(t/r,s/r), ||v||²=d,v_last>0. No altered code or optimized native upper.
Full initial L=logsumexp(A v)-m^T v-H and g=A^T softmax(A v)-m stored for all8808.
Support lower L-g^T v-16 sqrt(||g_first256||²+min(g_last,0)²) is a real-arithmetic
lower on relaxed half-ball and therefore actual hemisphere loss. Negative values
are retained; a numerically positive value is not an interval-certified floor.

Guard sphere defect<=1e-12; compare real F64/promoted head KL to already audited
native F32 warm-code KL, max absolute delta<=1e-5. This guard detects data/head
drift; it does not assert F32 probabilities exactly match real arithmetic.
First FIT row sin(i+1)/norm directional derivative at2e-6 and2e-7: relative error
abs(finite-analytic)/max(1,abs(finite),abs(analytic))<=2e-6 for both fixed steps.
No algorithm chosen from DEV; split identifiers/weights retained unchanged.

## Cost and caps fixed before observation

CPU six threads and logical affinity0..5; batch64. Two warmups and five measured
identical first FIT batch objective/gradient calls, all outputs exactly repeat.
Store all raw durations/median; median*139 is only a nominal one-pass workload
estimate, not optimizer cost or clean throughput. External known publishers
remain; timing admission=false. Source reading/moment compilation cost separate.
Runtime<=900s, OS peak<=4GiB, outputs<=96MiB. Full before/after input hashes,
fresh output namespace, result and terminal/PID/creation recorded on failure too.
No replay into partially written namespace. Independent audit must verify every
moment/entropy/ID/code/objective/gradient/support lower with different sums and
actual source reading; not a sampled audit. No optimization admission until PASS.

Decision after audit: use measured cost to freeze bounded search separately, or
change device/data scope if required. Optimizer and actual F32 upper replay remain
unimplemented. Better oracle codes still require causal compact SSM/SWA/ternary
generation, useful conditional capacity, CPU routing/DRAM and same-artifact50.
