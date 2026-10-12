# From a readout oracle to a causal pretrained-to-engine pipeline

12 October2026. Prospective transfer boundary,not an implemented student/training
run. Current all-label image diagnostic is still running;no final result admitted.
This record prevents treating independent per-label codes as an autonomous model.

## What a successful image diagnostic would supply

Fixed native head permits full categorical supervision through qualified source
moments: L(v_psi(history))=logZ(A v)-m^T v-H(q),
`grad_psi L = J_v_psi^T (A^T p-m)` in differentiable real arithmetic.
This trains a history-to-code function against probabilities,not winner IDs.
Native code normalization/lift and quantization need their own derivative or
approximation contract;the identity does not validate an arbitrary STE.
Head changes invalidate the old moments and require recompilation.

Useful per-label native oracle codes can become auxiliary targets,with the full
categorical loss remaining authoritative. A numerical real-domain obstruction
instead requires a changed reader/conditional decoder before costly core training.
An unresolved relaxed gap supplies neither conclusion. No stored target dictionary
substitutes for causal acquisition of codes from arbitrary user token histories.

## The output code is not the whole recurrent state

The actual target state includes SSM states,convolution buffers,SWA/KV memory,
position and any routing state. A256-dimensional emission code does not bound
all that memory. Histories with similar current output distributions may require
different future behavior. Current-code regression alone therefore does not prove
that a compact core retains the necessary information.

For a candidate state encoder S and update F,causal closure requires
`S(history+token) = F(S(history), token)` under its explicit approximation budget,
and the readout must recover the donor distribution from S. Validate these
separately and in composition;local source-operator equality is insufficient.

Exact state collisions have an information test:if two histories produce identical
ENTIRE target state and receive the same future continuation,the deterministic
engine predicts the same distribution. For differing donor distributions q1,q2,
the best common prediction has mean KL at least JSD(q1,q2),since
`min_p .5 KL(q1||p)+.5 KL(q2||p) = JSD(q1,q2)` at p=(q1+q2)/2.
This concerns demonstrated full-state collisions,not close emission vectors.
Approximate distances alone do not justify a collision bound,especially with
discrete expert routing/LUT thresholds. Router IDs and masses must be included.

## Next causal implementation choice after the current gate

Reuse actual source tokenizer/chat template,raw causal histories and native head
contract. Reuse trained source S6 generator/projection tools as initializers only
where their required feature/state widths fit the target;their costly exact lifts
and failed fixed384-channel assays remain limits,not selected cheap endpoints.
Any new compact core must generate its own states/codes and handle its own emitted
assistant history. Keep original copied-source baselines and old0/16 results.

Before training,freeze compact SSM/SWA state/active widths,trainable shared versus
conditional functions,ternary/precision transport,physical expert layout and CPU
structured IDs+masses. Verify a short causal forward/backward/native equivalence
with meaningful operator controls before a long adaptation run. Dense master/
gradient O(n) remains unresolved;do not assume RAM-driven100B conversion from
current small-array results. Announce T4 reason/budget/stops before allocating it.

Source-distillation targets may use existing FIT data;held-out learned-model and
own-history quality require sealed evaluations. Reserved/longalign remain unopened
until an explicit evaluation decision. First useful end-to-end case,then additional
families/scales;quality and>=50 useful accepted batch1 tokens/s must hold on the
same artifact with actual DRAM. Current head oracle/GPU timings prove none of these.
