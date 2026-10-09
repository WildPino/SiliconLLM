# Original tensor learner: prospective bridge verification

9 October 2026. Goal ACTIVE/INCOMPLETE. This is a prerequisite to the selected
source-informed whole-history pilot, not that pilot. New variables: direct
real-master tensor export and a trainable original-operator geometry with AQ63.
The original phase59 learner omitted deployed activation quantization. Its
training behavior therefore cannot certify a new original-engine conversion.

## Construction and declared approximations

Adopt all real phase57 master tensors from `moe_gran.pt`, with explicit complete
key/shape coverage. No optimizer/RNG/training history from this old checkpoint is
claimed. Original D256/L6/5 Mamba1 SSM+1 SWA/8 heads/window128, dReLU/E32/H128/k8,
native full softmax then stable lower-ID tie selection then selected mass
renormalization. Ternary scale is detached last-axis absolute mean clamped1e-5;
ties-even rounding. AQ63 uses detached maximum, reciprocal multiplication and
zero-vector zero scale. Gate/up integer sums -> activation scale -> weight scale;
relu(g)*relu(u)*selected mass -> AQ63 -> down integer sum/scales; fixed rank-order
addition. Integer sums are exactly representable in F32 at these dimensions.

STE is identity through effective ternary masters and quantized activations;
quantizer scales have no gradient. Differentiable linear surrogate is replaced
in forward by the native-ordered integer expression. This is a gradient choice,
not the derivative of discrete rounding. Router topology is discrete; gradients
flow through selected normalized masses. No dither or auxiliary load loss.

Torch RMS/matrix reductions, exp/softplus and scan multiply/add differ from
original AVX FMA/libm/fast exp. Algebraic operator matching is not bit parity.
These differences must be observed before native learner qualification.
Banks are CPU masters, parameters copied one selected expert at a time to the
activation device; no all-expert GPU quantization or E*maximum-token-count pad.
Backward can retain the touched union; this stage runs CPU only, E32/16 IDs.
No claim of bounded GPU residency at n1152 or of T4 feasibility follows.

Producer writes arbitrary supported runtime E/V into E4BPv001, streaming one
field at a time. It takes trainable master tensors, never legacy dequantized
expert references. Same field order permits exact file comparison with the
existing E32 packed export, which is reused without another old legacy run.

## Frozen observations and gates

1. Direct master export must match the already verified original E32 packed
   file bit for bit. No teacher or previous native outputs are regenerated.
2. Four predetermined independent bank inputs: all zero; `(i%127-63)/63`;
   `((7*i+11)%127-63)/31`; deterministic seed1709 Gaussian. All6 banks, all256
   outputs, actual selected IDs/masses are compared with calls to the frozen
   original `mlp_moe` body. Require exact IDs, max mass difference<=1e-6 and each
   output row RMS difference / max(reference RMS,1e-8)<=1e-4. Zero must be exact.
   Separately zero all router coefficients in a temporary bank and require
   tied selection IDs0..7 and mass1/8; no native file is exported for this control.
3. Whole learner forward on16 original input IDs reused from the packed stage's
   immutable64-input packet. Compare ALL16*1024 logits/16*6 actual routes against
   corresponding immutable native output prefix. Exact IDs, mass tolerance1e-6,
   each-row normalized RMS<=1e-4 and greedy agreement are required. Save actual
   arrays and all per-row measurements. This is a new learner observation,
   not a repeat of the original loader/storage comparison.
4. Only if2+3 pass: one NEW next-token CE update using complete16-ID history and
   shifted first17 original validation IDs. Adam lr1e-5/betas(.9,.999)/eps1e-8/
   weight_decay0/foreachFalse; clip global L2 norm1; CPU F32, deterministic seed1709,
   six CPU threads, no AMP/TF32. All parameter gradients and new Adam moments
   finite; aggregate nonzero gradient per organ/bank/router group. Save actual
   model/Adam/RNG/exposure/loss checkpoint immediately after this one update.
   Export changed tensors and run a NEW16-ID packed forward; apply same complete
   learner/native output/route gates. Native integer witnesses must equal
   independent CPU int64 dot products of newly quantized real masters.

No fallback loosens gates or performs learning after a failed bridge. A failed
gate records all completed measurements and stops before step4. A first actual
worker fault retains checkpoint if any and first-failure metadata; no replay of
completed native observations. Failure is useful diagnosis, not quality failure
of Falcon or a proof that the original geometry cannot hold it.

## Budget and scope

One CPU worker; family240s with final reserve20s; held worker plus largest direct
child plus launcher<=3GiB; namespace<=1GiB/log4MiB. Before launch freeze source,
producer/learner/probe, this protocol, reused packed stage inputs/results/body
hashes/executable, real checkpoint/corpus and runtime/compiler bytes. Worker
holds direct compiler/native child handles through exit; nested linker memory
not separately held. No source queries/GPU/T4/RESERVED. No throughput claim.
New training is at most1 update on original small-vocabulary model; this does
not substitute for source-informed n1152/full-Falcon-V/longest-FIT adaptation.

If bridge fails, localize numerical discontinuity under same original operators
before donor initialization. If passes, source-derived banks/core/readout and
full-history residency are the next decision; no additional E32 dose sweep.
