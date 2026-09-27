# METH-68: exact E128 initialization and dense-Adam CPU offload parity

**Question.** METH-67 shows that the METH-66 E128 failure cannot be
assigned to a harder development set alone. METH-66 also changed
initialization, optimizer semantics and training draws. Isolate the
first two numerically before another large-E train.

Use the exact METH-55 rank-64 product-key E128 initialization, width
896, rank-8 A/B, top four and BF16 factor arithmetic. Build one dense
GPU reference layer and one CPU-factor/GPU-router layer from the same
initial tensors. Use one packed router parameter so AdamW sees the
same parameter structure. With a fixed seed, run three successive
forward/backward/optimizer steps on identical 32×896 BF16 synthetic
inputs and fixed targets. Step zero begins at B=0. Use the same
objective and factor/router learning rates 3e-4/3e-5 for both paths,
without clipping or weight decay. The CPU optimizer must implement
the **dense** AdamW zero-gradient semantics: decay moments and apply
their resulting update even on rows not selected this step, while
receiving only sparse selected-row gradients from GPU.

At initialization require bit-identical A/B/router tensors. At each
step require identical selected top-four IDs, maximum forward/logit
error ≤1e-5, maximum dense-versus-reconstructed A/B/router gradient
error ≤2e-5, and post-update A/B/router maximum error ≤2e-5.
Record numerical errors, CPU RSS, peak allocated GPU, elapsed time
and count of selected rows. Stop above 5 GiB allocated GPU, 8 GiB
RSS or five minutes. This is an apparatus test on synthetic states;
it cannot promote a model. If it passes, a later full-donor E128
control and E1280 ladder must use new training/evaluation material
and external semantic quality checks. No T4.
