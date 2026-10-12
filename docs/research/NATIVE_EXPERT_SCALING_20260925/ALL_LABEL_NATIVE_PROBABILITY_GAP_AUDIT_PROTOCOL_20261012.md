# Full stored-point and actual native/source probability audit

12 October2026. CPU1200s/OS6GiB,no subprocess/Torch/GPU replay. Frozen completed
report/terminal/binding and all consumed inputs/output extents before/after.
All initial/every16/final stored point packets,not a sampled checkpoint audit.

Independent C/beta/epsilon F32 promoted F64. Scores sum32-feature blocks plus
normalized bias;partition function combines4096-word block partitions by
logaddexp,unlike CUDA reduction. Gradients use4096-vocabulary sums;moment linear
term uses8-coordinate blocks. Verify every current point/loss/g,best lower point/
loss/g,explicit support minimizer/bound/display allowance,and executable best
tF32 positive-sphere real loss. Identical current/lower point rows may reuse the
same independently computed result ONLY under exact coordinate equality for that
same row's fixed m/H;no approximate cache or missing rows. All recorded point
functions remain covered. Verify counts/steps/feasibility/monotonic checkpoint
envelopes/trace weights. No proof of convergence or exact unstored trajectory replay.

Independent scalar explicit-FMA DLL verifies ALL577249896 final actual native
score words/2254848 normalized words/8808 inverses bitwise. Direct actual source
BF16 head/gamma/multiplier/raw H/sourceepsilon probabilities computed with256-
feature sums;every native full KL/ID/entropy/case/domain/quality flag checked.
Independent moment real loss/native score span identity,precision flags/numerical
lower summaries checked. Numerical real-domain bounds do not imply interval-
certified/global F32-machine bounds or learned-model/causal quality.

Array norm(delta)/max(1,norm(expected))<=1e-9;scalar delta<=1e-8*max(1,abs(expected)).
Probability mass1e-11;K norm²<=256+1e-9;checkpoint lower<=upper+1e-8,monotonic
envelopes1e-8. Native score1e-4,norm/inverse2e-5 rederived unchanged;native-vs-real
KL gap<=score span+1e-9. Source splits/case/domain weights and quality limits
unchanged. Full hashes/terminal/PID/creation on failure;no partial audit admission.
Source quality can fail while stored mechanics audit PASS. One complete bounded
reader-image diagnostic then return to compact causal/conditional transfer goal.
