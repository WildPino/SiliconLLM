# Full failed-run backward and FD-scale diagnosis

V1 tangent transpose fault and V2 first FD gate failure retained;neither has a
candidate update. All24 FIT outputs/GY/all per-case gradients and aggregate
direction exist. Verify EVERY saved word with CPU autograd and independent
radial/tangent RMS backward;V1/V2 prefix arrays bit exact. Full independent
Kronecker Sylvester projection/constraint/slope qualification at original1e-8
gradient/scalar gates. Preserve all input/old-fault/output hashes before/after.

Evaluate FULL24-case FIT loss along actual frozen retraction at central steps
1e-3/1e-4/1e-5,original1e-4/1e-5/1e-6 error thresholds,and added1e-6 with1e-6
threshold. Record every flag including FALSE,do not mark coarse gate passed.
This diagnoses scale/curvature versus a wrong backward;it cannot accept an
update. All scalars/case losses retained.1800s/OS6GiB,six CPU threads,no GPU.
New derivative/selection protocol requires separate freeze if the evidence
justifies a changed probe scale. No model/native/n/50 admission.
