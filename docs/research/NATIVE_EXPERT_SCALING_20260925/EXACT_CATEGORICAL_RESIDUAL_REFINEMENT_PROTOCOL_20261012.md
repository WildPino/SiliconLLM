# Separate finite-derivative refinement

12 October2026, frozen before new observation. Original result NONACCEPTANCE
remains immutable:13/14 gates passed, joint coarse1e-5 relative error3.053e-6
versus tolerance2e-6;fine2e-6 error1.221e-7. Independent explicit full RMS Jacobian
and all gradient/point/update audit PASS. Error reduction25.00 for step reduction5
motivates testing central-difference convergence, not loosening tolerance.

Reevaluate all7 fixed directions at eps2e-7 (14 full FIT score packets), using
original frozen source head/states/reference/factors/directions. Same2e-6
relative/absolute derivative tolerance. Also report Richardson combination of
the two ORIGINAL scales, (25*d_fine-d_coarse)/24, as diagnostic only.

New acceptance requires all7 ORIGINAL2e-6 and all7 NEW2e-7 controls pass. Original
1e-5 outcomes still recorded as failed;do not reinterpret original acceptance.
Select from original frozen4 FIT candidates only, same minimum FIT/1e-4 relative
improvement criterion. No new gradient, alpha, data, chart definition or epoch.
Seal selected parameter extent/alpha before new selected DEV72 quality calculation.
Generic factors need not be idempotent;report W^T V-I defect. Source/core/native
requirements and original full-goal gates stay unchanged.

300s/OS4GiB/output1GiB/CPU6/no children. Full parent inputs AND every original
output/provenance plus new self/protocol bound/hashed before and after. No
source model/GPU/native/T4/RESERVED/longalign calls. Complete independent source
readout/probability/refined difference/selection/DEV audit regardless of failure.
