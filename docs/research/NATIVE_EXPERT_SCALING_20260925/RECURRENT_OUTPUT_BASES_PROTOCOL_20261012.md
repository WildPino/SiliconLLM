# Frozen FIT recurrent-output covariance and actual384 scan assay

12 October2026. New variable:FIT equal-case per-head GY_rec from source y-D*x,
inF64. GM copied exactly from old balanced basis packets;no new fitting of GM.
sqrt(GM)*sqrt(GY_rec) balanced SVD,biorthogonal V/W;fixed8 or min1/head plus336
largest remaining squared singular values,total384. No DEV fit/rank choice.
PSDrelative1e-10,min retained singular1e-12,biorthogonality1e-8. Independent
CPU covariance relative1e-10,weighted kernel/tail relative1e-8.

All48 histories/sites0/12/23,two layouts,new actualF32 chunk128 SSD scans with
full source B/C/delta/A. Save every projected input and scan coordinate. Full
scan and recurrent-only source projection agreement relative1e-4. Independent
CPU sequential recurrence uses all384 coordinates/state256/every timestep,
relative1e-4;projected input independently checked relative1e-6.
Not reuse of previous latents for a changed W.

Both projected/fullskip output arms use F64 on promoted actualF32 maps/scan/
projected inputs,captured source SiLU gate/gamma/source weights;input rounded
BF16,full output projectedF64 then roundedBF16. Full source denominator oracle
and jointly recomputed combined postgate denominator. Repeat old-basis old/full
skip F64/BF16 metrics against fully audited prior result,absolute1e-4.
Original oldF32 precision bridge failure remains;no equivalent precision claim.

Save all eight BF16 outputs/twoF32 inputs/twoF32 scan matrices per record.
Independent CPU NumPy/BF16 RNE full stored output check relative1e-4;
all metricsabsolute1e-4,energies/cross termsrelative1e-8,identityrelative1e-10;
full covariance/basis/rank/temporal sequences/aggregates/domains/gates/hash audit.
Every output/coordinate/timestep,not a sampled scalar substitute.

Original local DEV gates:equal-case output mean<=.1,worstcase<=.2,worstdomain
mean<=.2,centeredmean<=.1. Gates are not chatbot quality admission. Audit whole
artifact even if they fail;no changed tolerances after observation.

Producer1800s/audit1800s,OS6GiB,GPUallocated4GiB/reserved5GiB,output3GiB.
Principal runtime/version/GPU identity+actual source/operands/code frozen before
launch. Full source nonlinear generators/gates/conv4/state256 retained;no new
source forward/model/native/T4/RESERVED/longalign. No clean timing or n/50 claim.
