# METH-193/194: directly stored grouped-Q8 FFNs with centered E1280

METH-192 shows that the best rank-94 weight reconstruction under the
560 MB ideal addressed design allotment captures only 20.85% median
of the stored Q6 FFN error. Test a direct higher-precision format
before further low-rank correction work. Keep the exact METH-85 R8
tied head/embedding, BF16 attention, FP32 controls, and centered
METH-122 E1280 bank. Replace only METH-186's grouped-Q6 FFNs with
signed symmetric int8 codes, one FP16 scale per 64 input weights,
using round-to-nearest with scale equal to group absolute maximum/127
and zero-group scale one. Store one code byte per weight. Verify every
stored tensor byte and BF16 reconstructed FFN value after reload.

The 72 FFN matrices contain 313,786,368 weights. Compared with Q6,
Q8 adds 78,446,592 bytes/token with identical scale count and dtype.
The expected ideal core+router total is 559,981,568 bytes/token,
only 18,432 below 560 MB. Record physical artifact size separately.
This calculation ignores cache lines, metadata reads and any extra
router selection as n grows; it is not a DRAM or C-rate claim.

On the same 24 already viewed METH-121 sources, score BF16 donor,
BF16+E1280, Q8 donor and Q8+E1280 for document BPB and donor prompt
top-1. Reproduce METH-187 BF16 per-source values exactly. Require
Q8+E1280 minus BF16+E1280 pooled BPB <=0.01, each category <=0.02,
pooled top-1 >=-1 point and each category >=-2 points. These are
development gates only. If any fails, do not use fresh quality
sources or run full native C rate on this Q8 candidate. If all pass,
freeze a source-disjoint quality test and then implement/measure the
same stored format in `engine.c`; >=50 accepted batch-1 tok/s remains
the rate gate. Explicitly revisit byte cost as the router and expert
bank grow toward 10B/100B scenarios.

Use local RTX 3060 for export and scoring. Each run is capped at
15 minutes, 10.5 GiB GPU allocation, 20 GiB RSS, 1 GB file/report.
No T4 is authorized by this protocol.
