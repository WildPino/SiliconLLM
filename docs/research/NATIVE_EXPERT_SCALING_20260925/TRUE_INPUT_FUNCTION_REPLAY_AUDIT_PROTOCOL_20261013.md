# Independent actual-input source replay audit

Load actual BF16 H through raw-file bits/capture identity/shape checks. Source
MuP bits independently reconstructed from config using integer BF16 RNE.
Full output-row256 blocks of source projection,explicitBF16 stages,independent
stacked4tap convolution,sign-split sigmoid/SiLU and stable softplus/bias/clamp.
Check all5 BF16 predicted arrays/source relative whole/centered/label metrics,
every bit mismatch count/case flag/FIT/DEV/domain/mean/worst and full hashes.
No imported producer replay/metric/aggregate helper or new source/GPU call.

Independently read legacy H64/H_BF16 from qualified immutable inverse packets;
compare every vector to actualH in exact same units,all recorded diagnostics/
bit differences. These observed error metrics do not select another input.
All actual-input/legacy/predicted words counted;no inverse recomputation needed.
Numeric replay-vector1e-6/scalar1e-8,field-fidelity each1% remain distinct;
record all false flags and complete all144 cases before adjudication.
Audit1200s/OS6GiB/CPU0..5/sixBLAS,complete before-after hashes.
