# METH-240: full-feature functions are accurate, tenfold conditional growth loses

Frozen at `f25f0b1`; session53680 completes,exit0. All source/capture/native/
key/count/label bindings pass. Sixteen actual parents fit;all160 source
child priors pass the same derivative/value/conditioning/growth checks
(worst stored slope error.654414%). All176 actual output solves/intercept
checks pass and **176 decoded effective coefficient matrices are distinct**.
Every physical snapshot tensor reads back;no copied/noisy-capacity repair.

| Normalized output SSE | Source parent prior | Fitted E16 | Source child prior | Fitted E160 | Rotated E160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit |.0001851522|.0000574206|.0001182768|.0000994223|Not scored|
| Consumed validation |.0001856472|**.0001279367**|.0001658301|**.0001565790**|.0001914072|

Complete-function1% absolute gate passes easily: E160 output SSE/energy
is0.015658%,E16 is0.012794%. But E160 is **22.3879% worse** than E16,
failing useful-count gain. E160 beats rotated choice and original source
parent priors by the required10%,but its improvement over source child
priors is only5.58%,also below10%. Paired-window gain P05/P95 is
**-2.973170e-5 /-2.759479e-5**,robustly negative on these development windows.
All128 energies replay exactly;every window/arm is retained. Fit E160 is
also worse than fitted E16,so validation-domain overfit alone is not isolated.

**Decision: stop this fixed mixed full-feature conditional recipe.**
No routed-bank native/n-pool promotion,full-model/fresh prediction,
generation/tasks or accepted-rate claim. METH-238's8.727ms source component
cannot be combined with local accurate-but-count-failing functions into
final-goal evidence. Accurate dense-donor function approximation does not
demonstrate transferred approximately10B capacity or useful extra n.

[Raw result](meth240_mixed_conditional_pair_result.json),SHA256
`3c02724b1194f305036aae6896a489e1d19b477d7d8fe1d3ddc45fa9579fc505`.
Local ignored snapshot
`results/native_expert_scaling/meth240_layer12_mixed_conditional_functions.safetensors`,
1,591,450,068bytes,SHA256
`23f4a92b8092ccfbf79575ecd2dd59bd300b50fd6124e818cfbf862327def611`.
65536 unique fit inputs,131072 readout-fit exposures,16 parent+160 child
fits and160 original-source derivative compilations. Runtime278.734s
after imports,RSS3.249GB,GPU peak3.477GB,local3060,six host threads,no T4.
All source/compiler/codec/fit/physical/effective hashes retained;no job active.

Next: [METH-241 source-precision diagnosis](METH_241_SOURCE_PRECISION_REPLAY_PROTOCOL_20261001.md).
The training target is actual BF16 donor output,while source priors use
FP32 smooth FFN values/Jacobians. This possible mismatch is not yet measured
or proved to cause the count loss. Replay source precision at the exact
capture batch geometry and inspect the known constant-input child before
defining a changed precision-matched prior mechanism. No strength/count
retry or threshold change is licensed by this result.
