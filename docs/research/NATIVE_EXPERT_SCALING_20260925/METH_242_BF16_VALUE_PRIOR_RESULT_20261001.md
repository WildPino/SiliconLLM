# METH-242: source-value precision correction fails useful count gain

Frozen at `d5ab5e2`; session30493 completes,exit0. All160 BF16 point and
centered-mean controls pass (worst mean relative error8.04546e-10).
All snapshot tensors read back;only child-prior and fitted-child bias arrays
change. All176 effective weight matrices remain distinct and exactly equal
METH-240;parent fit and every validation-window parent score replay exactly.

| Normalized output SSE | Parent prior | E16 | BF16-value child prior | E160 | Rotated E160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit |.0001851522|.0000574206|.0001228090|.0001015402|Not scored|
| Consumed validation |.0001856472|.0001279367|.0001703975|.0001586853|.0001938640|

E160 is **24.0342% worse than E16**, versus22.3879% in METH-240. Absolute
1% function,rotated and parent-prior gains pass;E16/child-prior10% gains and
positive paired bootstrap fail. Gain P05/P95 **-3.186144e-5/-2.969343e-5**.
Precision-matched source values alone do not solve the count loss. This
does not reject BF16 source execution or all precision-matched adaptation;
it closes this fixed value-only correction without strength/rounding retries.

[Raw result](meth242_bf16_value_prior_result.json),SHA256
`a3bb1e94ff8decc40363b904c0b39a3d583d13d57a721e69fd1e6d913ea5ddfd`.
Ignored local snapshot `results/native_expert_scaling/meth242_layer12_bf16_value_functions.safetensors`,
1,591,450,188bytes,SHA256
`e05228c30b6f8bb6e12de3eca06020c246fc49852f3688d861f5b1c4cc7bf03d`.
Runtime43.000s,RSS3,374,682,112bytes,GPU peak3,450,042,368bytes,local3060,
six threads,no T4. No new label solve,160 source-point replays. Consumed
development windows only;no native pool/full independent quality or rate.

Next: change parent-child information coupling. METH-243 conserves the
already fitted parent's value at the child center while retaining actual
source-derived child slopes. This preserves parent adaptation instead of
resetting values to the original donor. It is separately frozen,not a
precision or tau retry. See [protocol](METH_243_PARENT_VALUE_PRIOR_PROTOCOL_20261001.md).
