# METH-243: parent-value conservation alone does not recover count gain

Frozen at `db17a96`; session49082 completes,exit0. All160 parent-point and
centered-mean checks pass (worst mean relative error7.82168e-10). Only
child-prior/fitted-child biases change;all176 distinct actual weight hashes
and all parent fit/per-window validation scores remain exactly METH-240.
Every saved snapshot tensor reads back. No new label solve.

| Normalized output SSE | Parent prior | E16 | Hierarchical child prior | E160 | Rotated E160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit |.0001851522|.0000574206|.0001184407|.0000995864|Not scored|
| Consumed validation |.0001856472|.0001279367|.0001661520|.0001568577|.0001913903|

E160 loses **22.6057% against E16**. Absolute1%,rotated and parent-prior
gates pass;E16/child-prior10% and positive paired gain fail. Bootstrap
P05/P95 **-2.999998e-5/-2.789555e-5**. Close this fixed parent-value/source-
slope hierarchy as a count solution. This does not reject preserving
learned parent information in other function/representation mechanisms.

[Raw result](meth243_parent_value_prior_result.json),SHA256
`ac40b7453cf6414df62911b7d3cf87b1aeb403b8a4005921506470345af25770`.
Ignored local snapshot `results/native_expert_scaling/meth243_layer12_parent_value_functions.safetensors`,
1,591,450,188bytes,SHA256
`5ab304516dac2d7c7c3abb6710052d25c7cf2d4dd90f9785a248ada0579e5d6d`.
Runtime41.094s,RSS3,128,193,024bytes,GPU peak3,447,201,792bytes,local3060,
six threads,no T4. Consumed one-layer development evidence only.

Next: [METH-244 fit-only encoding audit](METH_244_READOUT_ENCODING_PROTOCOL_20261001.md).
Separate continuous fixed-ridge fit error from weight/intercept encoding
distortion before choosing a new correction. No more fixed-coefficient
intercept-only count recipes;full independent/native/rate goals remain open.
