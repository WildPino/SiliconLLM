# METH-201: trained E12800 children have little functional separation

The [frozen protocol](METH_201_CHILD_FUNCTION_SPREAD_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth201_child_function_spread.py)
were committed at `3d53b7e` before execution. The run bound the
exact 4.404 GB METH-175 candidate bank and the already-viewed
METH-173 source manifest. Eight deterministic 128-token causal
windows were passed through the trained model, capturing all 24
pre-MLP states, parent IDs, gates and selected child IDs. Initial
BF16 teacher/student parity was exact on eight prompts. For every
content selection, reconstructing the selected BF16 child B output
and entire MLP output was exact.

The nine content children of each selected E1280 parent were applied
to the **same real hidden state with the same parent gate**. Across
98,304 top-4 selections, 95,616 selected a content slot and 2,688
the structural slot. On content selections, the gate-weighted root
mean square sibling spread around the nine-child mean is only
**1.583% of that mean residual**. Its root mean square is **0.00396%
of the captured full MLP output** in the pooled energy calculation.
Per-layer sibling/mean ratios range 1.288–2.454% (median 1.653%);
per-layer sibling/full-MLP ratios range 0.00081–0.02821%
(median 0.01779%). The selected child deviation energy is 0.999×
the nine-child average, so the exact route is not selecting unusually
large child deviations on this sample. None of the 24 layers meets
both preregistered 10% sibling/mean and 1% sibling/full-MLP
thresholds.

The [raw result](meth201_child_function_spread_result.json), SHA-256
`1b12c7db45edd66c3f0e59c26ebc9c88501c66da3fddf5a3989db17d45833a24`,
contains all 192 source-layer rows, counts, energies, output parity
checks and runtime. Model-side work took 42.469 s on the local
RTX 3060 with 4.673 GB peak GPU allocation and 11.306 GB ending
RSS; initial large-file hashes are outside this elapsed counter.
No T4 or new source was used.

**Decision:** change the specialist-training signal before spending
more work on the route alone. This functional result complements
METH-178's 2.318% child-specific weight-difference fraction and
METH-179's failed route-rotation ablation. It is not an upper bound
on future trained specialists: forcing child-specific learning,
changing the route, or using a donor with more capacity could alter
the geometry. It is also not a quality or full native-rate test.
