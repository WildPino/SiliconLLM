# METH-39: isolate fine-rescore arithmetic in the route replacement

METH-38 includes every exact expert with 128 candidates yet changes
21/6,144 next-token top-1 IDs. The route *sets* match perfectly,
so candidate omission cannot explain that control failure. This
diagnostic compares three ways to calculate the same fine scores
on the same already inspected METH-37 prompts. It does not tune
or promote a route on those prompts.

Bind all METH-37 source/core/adapter/sketch and prompt hashes,
METH-37 result SHA-256
`d3d69a24c767987b38b57bf6dc240194259174ad5476ab59f757661d12b4b5d5`,
and METH-38 result SHA-256
`a14e5a04475ffac972a45ad405e7382bcbe883ae61ed07f7c07faedc9132bbe4`.
Reproduce the saved METH-38 exact, C64, C96 and C128 top-1
counts before interpreting alternatives.

Keep the same rank-64 stored int8 shortlist and exact candidate
count. At C128, test (1) existing elementwise multiply+sum,
(2) batched matrix-vector multiplication of candidate rows and
hidden input, and (3) exhaustive `F.linear(x,W)` followed by
gather of the same candidates. The third arm is an **oracle
numeric control** that reads every fine-router row and is not
a deployable bounded route. The first two are candidate-only
rescores. Also run C96 with batched matrix-vector multiplication;
interpret its diagnostic gate only if the C128 batched arm
achieves at least 99.99% top-1.
Each arm runs on its own 24×256-token trajectory. Compare every
top-1 stream to the saved exact arm and report route-set agreement
with the exact router on the same arm's layer inputs.

The oracle control must reproduce **100%** exact top-1 and route
sets; otherwise stop and fix the apparatus. The batched C128
numeric gate is ≥99.99% top-1 and 100% route-set agreement.
If it passes, C96 batched has a diagnostic ≥99% top-1 gate;
any promotion still requires new documents, generation and CPU
timing. A failure closes only that rescore kernel. Do not
reinterpret METH-37/38's failed gates.

One local RTX 3060 run, ≤15 minutes wall time, 10.5 GiB
allocated GPU memory and 20 GiB RSS. No T4 job is planned.
