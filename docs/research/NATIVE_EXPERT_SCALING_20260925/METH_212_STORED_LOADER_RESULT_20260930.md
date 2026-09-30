# METH-212: stored-core loader passes development reconciliation

The config-only candidate loader reads all 364 METH-211 tensors and
overwrites all 290 model parameters. Each copy is checked exactly;
the original BF16 embedding/head remains tied. All 24 document NLLs
and prompt donor-match counts equal METH-210's exact-tied arm, and
both proposal buffer hashes match. No pretrained core weights supply
a missing candidate parameter. The centered E1280 factors are loaded
from their separately hash-bound checkpoints.

Local RTX 3060, 65.125 seconds, 2.257 GB end RSS and 4.210 GB peak
allocated GPU. [Raw result](meth212_stored_loader_result.json) SHA256:
`cbc971d4850aa6532c31c7911c0af10001c7df2aee79a5fb2480027b707e89b1`.

Decision: freeze fresh quality selection and scorer. This is a loader
and consumed-source parity pass; no fresh semantics, useful E12800
functions, native precision parity or accepted token rate follows.
