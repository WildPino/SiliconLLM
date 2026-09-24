# STRAT-01 block-0 Q6_K×Q8_K AVX2 parity — VOID 1

**Status: `VOID_BLOCK0_Q6K_Q8K_AVX2_PARITY`.** The scientific evidence is
complete and valid, but the original external decision table omitted one
frozen branch. This record is not the scientific verdict and the model must
not be rerun.

The raw directory is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_avx2_parity_20260924`.
Its original `adjudication.json` SHA-256 is
`c38fad6017cb1524bc92025410cd16c70398c5387a2cb9357a9fd801aa42cad2`;
the executed implementation HEAD is
`e46205390b2e4117ae8a63b1946f5b2c2773d6e1`.

The accepted execution performed two admitted model/matrix reads, emitted two
Q8 populations, ran one C diagnostic, and ran zero donor and reference model
graphs. It completed without errors. C and pinned-oracle Q8 bytes are exact;
C AVX2 and pinned-oracle outputs are exact; the generic history and mutation
controls pass. Both candidate and oracle differ from the immutable graph
reference.

The runner classified only `candidate != oracle` as
`BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT`. It accidentally sent the observed
case, `candidate == oracle != reference`, to VOID through the aggregate exact-
repair controls. The frozen protocol explicitly assigns a candidate/reference
difference with exact Q8 bytes to the insufficient branch. Commit
`d92dbb2` repairs the decision table; commit `4568735` repairs only the offline
script bootstrap. The canonical result is an offline, hash-bound
re-adjudication of these same bytes with zero new scientific work.
