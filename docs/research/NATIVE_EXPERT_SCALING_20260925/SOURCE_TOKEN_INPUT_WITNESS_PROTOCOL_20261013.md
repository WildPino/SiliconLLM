# Initial source input:exact finite-input lookup opportunity

Local pinned source code:embed_tokens(input_ids)*embedding_multiplier,
layer0.input_layernorm,then mamba SSMin multiplier BEFORE in_proj. Attention
and source SSM consume normalized input in parallel;position embeddings enter
attention,not this initial SSM input. Thus the declared initial input chain
is a fixed word-ID function in this source/eval/batch1/no-padding setting.
Do not extend this to later residual layers,postconv outputs or recurrent state.

Before witness observation,check EVERY22547 layer0 actual H row and compare
EVERY repeated word's complete2048 BF16 bits across all48 histories/positions.
Store unique row hashes/first locations/counts/all mismatches. Require exact
same bits for repeated IDs;complete even mismatch. This is empirical on
observed tokens and source-graph deduction,not execution proof for all65537.

Pin capture/adjudication/binding/config/source implementation/protocol/runtime/
actual H0 fields before-after.60s/OS1GiB/CPU1/no GPU/source/native/rank fit.
A proposed word->gate/raw-conv/delta BF16 table has65537*6704*2=878,720,096B.
This is a costed compiler alternative for layer0,same capacity/redundant
storage,not additional donor intelligence or ternary expert scaling. Full
vocab derivation/rounding/field/native equivalence/actual DRAM/latency and
the remaining23 context-dependent functions still need qualification.
