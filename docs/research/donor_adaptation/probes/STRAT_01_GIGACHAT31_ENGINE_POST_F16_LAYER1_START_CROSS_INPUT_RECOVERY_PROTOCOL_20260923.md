# STRAT-01 post-F16 layer-1-start offline recovery protocol

**Status:** `FROZEN BEFORE RECOVERY EXECUTION`

The sole scientific invocation of the post-F16 layer-1-start split completed
the C diagnostic successfully, emitted both 32-checkpoint arms and both causal
controls, then the external runner declared
`VOID_POST_F16_LAYER1_START_CROSS_INPUT`. The only error was an incorrect
Python-side expected helper count:

`post-F16 helper count mismatch: {'mode': 'pinned-generic-f64', 'qk_invocations': 4608, 'value_invocations': 524288}`

No model or graph rerun is permitted. This protocol authorizes one offline
recovery that validates and adjudicates the already emitted immutable files.

## Correct count derivation

One eight-token attention arm performs:

- QK: `32 heads * (1 + ... + 8) = 1,152` helper calls;
- value: `8 tokens * 32 heads * 512 values = 131,072` helper calls.

The command ran four attention arms: reference, current, token-6-negated
reference, and rows-0/7-swapped reference. The exact totals are therefore
`4,608` QK and `524,288` value calls. The rejected expectation
`9,216 / 1,048,576` had accidentally doubled both totals. The frozen parent
protocol requires exact nonzero counts in `pinned-generic-f64` mode; it does
not prescribe the erroneous values.

## Immutable recovery inputs

Raw root:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_20260923/`

| input | SHA-256 |
|---|---|
| raw `adjudication.json` | `6ed2d017ca36c166929f658b4a79a77a242adffb5afb39ee986bdda10494e6bf` |
| C diagnostic report | `7c49afcf18272575124455d70ed82979e931ee7c29b8b6c02119d41b78538a3f` |
| helper-count report | `19892ac3af937f601854246657f91b9ff0042c168fa7ed7a8761b4a24a8553da` |
| producer binary | `6dccd8941166f55beed2a4020ed0e87046f07a0857e5b8e8aae3c4fc81561629` |

The producer commit is
`8f03fa3963723fc2083cc2d608437d1a487c9666`. The raw record must report one
successful diagnostic invocation, zero donor/reference graphs, and exactly
the helper-count error above. Every C-report payload and control must still
match its recorded size and SHA-256. The accepted artifact, frozen reference,
post-F16 current inputs, source hashes, exact current replay, mutations, label
swap, and causal controls remain mandatory.

## Recovery command and stop rule

After committing the corrected arithmetic and this protocol, run exactly:

```powershell
& '.\.venv\Scripts\python.exe' -u 'benchmarks\donor_adaptation\engine\run_strat01_post_f16_layer1_start_cross_input.py' --recover-existing
```

The command must create a fresh `recovery1` directory, execute no compiler,
binary, model, donor graph, or reference graph, and write one externally
adjudicated record. Any raw-input mismatch is a new VOID. A valid recovery
inherits the original single diagnostic invocation and may issue only one of
the parent protocol's two scientific verdicts. Do not overwrite or relabel
the preserved raw VOID.
