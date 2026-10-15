# Context-program pilot: first launch

15 October 2026. This record establishes an authorized cloud launch, not a
scientific result. The final pipeline objective remains incomplete.

## Observed state

- Private kernel: [context-program-pilot-a-20261015](https://www.kaggle.com/code/wildpino/context-program-pilot-a-20261015).
- Account: `acct1`, authenticated server identity `wildpino`.
- Kernel version 1 was accepted by the API. The subsequent status query returned
  `KernelWorkerStatus.RUNNING`.
- Maximum session duration: 14,400 seconds. Internal bootstrap deadline is lower;
  the numerical pilot reserves time for evaluation and output.
- Allocation request: `NvidiaTeslaT4`. The numerical process rejects anything
  other than two T4 devices. It uses GPU 0 and records GPU 1 as unused; conservatively
  budget the whole first allocation as at most eight T4-hours.
- All three existing accounts authenticated as distinct server identities. Their
  read-only quota queries reported 30.00 GPU quota hours remaining before launch.
  Only `acct1` received a new kernel. `acct2` and `acct3` remain available.

No local source weights, dataset captures, tensor self-tests or training were
executed. Nine standard-library structural/receipt tests passed locally. Causal,
finite-value and actual decoder integration smoke tests execute in the cloud
before donor acquisition and fitting. RUNNING does not mean these checks passed.

## What executes

The [protocol](PROTOCOL.md) fixes one source revision, one small natural-text
dataset, one four-layer span, three separately trained controls and the
validation decision. Test data confirm the frozen recipe after that decision.
The exception and archive diagnostics execute only if the validation gate is
positive; otherwise their non-execution is recorded. No automatic hyperparameter
search, second GPU session or model promotion is authorized by a positive flag.

The bootstrap uploads its complete source-only archive, including the protocol,
inside a private script. Model and dataset acquisition happen on Kaggle.
Source/config SHA256 pins and the private launch metadata are preserved in the
[launch receipt](launch_receipt.json).

A subsequent [remote source/metadata readback](remote_readback_receipt.json)
confirmed private visibility, T4 selection and the uploaded source. The Windows
CLI rewrote downloaded source line endings to CRLF; normalized source and AST
match, while both distinct raw hashes are retained explicitly. The embedded
payload remains identical.

## Where to resume

The frozen bundle and append-only operational receipts are at
`E:/SiliconLLM_Artifacts/campaigns/context_program_20261015/pilot-a-20261015`.
These paths are outside all in-flight storage migration units. The ongoing
storage copy was left running.

```text
python -B scripts/context_program_kaggle.py status --run-id pilot-a-20261015
python -B scripts/context_program_kaggle.py output --run-id pilot-a-20261015
```

Do not relaunch the same attempt or create a new ID to bypass the ledger. Inspect
the live state and retrieve outputs first. A failure remains evidence; repair
requires a distinct, budgeted decision and retains the original failure.

Inspect cloud `bootstrap_terminal.json`, `status.json`, `training.json`,
`dev_decision.json`, `test_confirmation.json`, source/data pins, per-partition
metrics and own-history generations. Successful execution alone is not success
of the hypothesis. No CPU-speed or large-model-capacity claim follows from this
pilot.

The current four-hour job fits below Kaggle's documented 12-hour CPU/GPU cap.
No GitHub cron was created or activated. The read-only workflow inventory did
not identify the historical LM relay on the inspected remote branches; no
existing workflow was modified.

The previously paused long-term research goal is not marked achieved. Waiting
for this bounded cloud job does not authorize unrelated research, model training
or another long run.
