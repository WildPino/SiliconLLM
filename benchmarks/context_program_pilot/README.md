# Context-program pilot

Read the [frozen scientific protocol](../../docs/research/CONTEXT_PROGRAM_PILOT_20261015/PROTOCOL.md).
This is a bounded representation experiment, not an admitted efficient chatbot.

## Files

- `pilot.py`: pinned source acquisition, causal captures, three controlled fits,
  separate validation/test decisions, physical layer replacement and own-history
  generation. Model execution is restricted to Kaggle.
- `followup.py`: conditional exception-benefit and archive-reuse diagnostics.
- `config.json`: source revisions, prescribed schedules and wall-clock limits.
- `test_pilot.py`: local-safe standard-library receipt and structural checks.
- `../../scripts/context_program_kaggle.py`: source-only private packaging,
  isolated account operations, launch ledger, status and output retrieval.

## Operations

```text
python -B benchmarks/context_program_pilot/test_pilot.py
python -B scripts/context_program_kaggle.py build --account acct1 --run-id pilot-a-20261015
python -B scripts/context_program_kaggle.py launch --run-id pilot-a-20261015
python -B scripts/context_program_kaggle.py status --run-id pilot-a-20261015
python -B scripts/context_program_kaggle.py output --run-id pilot-a-20261015
```

`build` refuses to overwrite a frozen directory. `launch` records an attempt
before the remote mutation and refuses blind retries or another first-stage
launch in the campaign. Inspect status after an unknown outcome; do not generate
a new run ID merely to bypass the attempt ledger. A successor requires a new
reviewed decision within the aggregate budget.

The package and non-secret operational receipts live under
`E:/SiliconLLM_Artifacts/campaigns/context_program_20261015`. Credentials remain
outside the repository in the existing account store. Source weights and natural
text are downloaded on Kaggle, not on the local workstation.

The cloud bootstrap installs the pinned Transformers/Datasets versions, runs
cloud-only tensor smoke tests, and enforces a total deadline. The model uses
GPU 0 of the T4 pair; GPU 1 is recorded as unused, and budgeting conservatively
charges the full allocation. This small sequential pilot prioritizes a clear
controlled comparison over distributed-training complexity.

Inspect `bootstrap_terminal.json`, `status.json`, `dev_decision.json`,
`test_confirmation.json`, `training.json`, the per-partition mapping/whole-model
reports and `generations.json`. A complete process is not a positive result.
Follow-ups execute only after a positive validation decision. Their prototype
retrieval scans the small archive and their benefit labels use source-input
counterfactuals; neither is a deployment claim.

No GitHub cron is enabled: the four-hour cap is below the current documented
Kaggle session limit. No historical training workflow is resumed.
