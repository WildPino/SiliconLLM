# Progressive ternary quantization: research setup dossier

5 October 2026. Status: **RESEARCH ACTIVE; FOUR AUDITED SCREENS COMPLETE; NO ARM PROMOTED**.
Read [TERNARY_INDEX.md](TERNARY_INDEX.md) for current progress and resumption.
The authoritative owner-facing [research goal](GOAL.md) is in Italian.
Technical and scientific documentation is in English. Documents must not be
signed or attributed to the assistant, its model or its tools.

## Objective and scope

Investigate whether progressive conversion of pretrained floating-point
weights to ternary codes, guided by the source model's behavior, can preserve
predictive and generative capabilities without pretraining from scratch and
improve the practical quality, storage and execution tradeoff.

The contribution should support native expert scaling by preserving pretrained
expert functions in an economical CPU representation. Methods and next steps
remain evidence-driven; random search is optional. The first prospective screen
is described in the progress index and its protocol.
Local reconstruction, whole-model quality and native performance are separate
claims. Scale and correction overhead must be counted. Negative findings close
the tested hypotheses, not every possible ternary method.

## Isolation and provenance

- Branch: `research/progressive-ternary`.
- Base commit: `022242cbf4e68f28c7576205934289b0e9d31029` from
  `research/native-expert-scaling`; this base stays fixed as the other branch
  advances. The attached worktree and exact local paths are recorded in
  [setup_manifest.json](setup_manifest.json).
- Main checkout: `D:/_THINGS/Progetti/SiliconLLM`. Treat completed artifacts as
  potential read-only inputs after identity and hash verification; never write
  this research's outputs there.
- Dedicated operational runtime:
  `D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.venv/Scripts/python.exe`.
  Python 3.12.10 and Kaggle 2.2.2; installed dependencies are pinned in
  [requirements-setup.lock.txt](requirements-setup.lock.txt).
- The main Python environment is unchanged. Qualified GPU runtime is pinned
  by the experiment protocols; separate owned CPU scientific runtime is
  recorded in `PQT_CPU_RUNTIME_ADMISSION.json`.
- Credentials stay outside the repository. No credentials are stored here.
- Untracked base files were not implicitly copied. Full Switch archives have not been duplicated; four original expert
  tensors are admitted under PQT-SRC-001. Qwen/corpus acquisition is pinned
  and verified in the remote experiment evidence.

This work owns `PQT-*` IDs, `scripts/progressive_ternary/`,
`benchmarks/progressive_ternary/`, this dossier and
`results/progressive_ternary/`. Preserve the other work's METH numbering and
research index. Do not merge or push automatically.

## Three-account Kaggle operations

The accounts have separate weekly budgets; they are not three individual jobs.
The owner supplied a nominal allowance of 30 hours per account per week using
T4Ãƒâ€”2. Read actual used and reserved quota before each future launch. Do not
arbitrarily multiply API hours by the number of GPUs.

API snapshot: **5 October 2026, 19:56 Europe/Rome**.

| Alias | Verified server identity | API allowance | Used | Reserved | Conservative unreserved |
| --- | --- | ---: | ---: | ---: | ---: |
| acct1 | wildpino | 30 h | 0 h | 0 h | 30 h |
| acct2 | giggio253 | 30 h | 0 h | 0 h | 30 h |
| acct3 | sirwildpino | 30 h | 0 h | 0 h | 30 h |

Observed total: 90 API quota hours. This snapshot does not establish absence of
interactive sessions, future accelerator availability, exclusive account
ownership or a reset date. It is not launch admission.

Evidence: [setup_account_snapshot.json](setup_account_snapshot.json).
The read-only inspector provides no upload, push, cancellation or notebook
modification. It reuses the frozen `scripts/kaggle_ops.py` authentication
isolation and `scripts/h4_kaggle_stage_a.py` quota query/Duration compatibility.
Each account runs in a child process bounded to 90 seconds. SDK stderr and
exception messages are not forwarded. Existing snapshots cannot be overwritten.

Capture a later snapshot with a new output filename:

```powershell
& 'D:/_THINGS/Progetti/SiliconLLM-progressive-runtime/.venv/Scripts/python.exe' scripts/progressive_ternary/kaggle_setup.py --out docs/research/PROGRESSIVE_TERNARY_20261005/accounts_next.json
```

Operational lessons retained from `KAGGLE.md`:

1. Isolate authentication per account without changing global credentials.
2. Use UTF-8 I/O. The correct setting is `PYTHONIOENCODING=utf-8`; the historical
   document contains the typo `utf-utf8`. The inspector configures its streams
   directly and does not modify the other researcher's environment.
3. Use SDPA on supported T4 architectures. Eager fp16 requires separate numerical
   qualification; NaN/Inf cause an explicit failure.
4. Locate Kaggle inputs recursively and verify their hashes. Do not assume fixed
   mount paths. Enable Internet only when required and record that choice.
5. Count applied optimizer updates, not loop iterations. Preserve optimizer,
   RNG and data-position state for checkpoint continuation.
6. Before future launches, verify owner, quota/reservations, active sessions,
   inputs, fresh PQT slug, privacy, actual hardware and resource budget. Preserve
   notebooks and datasets belonging to other work.

Account allocation remains a protocol choice. Start with economical feasibility
work; replication and independent evaluation follow when justified. Manage the
two T4 devices explicitly using declared independent jobs or a multi-GPU
strategy. Do not assume unified memory. Record actual devices in each run.

## Local resources and concurrent research

The worktree isolates files, while CPU, RAM, disk and GPU remain shared. Setup
does not reserve the RTX 3060 or 80 GB RAM. Coordinate heavy fitting, downloads,
checkpoint reads and benchmarks with the researcher using the main checkout.
Final CPU timing requires an exclusive window on the target machine. Kaggle
supports fitting and GPU evaluation; it cannot establish the local C rate.

No daemon, scheduler or recurring polling was started. Future scientific inputs
must have immutable manifests rather than point into outputs still being written.

## Evidence and research readiness

[PROTOCOL_TEMPLATE.md](PROTOCOL_TEMPLATE.md) lists fields to freeze before each
experiment. [AGENTS.md](../../../AGENTS.md) records worktree operating rules.
Inherited METH32/33, METH88/89/90 and METH477-R1 evidence keeps its original
scope. Already observed validation data is consumed evidence, not fresh holdout.

Setup verification: three operational guard tests, live identity/quota checks,
`pip check`, Git checks and exact file/staged-byte hashes. The immutable initial
setup manifest identifies the files at setup time. Later revisions are recorded
in versioned documentation manifests linked to that original setup.

The owner started research execution on 5 October 2026. Bounded private
operations and experiments follow preregistration and live resource admission.
The operational [runtime probe](PQT_ENV_001.md) precedes scientific GPU work.

PQT-003 head-fidelity experiment completed with independent nonlinear replay; all quality gates failed. See TERNARY_INDEX.md for results and the next research boundary. Overall research remains incomplete.

PQT-004 coverage study completed: relative KL benefit without preservation. Original Switch weights and explicitly consumed native inputs are admitted; function/native comparison remains pending. See FEASIBILITY_EVIDENCE_MATRIX.md for the full missing scope.
