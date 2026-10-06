# PQT-ENV-002: GPU runtime admission

PQT-ENV-001 completed, private/CPU/no Internet confirmed by exact server metadata.
The CPU image uses Python 3.13.15 and PyTorch 2.11.0+cpu, so it cannot qualify the
GPU scientific runtime. Raw outputs and hashes: `env_001_output/` and
`env_001_fetch.json`. No scientific observations or input downloads occurred.

Prospective second operational probe: same inventory code plus device names and
memory, acct1, fresh private reference `wildpino/pqt-gpu-admission-20261005-002`,
Internet disabled, accelerator `NvidiaTeslaT4`, 600-second server timeout.
GPU quota reservation may cover that full timeout; expected actual probe time
below one minute, but startup/runtime overhead is recorded rather than assumed.
No tensors fitted, pretrained weights loaded or corpus evaluated. Save admission,
dispatch, response, exact server metadata, terminal outputs and their hashes.
Two physical T4 devices and a CUDA-capable build must be confirmed before PQT-001.

Operational correction: `kernels_list` returned default/omitted metadata fields;
search did not find the just-pushed private reference. Neither establishes public
visibility. Read exact `get_kernel`/`kernels_pull` metadata for privacy and image.
Preserve the initial status record, and use the corrected exact metadata reader.
