# Original expert source admission: remaining native scope

5 October 2026. Metadata/code inspection only; no new expert numerical evidence,
checkpoint scan, source copy, native benchmark or shared-checkout mutation.

The frozen parent branch contains an admitted original MoE donor:
`google/switch-base-128`, revision
`86c815ec05361a33a8b49fc717277da9c0a4e711`. METH-378 acquired all original
archives, METH-379 bound actual tensors/functions, and METH-380 exported row-I8
expert matrices with F32 core/control information. The latter is a derived
quantized representation, not an original floating-point expert-weight input.

Original archives (three PyTorch .bin shards) total about 29.86 GB. Qualified
metadata lives under inherited
`docs/research/NATIVE_EXPERT_SCALING_20260925/`:
`meth378_switch_base128_acquisition_result.json`,
`meth379_switch_tensor_binding_result.json`,
`meth380_switch_base128_export_result.json`. Their acquisition/export controllers
describe identities and guards. Original local archive directory in the main
checkout is `results/native_expert_scaling/meth378_switch_base128_source/`.
The dedicated worktree did not copy those untracked archives.

The `native_FFN_controls` field in METH-477-R1 bindings refers to captured
function/context observations, not a small pristine FFN weight bank. These
captures are already consumed evidence. They must not be relabeled as new
holdout or mistaken for float checkpoint matrices.

After the Qwen method screen, an expert test must independently admit the
original floating-point tensors and establish input/reference identity. A
bounded selected-tensor extraction from already hash-qualified archives could
avoid reacquiring the complete 30 GB donor, but requires a new byte/hash manifest
and read-only source checks. Do not scan/copy large shared archives while a
native timing run is active; check resource ownership before that operation.
Any local tensor reader must use the separate runtime and safe weight-only
loading, not modify the main environment or trust a moving output directory.

Missing goal evidence remains: preservation of actual pretrained expert
functions, changed-route nonlinear usefulness, deployable native packed codec,
measured CPU/storage cost on the same artifact, and broad predictive/generative
validation of whatever conversion is eventually promoted. Successful single
Qwen projection experiments cannot substitute for those requirements.

## Follow-up: PQT-SRC-001

Original decoder-11 expert 8/105 wi/wo tensors are now admitted with exact
original float hashes into the owned results namespace. See
`PQT_SRC_001_PROTOCOL.md`, `PQT_SRC_001_RETENTION.json` and raw admission record.
This read-only selected-source qualification scanned neither the entire archive
nor a new input/reference corpus. A later experiment must freeze its own data
roles and resource admission before fitting or native timing.
