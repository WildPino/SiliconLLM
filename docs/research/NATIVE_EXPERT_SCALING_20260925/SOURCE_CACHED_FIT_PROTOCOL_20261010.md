# Coherent cached final FIT features: frozen protocol

10 October 2026. New missing observable, no codec or recovery training.

## Decision and provenance

The qualified one-case paired control reproduced all53 original tokens/logits,
and reconstructed its observed norm/head exactly. Its full audit and first
startup fault remain immutable. ALL48 full-prefill source alignment FAIL and
the catastrophic actual51 recovery/readout results remain unchanged.

New variable: actual cached-generation final raw state AND actual postnorm
features paired with same-call logits for the authoritative24 FIT records
(12 domains,2 cases each,4422 labels). Original cached logits were retained;
these intermediate final states were not. Capture them once, with one source
model instance and24 generations. No optimizer/native/RESERVED query or DEV
generation. One prior DEV case is only instrument-qualification provenance;
it is not representation-calibration data.

PASS admits coherent FIT features for selecting ONE output representation/head
pair compatible with original engine RMSNorm. FAIL preserves every completed
case and requires a mechanism diagnosis; no quality gate is relaxed. Any hard
fault preserves first stage/counters/completed IDs. Do not replay completed
observations; a necessary repair uses new code/binding/namespace.

## Operators and custody

Tool: `benchmarks/native_expert_scaling/source_cached_fit_capture.py`.
Falcon-H1-1.5B pinned source411 parameters/1,554,863,488 elements, BF16,
Transformers5.13.1/Torch2.6.0+cu124/NumPy2.4.6, eager attention, qualified
tiled SSD installation. Seed0, TF32off,6 CPU threads/affinity0..5, batch1,
max_new_tokens256, do_sample=false/use_cache=true, EOS[11,228]. Assert the
runtime fast-path flag AFTER model construction, as qualified in repair1.

Every generation hooks base-entry IDs/cache length/position IDs and the final
norm input/output. Save only the last prompt row then each cached-step row.
Capture actual logits, verify their BF16 representation is lossless and argmax
is the generated ID. Reconstruct norm once per case and head once per label
using the ACTUAL source operators. Hook removal precedes these reconstructions.
Full411 BF16 parameter byte SHA256 against safetensors before/after family,
plus object identity/version checks. This claim covers parameters, not buffers.
The final family seal must equal its initial seal. Case artifacts are exclusive,
flushed/fsynced and saved before the case scientific verdict is reported.

Four exact gates per case, all must pass: old output IDs; all cached-logit BF16
bits; observed postnorm versus norm(raw); observed logits versus head(postnorm).
Scientific failure does not stop the remaining cases or the stored audit.
No approximation threshold is substituted for an exact gate.

## Full independent stored audit

Verify every bound input and terminal-listed output hash; sealed case records;
all four exact gates; every actual input/position/cache frame; all411 parameter
seals; generation/base/head/reconstruction call counts; terminal resource caps.
BF16 postnorm versus F64 analytic RMSNorm(gamma,eps1e-5): relative RMS<=.01
per case. Up to24 scalar head witnesses/case via math.fsum, absolute delta<=1
to accommodate the real BF16 head rounding. No F64 full-head contraction.
Exact identical score bits imply KL=0; audit their decoded argmaxes explicitly.
Cache-coordinate rounding was already qualified separately, not recaptured here.

## Costs and stops fixed before observation

Earlier FIT source observations total548.904s;4422 labels. Forecast new capture
10-15min. Numeric output1,213,555,992B if all old token trajectories match:
three BF16[4422,2048] state/feature matrices and two BF16[4422,65537] logits.
No width/rank/scale grid, codec fit, or benchmark during this capture.

Held capture cap1500s including pre/post input seals, worker reserve25s;
OS worker+launcher<=8GiB, GPU allocated<=10GiB/reserved<=11GiB;
namespace+result<=2GiB/log<=8MiB. Independent audit cap300s/OS<=1GiB,
result<=512KiB/log<=8MiB, no GPU. Hard deadline/resource/unexpected-child
violations are preserved terminal faults. Observation timeout is not exit;
retain the exact session and Win32 PID/create_time through termination.
Launcher checks overlap; only the exact foreign publisher daemon is exempt.

## Reproduction

Run the tool with isolated Python3.12 `-I -S -B -X utf8`:
`--bind --binding <new-binding> --protocol <this-file>`; inspect/freeze code,
protocol and binding in a scoped commit. Then `--binding <binding>
--binding-sha <SHA256> --freeze <commit> --directory <new-namespace>
--out <result>`. After terminal completion and PIDs gone, run same frozen
tool with `--audit --source-result <result> --out <new-stored-adjudication>`
and the same binding/SHA/freeze. Save both logs and terminal receipts.

Goal remains incomplete: compact codec/history, useful experts, same-artifact
quality/50token/s, physical DRAM, structured routing and family scaling unproved.
