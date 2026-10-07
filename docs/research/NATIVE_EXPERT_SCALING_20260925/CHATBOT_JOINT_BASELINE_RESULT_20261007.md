# First jointly learned compact layer: fit improves, novel response remains inadequate

7 October2026. Goal INCOMPLETE. [Original calibration operands](CHATBOT_SOURCE_CAPTURE_RESULT_20261007.md)
are available and canonical input is qualified. The first actual joint nonlinear
conversion trial supplies two separate negative results, not an admitted chatbot.

## One fixed finite fit actually completed

UniformE160 [exposure](CHATBOT_JOINT_EXPOSURE_RESULT_20261007.md) fails28/160 leaves
before initialization/fit. E16 exposure passes. The separate
[baseline protocol](CHATBOT_JOINT_BASELINE_PROTOCOL_20261007.md), source/input/runtime
frozenf719fab, enters E16 source-row initialization/shared+leaf G/U/B joint
optimization for the FIRST time. Reuses original saved geometry/parent choices/
mass; no PCA/clustering/control/source full forward replay. No E160 fit/admission.

Layer12 ZERO-BASED, full input896, shared512 plus four selected128-channel
SwiGLU functions from16.32-dimensional fixed selector does not replace the
896-dimensional function input. Source coefficient rows select by FIT source
activation-squared times down-column energy; shared/private indices and actual
initial coefficients retained. Source feature-energy matrix newly acquired ONCE
on3166 unique original FIT x, not full-response targets or original BF16 replay.
Original y bits remain targets. Reciprocal exact-x multiplicity weights avoid
counting shared prompt repetitions as new information.

Exactly24 epochs/744 updates, batch128 including5-row last batch, Adam.0003,
anchor.001/clip1, seed2601007/F32/TF32-off/deterministic. Final epoch only, no
development checkpoint choice, width/update/regularizer ladder or precision tuning.
All16 leaves changed from their own initializer and have distinct BF16 hashes.
Normalized squared coefficient displacement shared.1749839, leaves.076879..451513;
this is actual learning/distinctness, not useful extra capacity.

## Complete response errors

| Arithmetic / domain | Initial relative RMS | Final relative RMS | Final normalized SSE |
|---|---:|---:|---:|
| F32 / reciprocal-x FIT3845 rows |74.6971%|21.6779%|.04699330152|
| F32 / novel reserved DEVELOPMENT1495 rows |76.2348%|61.3921%|.37689848254|
| BF16-rounded coefficients, F32 arithmetic / FIT |—|21.7206%|.04717865066|
| BF16-rounded coefficients, F32 arithmetic / novel DEVELOPMENT |—|61.3923%|.37690199611|

Primary development excludes exact x present in FIT. Its1495 occurrences
include one duplicate novel input (integer multiplicity LCM2); all1864 original
development occurrences remain reported separately. All16 categories FAIL the
fixed3% RMS policy: F32 category range43.5732% translation..70.3071% continuation.
Novel generated-input1054 rows65.2643% RMS; prompt441 rows51.7301%. Values above
are RMS, not squared percentages. All four fixed1% novel/3% category fidelity
criteria fail. Output coefficient rounding changes the primary dev RMS by
about.000286 percentage points; it is not the dominant observed error here.
This reference includes BF16-rounded router matrices/centers and their actual
F32 norms/choice/mass. It is NOT a native BF16 reduction/execution qualification.

Fit improves substantially and novel development also improves relative to
initialization, but the final normalized development error is8.02026 times
FIT error. Both residual training error and poor generalization remain under
this finite schedule. We have not established the optimizer's best achievable
loss, a general capacity lower bound, or which additional source information
will resolve it. No conclusion that all joint nonlinear conversion is impossible.

## Independent retained numerical and information verification

[Audit](chatbot_joint_retained_audit_20261007.json), source37671bc, actual exit0,
uses ONLY stdlib plus psutil; no Torch/NumPy/GPU/function/optimizer/source forward.
A restricted storage-descriptor parser verifies exact shape/stride/offset/
storage counts/little-endian buffers of original saved Torch route files.
ALL16/160 independent exposure fields match; adding children leaves every parent
ID AND F32 mass byte identical. Original28 failed leaf IDs independently match.

All four saved C-order F32 NPY prediction matrices are reconstructed against
original BF16 source operands/positions. Independent scalar math.fsum verifies
complete primary/occurrence/generated/prompt/ALL category rows, energies,
normalized SSE and RMS within fixed1e-12 absolute OR relative tolerance.
No model/function/prediction/fit rerun, and no retuning after values.

Both primary1%-RMS failures also have EXACT integer witnesses. Finite F32/BF16
values are integers in units2^-149, squared units2^-298. Reciprocal multiplicity
weights clear withLCM2. The first64 novel-row exact error energies divided by
the FULL1495-row source energy are.01876881177 F32/.01875914469 rounded-reference,
already greater than.0001 (the squared1%-RMS threshold). The audit verifies
`10000*prefix_error_integer > FULL_source_energy_integer` strictly. Remaining
nonnegative error cannot reverse it. This proves the saved finite gate failure,
not a universal error lower bound for this architecture. Exact integers retained.

## Actual resources, actors and artifacts

| Stage | Worker elapsed | Whole monitored family | Worker OS peak through exit | GPU allocated / reserved peak |
|---|---:|---:|---:|---:|
| Original exposure |8.594s|32.469s|1007767552B|331643392 /371195904B|
| First E16 baseline |75.312s (arm66.187s)|100.610s|1721667584B|367100416 /501219328B|
| Independent retained audit |12.984s|14.750s|461168640B|0 /0|

All actual exits0, all fixed resource limits pass; scientific eligibility fails.
These are conversion/diagnostic costs, never accepted inference speed or DRAM.
Baseline parent last33120256B/summed1754787840B; audit parent last29569024B/
summed490737664B within its prospective512MiB. Parent final receipt/stdout tail
outside last snapshot explicit; worker actual exit/serialization peaks held by
Windows process handle. Source/runtimes/foreign SHA pre/post unchanged.

Baseline worker30564/create1791388239.9788437/FILETIME134358618399788436,
launcher23872/create1791388228.1579864. Audit worker4644/create1791389140.4919896/
FILETIME134358627404919896, launcher32020/create1791389139.5567262.
[Terminal closure](chatbot_joint_terminal_20261007.json):ALL6 original exposure/
baseline/audit known instances closed, typed UTC Event1000 zero matches,
positive179810/179791 controls; exact worker FILETIME/declared launcher16tick
float tolerance. No scientific repeat for terminal metadata. All three foreign
tracked SHA remain unchanged.

Raw [baseline](chatbot_joint_baseline_20261007.json)/[terminal](chatbot_joint_baseline_20261007.terminal.json),
SHA256 `42d5356d4bd0e7dc009ad8108c0365da0427449427cdd1781034eee6c691ad97`;
audit SHA256 `8eca7c810284a8dd1effa3d64d23a1f4d7010028df4748aeb54a6e0c843c23c8`.
Executors:baseline6648cf/session52052/final8d8aea exit0;
audit2ab3b8/session38141/final38d985 exit0. Binding/actual full source freezes,
coefficient/prediction/update-journal file sizes and full SHA are retained.

Initial and final F32 layer checkpoint27647285B each, proposed BF16 layer
checkpoint13825333B; source initializer energy61722868B separate. Torch
checkpoints contain conversion buffers (including unused C1 child-center copies)
and serialization metadata. They are not complete24-layer deployment banks or
the previously priced mixed-payload format. The future C1 exporter would omit
unused child centers/norms; complete proposed cost ledger is not a measured
checkpoint/DRAM/rate claim. No full source basis is evaluated by the fitted block.

## Decision in the complete pipeline

Close this exact finite E16 initializer/24-epoch objective recipe. Original
uniformE160 exposure remains closed, its response/utility unknown. Do not
assemble24 failing blocks, qualify a native candidate or measure its speed as
if donor capability had transferred. Existing native297 semantic failure and
Switch511 infilling scope remain preserved.

[Next source-information decision](CHATBOT_SOURCE_INFORMATION_NEXT_20261007.md)
prioritizes a NEW source-structural/directional transfer constraint over an
update/width/precision ladder. Uniform child allocation and source-function
generalization are separate problems. Keep final requirements explicit:
complete compact chatbot export/phase60 execution, fresh own-history/tasks AND
>=50 accepted IDs/s on SAME artifact, useful large n/RAM/LUT winner+mass/actual
DRAM and real additional families/scales. No goal completion or percentage claim.
