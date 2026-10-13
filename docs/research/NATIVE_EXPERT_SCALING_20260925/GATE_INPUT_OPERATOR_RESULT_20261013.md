# Source gate input information:complete rectangular operator qualification

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / MATRIX PREFLIGHT ONLY.**
No recovered input vector,model forward or causal encoder. Goal active/incomplete.

The missing generator input was not saved by recurrent hooks. Captured3072
gate values are source in_proj linear measurements of2048 inputs before SiLU,
with MuP/output rounding. First check raw trained gate-row matrices for an
unrounded necessary recovery condition,without selecting rows/using FIT or DEV
data. Nonzero scalar MuP/upstream input factors do not change rank/condition.

| Site | Min singular value | Max singular value | Condition | Unscaled inverse gain | Robust rank |
|---|---:|---:|---:|---:|---:|
| 0 | .419381018 | 11.440433641 | 27.279331069 | 2.384466527 | 2048 |
| 12 | .218365372 | 15.561662758 | 71.264333710 | 4.579480665 | 2048 |
| 23 | .294877858 | 16.155576955 | 54.787351923 | 3.391234623 | 2048 |

Full column rank with frozen sigma>1e-6*sigma_max andcondition<=1e6 permits an
OFFLINE recovery preflight at all3 sites. It does not show small actual BF16
error,exact hidden bits or a cheap runtime algorithm. For gate=G*h+e,
`||G^+gate-h||<=||G^+||*||e||`;table gains apply to UNSCALED G only. Actual
capture has projection/MuP BF16 rounding and missing true h. Noise bounds and
scaling/dtype must be established before reconstructing calibration inputs.

Independent audit rebuilt all2048x2048 Grams in256-row blocks,checked full
eigenvectors/orthogonality/residual/spectra and directly SVD-factorized the
RECTANGULAR3072x2048 source matrices. All25,178,112 stored matrix/spectrum
words verified;Gramrelative1.102e-17,rectangularsigma/max3.184e-15,
conditionrelative2.638e-14. Before-after source/runtime/result/arrays hashes exact.
No eigenvalue floor used to assert positive rank;original values retained.

Producer16.80s/terminal16.8138,OS364,679,168B;independentaudit18.5760s/
terminal18.5846,OS398,135,296B. New3 immutable NPZ packets201,428,058B on E,
under512MiB cap. CPU0..5/six BLAS threads,no GPU/source/T4/native/RESERVED/
optimizer/hidden-recovery calls. Execution followed completed native timing.

Preparation V1 completed5.3303s but prefix-based runtime selection omitted
linked NumPy/psutil packages;no matrix run was launched with that binding.
V2 preparation6.7741s explicitly binds resolved primary ABI/BLAS modules;
initial binding/terminal preserved. [Alias witness](numerical_runtime_alias_witness_20261013.json)
records older matching byte fingerprints and their temporal scope limitation.
Producer/auditor0b3639f,V2 source59fe6ed,bindingSHA
`33dfe5b9f774a6d55a54e556c2f35df75d8dd63ec2317156135f0b6ffd1ab1e0`,
launch7f0321b,audit launch4f6e1f4. Producer8549/audit11216 CLOSED0.
Both short preparation commands completed terminal0 without long session handles.

- [Producer V2](../../../benchmarks/native_expert_scaling/gate_input_operator_preflight_v2.py), [protocol](GATE_INPUT_OPERATOR_PROTOCOL_20261013.md), [storage](gate_input_operator_storage_20261013.json), [V2 binding](gate_input_operator_binding_v2_20261013.json), [launch](gate_input_operator_launch_20261013.json)
- [Result](gate_input_operator_result_20261013.json), [terminal](gate_input_operator_result_20261013.terminal.json), [independent audit](../../../benchmarks/native_expert_scaling/gate_input_operator_audit.py), [adjudication](gate_input_operator_adjudication_20261013.json), [audit terminal](gate_input_operator_adjudication_20261013.terminal.json), [closure](full_channel_and_gate_operator_session_closure_20261013.json)

## Selected next:uncertainty-qualified generator-input recovery

**PROPOSED / UNIMPLEMENTED / UNBOUND.**
[Selected implementation boundary](GATE_INPUT_RECOVERY_NEXT_20261013.md).
Freeze actual multiplier dtype/BF16
rounding stages and least-squares or interval recovery policy on FIT only.
Recover source in_proj input AFTER its input multiplier,not raw residual hidden
or a different FFN module input. Replay full source input projection/MuP,
conv4/SiLU and delta softplus/clamp against OTHER stored x/B/C/delta fields on
all48 histories/all3 sites. Gate reconstruction alone is circular validation.
Record uncertainty/residuals/rounding failures;do not claim hidden-bit recovery.
If fidelity insufficient,capture the true source input under a separately bound
donor run. No current source execution/T4 allocation. Data may initialize
source-function transfer only after this consistency gate;compact causal
hidden generation,ternary conditional functions and same useful50 still required.
