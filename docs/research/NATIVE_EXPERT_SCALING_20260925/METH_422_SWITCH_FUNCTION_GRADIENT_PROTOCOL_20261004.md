# M422 protocol: stable equivalent loss, same native/gradient bounds

Freeze both NEW422 Python sources/protocol before first numeric outcome.
420 first real FD failure retainedb512542; diagnostic421 frozen8c68486/retained
bdf6de3, raw2d9f26ed199f6cc73817af613fd24a94f46ef3eda602319b28e29a6449d16111:
same original256 A FD error exactly reproduced; equivalent centered loss AND
independent imaginary derivative agree with old autograd at SAME1e-5, no ReLU
crossings.420 stays FAIL, no retroactive qualification or relaxed bounds.

## Sole numerical change, original controls retained

Replace final cross-entropy's subtraction of leading full logits by equivalent
F64 centered loss: k=argmax(z), s=z-z[k], log1p(sum(exp(s[j]),j!=k))-dot(target,s).
Torch actual/smooth loss AND independent NumPy FD reference use this form.
This changes numerical conditioning, not target/T1/objective/rank/seed/direction
or error threshold. Target normalization from same detached captured logits.
Forward values and ALL primitive approximate backward rules in422 helper must
be byte-identical SOURCE to419 before prediction_loss definition. Exact prefix
identity required. No model/weights/native engine/arithmetic/selector changes.

ALL tiny and real controls exactly [419 protocol](METH_419_SWITCH_FUNCTION_GRADIENT_PROTOCOL_20261004.md),
with repaired Windows disk-guard420: tiny smooth/FD1e-5, native STE1e-3; real
both sources first shared teacher.book0.case0 positions, rank8 A/C0/B/D seed419+n,
live A/C>1e-10 and exact-zero B/D gradients. ALL states/32128 logits with zero
correction exact. Opposite source target/T1, central FD epsilon1e-4 on SAME fixed
Rademacher directions, same1e-5 and1e-8 denominator floor. All pre/scores/A/B/C/D
native-STE vs smooth gradients1e-3, smooth-forward approximate1e-3. Finite
differences test smooth model; no derivative claim through native rounding/top1.
ALLfive native/gradient negatives must now be reached and pass. No tuning/sweep.

## Reuse of exact whole-forward proof, without unnecessary recomputation

Fresh420 failure SHA17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427
and its384 complete archive hashes bind ALL4895 original final-bank paths and
ALL157266560 vocabulary rows, previously byte-exact against418 C. Since helper
forward/primitive source is EXACT419 and only loss changes, INHERIT whole384
forward evidence through this identity and freshly rehashed complete archives.
Do not claim384 native paths recomputed422. Two real rank8 forwards ARE freshly
computed and compared at every endpoint/full head. Whole-gradient qualification
does not follow from loss diagnostic alone: run all tiny/real/negative gates.

Fresh418 raw/ALL1936 native inventory/helpers, complete22.36GB qualified source
payloads/manifests/every lookup and tokenizer/source provenance. Native teacher
approximation and natural qualification-only exclusions remain. No new corpus,
reference/native model job or GPU/T4/network; CPU only main logical0/Torch1/BLAS1.

## Resources, decision and next scope

New run<=5min/RSS or Windows peak working set<=3GiB/output<=16MiB. Two real
head I64/dequant workspaces and full native/smooth gradients charged as before.
One selected function cache per point, no optimizer or training updates.
Record inherited vs newly run forward scopes, all hashes/runtime/gradient/FD
errors/negative controls, complete gradients/factors/logits and resource ledger.
Rank8 factor/storage/gradient/Adam prospective ledger unchanged419, full head
charged; selector and actual fitting budgets must be fixed next.

PASS licenses ONLY a NEW bounded function-specific correction/selector fit
protocol on consumed40518dev/6val all cases: frozen256 core+old256 functions,
actual128 matching-bank functions in ONE final decoder bank, exact no-added
baseline, causal original-function ablations and explicit hard active budget.
No adapter-only or gate-always-old capacity claim. Positive calibration then
requires excluded original-relative quality/generation/task and SAMEartifact
native FULL accepted50, plus actual added-function consultation/benefit.
FAIL retain first before repair. All previous failures preserved.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth422_switch_function_gradient_contract.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth422_switch_function_gradient_result.json`

Goal active/incomplete. Useful large n/RAM/CPU LUT/realDRAM/other-family/~100B
remain open; numerical qualification alone establishes none of those.
