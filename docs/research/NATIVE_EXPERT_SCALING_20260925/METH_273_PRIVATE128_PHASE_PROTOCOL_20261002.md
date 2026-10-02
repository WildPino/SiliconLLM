# METH-273: prospective actual private128 CPU phase diagnosis

271 source/reserve fidelity passes but native cost fails;272 input-dot
fusion preserves arithmetic yet does not improve contemporaneous cost.
233's earlier scalar-SiLU diagnosis predates LUT/private128/escapes/rank32
and is insufficient to pick the next optimization. Changed variable here:
measurement instrumentation only,copy the unchanged271 operator/fixture.
No new selector/precision/layout/thread/affinity changes or full archive.

Bind271 raw SHA
`638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3`,
272 raw SHA
`7109476722266d7dcc37abbfbde30bb11ff2c8772cbf7abcc73a4570149b343f`,
fixture337,596,452bytes SHA
`ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8`,
original271 C/check/include hashes and125vectors/all361 segments.
Instrument phase boundaries with QueryPerformanceCounter/`seconds()`:

1. BF16 input decode plus team entry/start measurement boundary;
2. shared Q8 gate/up,LUT/product workshare;
3. private BF16 gate/up,LUT/product overwrite workshare;
4. unchanged concurrent mixed down and rank32 right workshares;
5. rank32 left plus biases/finite check;
6. team exit/final measurement boundary.

Clock boundaries inside the parallel region use explicit single/barrier
handoffs so measurements do not race; retain added instrumentation/barrier
cost in these intervals. Do not decompose concurrently executing down/right
by serializing them. Accumulate all six intervals for each original three
256-token24-layer/six-thread sweeps. All384 vectors must be bitwise equal
271 and aggregate256 checksum identical before interpreting times.
No metric here is an uninstrumented speed qualification or DRAM counter.

Prospective decision: if private phase plus entry/exit median sum is>=20%
of sum of all six phase medians,inspect private-stage/team handoff execution
next. Otherwise if shared plus concurrent down/right medians are>=80%,inspect
matrix/operator layout or memory execution next. Otherwise report unresolved
phase distribution before choosing another optimization. Preserve all three
timings,do not use a best pass or promote271/272 from instrumented totals.

CPU only,six threads,10min/20GiB RSS,native timeout180s,1GiB new output,
no Torch/GPU/model overlap/T4/download. Code must be implemented and committed
before observation. At this protocol freeze no273 code,executable or result
exists. Diagnostic interpretation cannot override the original10ms gate or
267 semantic stop. Full archive,complete quality/native>=50,useful RAM-scale
n/routing-LUT-DRAM and family/10B/100B transfer remain separate requirements.

## Apparatus freeze

`meth273_private128_phase_cpu.c` copies271,adding clocks/single boundaries,
per-pass accumulators and stderr phase JSON. Check header remainsM271OUT1
so entire native check hash must match271. No arithmetic or stage reordering.
`meth273_private128_phase.py` binds the272 runner helper and all immutable
inputs,compiles once,runs once,records all18 intervals and the frozen branch
decision. Apparatus is committed before execution; no273 observations exist
at this apparatus freeze.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth273_private128_phase.py --exe benchmarks/native_expert_scaling/meth273_private128_phase_cpu.exe --check results/native_expert_scaling/meth273_private128_phase.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth273_private128_phase_result.json
```
