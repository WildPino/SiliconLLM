# METH-233: frozen phase-cost diagnosis of stopped row-Q8 arithmetic

## Question and decision

METH-232 conserves the full source function on component states but costs
11.023ms/24 layers,above10ms. Before choosing a changed operator, determine
whether the matrix phases dominate or scalar nonlinear work is material.
Reuse the completed result, immutable fixture, vectors and C output hash.
No fitting, new data, GPU or external source is required.

Add QueryPerformanceCounter timers around three existing arithmetic phases:
1. BF16 input decode plus gate/up projections;
2. FP32 SiLU/product;
3. down projection, bias and finite-output check.

METH-233 source copies the METH-232 arithmetic and loader, adding phase
timers/accumulators/reset and a stderr JSON record. Verify the entire first16
states across24 layers (including the unchanged output header) reproduce
the METH-232 check SHA256 exactly. Bind the original source hash, prior
result SHA256 `5dad8dcb67d2b7eeae0c747d63b5cada6e2ce6bd6faed94bac32dd6052c94368`,
fixture/vector/source hashes and record new source/executable/runner hashes.

Three fixed passes over all256 states,24 layers,six threads after checks;
retain all three phase/total timings. Phase clocks add overhead and matrix
phases include OpenMP dispatch, decode/scale and associated boundary work.
They do not isolate DRAM from computation or price a proposed fusion.

Compute each phase median; if input/down medians together are >=90% of
their sum, choose a changed matrix operator as next research mechanism.
Otherwise further activation/overhead diagnosis is required. This is a
diagnostic decision threshold,not a new performance gate. **METH-232 stays
stopped regardless of this instrumented run's total.** No best-pass choice,
retiming, revised10ms tolerance or full-rate/quality promotion.

## Budget and commands

CPU only,six threads;maximum10min,2GiB native peak working set,600s subprocess
timeout. Reuse314.9MB fixture; only~1.4MB check plus small JSON and executable.
One execution, no concurrent project inference. No T4. Freeze before run.

```powershell
clang -O3 -mavx2 -mssse3 -mfma -fopenmp -std=c11 -Wall -Wextra benchmarks/native_expert_scaling/meth233_row_q8_phase_diagnostic_cpu.c -o benchmarks/native_expert_scaling/meth233_row_q8_phase_diagnostic_cpu.exe -lm -lpsapi
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth233_row_q8_phase_diagnostic.py --exe benchmarks/native_expert_scaling/meth233_row_q8_phase_diagnostic_cpu.exe --check results/native_expert_scaling/meth233_row_q8_phase_diagnostic.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth233_row_q8_phase_diagnostic_result.json
```

The runner refuses existing check/result paths and preserves any binding,
arithmetic/resource failure with its stage. Result bytes use pinned CRLF.
