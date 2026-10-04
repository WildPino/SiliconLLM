# METH-399: both table construction and row gathers exceed budget

Frozen9e3ed75, complete execution exit0, MAIN27.750s. Instrumentation removal
reconstructs398 source exactly; engine new opt-in only/default exact0ff9705.
All90 same64-bit mixed output/route fingerprints EXACT398; all original integer,
source control, negative and LUT tests PASS. These fingerprints are not archived
whole-output byte comparison. All profile category/call/nonoverlap gates PASS.
No changed arithmetic/precision/layout/fixture or affinity/wait settings.

## Nonoverlapping component attribution

Nine repetition summaries: each eight fixed nonwarm input median, three fresh
processes. Full median172.439-233.961ms; pooled repeat1.356777 fails1.10. This
instrumented pipeline cannot qualify speed or quality. Timestamp overhead and
execution scatter can change absolute latency relative398; no precise paired
causal slowdown ratio claimed. Attribution remains diagnostic for this profile.

| Component | Min/max of nine medians, ms | ALLnine exceed14ms? |
| --- | ---: | --- |
|builder|51.580350 /69.904450|True|
|rows|110.311750 /149.152750|True|
|router|1.489800 /2.069700|False|
|head|4.784000 /6.760400|False|
|other|4.558950 /5.960300|False|

Both table builder and row consumption alone exceed14ms in ALLnine medians.
**Decision:** changing only one cannot rescue the unchanged other component
within this operator budget. Require a joint algorithm/layout change under a
new frozen contract, before source-aware book fitting or large-bank work.
These are empirical necessary-cost observations, not a universal impossibility,
PMU/cache/DRAM diagnosis or proof of which instructions/hardware cause them.

| Operator category | Builder range, ms | Row range, ms | Calls/token |
| --- | ---: | ---: | ---: |
|mla|27.335700 /34.020750|49.684550 /69.633150|130|
|routed|17.053650 /25.621950|44.205600 /57.959050|75|
|dense0|0.522950 /0.668850|2.646450 /3.327700|3|
|shared|6.768650 /9.749900|12.823150 /20.227400|75|

Category medians are separate summaries, not additive quantiles. For EACH of
90 individual records, builder+rows+head+router<=FULL+1e-8s, positive region
times and nonnegative remainder. Calls [130,75,3,75] reconcile283 executed
matrix operators. Tables include OpenMP construction; rows include dispatch
and same scaling. Router/head measured separately; remainder includes quantizers,
other norms/activations/mixtures/glue. All work remains within timed execute;
fixtures/load/selftest/hash/serialization outside. Profile rows retained with
full times and each rep/input in raw, not estimated from descriptor bytes.

SAME398 source-sized26-layer/D1536/64-parent/top4 GigaChat geometry; synthetic
additive codebooks/codes and actual original Q6 head/F32 router/norm/BF16
embedding. Independent per-layer residual/context fixtures exclude causal
attention/RoPE/KV/full layer composition. No source knowledge fitted, useful
new n/held-out quality/accepted model rate/physical DRAM measured. Builder
logical340,000,768B writes/token remains not physical memory traffic.

## Resources and bindings

Maximum process RSS3,232,645,120B;20min/12GiB/compile120s limits met. Three
six-worker OpenMP/PASSIVE/DYNAMIC false/KMP_AFFINITY none processes, PROC_BIND
absent/inherited settings sanitized; physical masks unmeasured. No new weights,
training/GPU/T4 or model/native timing overlap. Fresh source segments/lookup
match318/398 spec; whole old archive hashes remain prior evidence only.

Raw `meth399_integer_lut_cost_profile_result.json`, SHA
`0749edfb40ec44a5ddecf0c943cc9d2df9821e2787c486e23170d19de40e5314`.

- cpu_source_sha256: `5e1aeb664d5b9192fdb82204648bbf649b2e9272e6ae730e53f03aa9be15597a`.
- controller_sha256: `108b08bc3b6310ca254bb1b809ea7d02aec268920070be62c4d4775e2b0e10cd`.
- protocol_sha256: `defc4adcec9dd435ec6c8acac149ae817967c51b5daf253bf380143534d0506b`.
- engine_sha256: `4281439139926d349a94134aea7860b97ba10093ab873f69b164854a79689acb`.
- executable_sha256: `c62d1a50f96304ad7a0046fef6f50b7112d3935eba8ec2415183264f8e221ba6`.

Local spec/binary/logs: `results/native_expert_scaling/meth399_additive_i8`.
Previous398 rawSHA44647e766b0918504ff3d3209f88bd6254d452c979bce9079296d81aeb7beb48
and code/fixture/hash bindings retained. Command:

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth399_integer_lut_cost_profile.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth399_integer_lut_cost_profile_result.json
```

Qualified Switch source128/256 artifacts/rates remain separate and unchanged.
Generic donor-adaptation runtime paused; final goal incomplete.
