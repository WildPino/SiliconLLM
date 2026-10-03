# METH-309: verified Windows binding, cost profile inconclusive

Freeze `dab4d88`, native exit0 in all three processes; total 20.296s. All90 output/route hashes, entire selftests and fixed counts match306. Seven singleton affinity checks/process pass: six separate physical cores, CPUs0,2,4,6,8,10, before and after every repetition. Runtime ACTIVE200ms/library-affinity-none verified. Bootstrap masks verified, current CPU required at subsequent checks.

| Fresh process | Repetition medians (ms) | Max/min | Peak RSS (bytes) |
| --- | --- | --- | --- |
| 1 | 34.68425/15.54165/17.83365 | 2.231697 | 5127380992 |
| 2 | 23.53010/41.14840/16.10130 | 2.555595 | 5126119424 |
| 3 | 18.40090/22.88245/22.25730 | 1.243551 | 5126139904 |

Cross-process median ratio 1.319421>1.10. Every process fails variation; all nine medians also exceed14ms. Decision **INCONCLUSIVE**, no candidate training/promotion. Recompilation changes the binary; this is not an isolated affinity comparison. Binding is verified, the cause of timing variation is not.

Raw [result](meth309_windows_affinity_result.json) SHA `446ac7d8732966d629f5aaefd1584044f7c0a23e55d62d7ef347a2a4d0916dd1`. Controller `e5b387035668f003967ba69efed485d1712689bd002575a354d53f912d3c8613`, C wrapper `3bc0596f0aed166d23ea68f979e69bb2ca1158450dcdcdc40cbbd34589885729`, exe `76ca9c8317c4381442ebb952c9026ae4eee9d3a8d1b0f51c6bde3b66fd1138d9`, engine `8de41952d1e358a739fdd07cbb24310f822ecbca33551262a76581c759a375cd`. [Frozen protocol](METH_309_WINDOWS_AFFINITY_PROTOCOL_20261003.md). Exact spec/source bindings remain306. Three logs under `results/native_expert_scaling/meth309_windows_affinity/1`, `/2`, `/3`; native commands and requested/effective environment are recorded in raw result/logs.

Reproduce in a clean output area with `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth309_windows_affinity_profile.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth309_windows_affinity_result.json`. Controller refuses existing outputs/exe/logs; preserve original observations. No concurrent model benchmark.

This closes the affinity investigation for now: [308 startup failure](METH_308_PHYSICAL_CORE_PROFILE_FAILURE_20261003.md) is preserved;309 supplies actual binding proof but no qualifying cost. All640 functions/layer are synthetic. No learned donor quality, increased useful capacity, physical DRAM or accepted-token rate follows. Next investigate the regional-function representation using existing298 captures, as a mathematical feasibility diagnostic, without training the failed execution profile or recollecting source data.
