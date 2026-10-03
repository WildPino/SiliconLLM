# METH-307: unchanged binary, active waiting promising but inconclusive

**INCONCLUSIVE variation; active14ms gate FAIL. No collection/training.**
Freeze21b9ce3, exact306 executable/spec/geometry/weights/head, no recompilation.
Three sequential fresh processes PASSIVE/ACTIVE/PASSIVE; terminalexit0,
12.812s overall. ALL90 complete output/route hashes and entire selftest/ready
counts exact306. Fresh source component/spec bindings exact; peakRSS5.126GB.

Runtime KMP_SETTINGS confirms PASSIVE effective blocktime0ms in A/C, ACTIVE
200ms in B. This changes worker waiting, with CPU used while waiting. Source
quality/precision/synthetic-bank limitations remain306; no new model artifact.

| Arm | Three measured medians ms | Max/min |
| --- | --- | ---: |
| A PASSIVE |19.27175 /18.52720 /17.72990 |1.0869633 PASS |
| B ACTIVE200ms |14.78005 /11.76440 /11.31985 |1.3056754 FAIL |
| C PASSIVE |18.01785 /22.55460 /21.42390 |1.2517920 FAIL |

Passive A/C median drift1.1563485 FAIL<=1.10. B/A=.63498 and B/C=.54913
are observed ratios, not a stable isolated speedup. B's first14.78005ms
median also FAILS EACH<=14ms. Do not select later/faster repetitions or
replace306's PASSIVE cost failure. Active profile remains unqualified.

Next: freeze an exact-binary physical-core affinity profile to reduce thread
migration/sibling ambiguity, retain six cores and all gates. No evidence yet
that affinity caused this variation or will pass. Windows metadata read after
the terminal run identifies physical logical-ID pairs(0,1),(2,3),(4,5),(6,7),
(8,9),(10,11); process permits0–11. One per core is0,2,4,6,8,10. Require
actual runtime binding verification before interpreting another cost run.
See [prospective affinity profile](COMPACT_NATIVE_AFFINITY_PROPOSAL_20261003.md).

## Bindings and independent recomputation

- [raw result](meth307_openmp_wait_policy_result.json), SHA256
  `42d6959cf7be160137e5d94dceb8ebeac76ae886b7f0315cb5945fc0f59e059a`.
- Controller `3c08734e08a890028d6bc16923358ebde9ac727a32a1271237c0b9464d276352`;
  unchanged306 executable `f7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1`.
- All logs isolated under `results/native_expert_scaling/meth307_wait_policy/`:
  A stdout `fe623e37e535968c47b45e78a7c6ea8d22dac381e7f8b79d1625490c33993dbf`;
  B `317b5eb7566f7b642120f31db0968d71f5bb5b21dcdc9998e5cfee50c41ec496`;
  C `25e8b462c228ed01b734ef6b476d590d4983cac23bbfa50446419f13e401646a`.
  Runtime stderr/settings hashes retained in raw record.

Independent recomputation of all90 original hashes/selftest fields/three-arm
medians confirms controls and failed variation. Frozen controller/protocol
working bytes exactHEAD before execution. Original source/binary/results kept.
[LLVM runtime documentation](https://openmp.llvm.org/design/Runtimes.html)
motivates worker-wait interpretation; effective runtime settings are observed
directly. Final transfer/quality/useful n/SAMEartifact accepted50 and verified
multiple-family/scale requirements remain open.
