# METH-308: startup access violation, no native timing evidence

**APPARATUS FAILURE. No gate/repetition/source-quality observation.** Freeze
41d7dd7; first native process terminal status0xc0000005 after2.797s, sampled
peakRSS2,547,712B. Stdout empty: no ready/selftest/operator events. Three-run
profile stopped; do not infer a cost failure/pass from this launch.

The exact306 binary was launched with ACTIVE200ms and explicit verbose
KMP_AFFINITY logical IDs0,2,4,6,8,10. Windows API topology confirmed six
physical pairs/allowed0-11. Runtime stderr confirms requested effective
settings but contains NO actual worker binding proof. It fails before source
fixture allocation/timing. The raw native environment field is the reused
launcher's PRE-wrapper PASSIVE snapshot; actual child settings are in stderr
and frozen308 controller. That snapshot must not be misread as applied policy.

Windows Application Event1000 at03/10/2026 11:41:28 identifies matching PID
0x5DE0/24032, exception0xc0000005, fault module
`C:\WINDOWS\SYSTEM32\ntdll.dll`, version10.0.26100.9444, offset0x12c2e9.
This locates the reported fault; it does NOT establish a root cause or prove
which runtime/OS interface is defective. Original306/307 unbound evidence
remains valid, explicit library affinity profile remains unavailable.

Preserve failed record/logs before a NEW direct Windows thread-affinity
profile: bind through native SetThreadAffinityMask, read actual group masks
before/after each repetition, keep source/model math/geometry/precision/head,
two warmups and14ms/repeatability gates. This requires a new opt-in C wrapper
and frozen protocol/controller, not silently repairing or regrading308.

- [failure record](meth308_physical_core_profile_result.failure.json), SHA256
  `1ae635e3a2751a4a1149d32abfc08c72e2d1304a62c8a2ed25a2e7fd1dacb605`.
- Controller `7f0a8951dcc011c898edae3b2128fde568fce6654d366dc3da6239aeb5cb001f`;
  executable still306 SHA
  `f7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1`.
- Stdout empty; stderr under
  `results/native_expert_scaling/meth308_physical_core/1/`, SHA
  `240e9dd3085e692bddd27b615e6229026841ea28c50a2ebc9fd703371da91d83`.

No model training/teacher collection or final quality/rate promotion.
Full pretrained-transfer/useful n/SAMEartifact50/multiple-family goal stays open.
