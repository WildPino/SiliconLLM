# PQT-001 first failure: runtime admission

PQT-001 version 1 ended ERROR. It failed in `configure()` before runtime.json,
apparatus execution, donor download, tokenization, fitting or held-out evaluation.
There are **no scientific results**. Original source and run remain immutable.
Raw retrieval: `pqt_001_fetch.json`; committed small raw evidence:
`pqt_001_evidence/`; complete local retrieval:
`results/progressive_ternary/PQT-001/remote_001/`.

Installation succeeded (17.0167 seconds); experiment process stopped after
7.8387 seconds, configuration check after 0.0094 seconds. Exception:
`ValueError: scientific runtime differs from frozen environment`.
The requested pinned GPU image, private visibility and accelerator were confirmed
by exact server metadata. The mismatch is unresolved, not a quantization failure.

Apparatus limitation: initial environment inventory collapsed all distribution
records into a name/version dictionary, whereas admission uses
`importlib.metadata.version()`. Multiple installed distributions may select
different versions. This is a hypothesis, not yet a diagnosed cause. The failed
check also did not save its actual version dictionary before throwing.

Prospective operational repair PQT-ENV-003: fresh private CPU-only reference
`wildpino/pqt-env-repair-20261005-003`, requested same pinned GPU software image,
Internet enabled only to install the same three package pins. No donor/data
access. Capture selected metadata versions, all matching distribution versions,
and the actually imported library versions. 600s server timeout, installation
300s. Verify exact returned image; a different CPU image does not qualify the
GPU environment. Expected GPU quota zero. Preserve outputs and all failures.
Only after diagnosis, preregister a numbered PQT-001 repair with unchanged
question, methods, data, metrics, thresholds and seeds.
