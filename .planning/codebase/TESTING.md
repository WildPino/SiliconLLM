# SiliconLLM — testing and verification map

## Test model

This repository uses research gates and executable self-checks rather than a single conventional unit-test suite. The test unit is usually: construct a known positive, assert the engine/config/provenance, run a controlled A/B, then adjudicate against a pre-registered threshold.

## Core verification layers

1. **Exporter/layout:** sidecars, tensor counts, byte counts and Gate V3/layout checks in `benchmarks/donor_adaptation/engine/e1_bpb_through_engine.py` and exporters.
2. **Kernel parity:** reference-vs-optimized outputs, `--kselftest`, logit dumps and E25/E28 parity checks.
3. **BPB identity:** E1 is the standing engine-vs-PyTorch protocol; E62/E64/E65 reuse the frozen 24×512 slice and `N_PREDICTED=12264` where registered.
4. **Fidelity/ranking:** per-position top-1, greedy generation over 160 tokens, teacher-forced vs free-running separation and explicit chance lines.
5. **Speed:** repeated/interleaved arms, clean/contended classification, component timing and bandwidth/occupancy witnesses.
6. **Training health:** finite tensors, optimizer updates, checkpoint/resume identity, stage-transition eval and no false `.done` on divergence.

## Phase 64 test inventory

`benchmarks/phase64/` has 49 Python/C/shell/Markdown files, with smoke/acceptance/assertion logic in most of the data and MVE paths. Key checks include `data/assert_package.py`, `data/ws3_*_audit.py`, `mve/data_smoke.py`, `mve/ws1_acceptance.py`, `mve/ws2_acceptance.py`, `mve/ws4_branch_check.py`, `mve/ws6_probe.py`, `teacher_bpb.py` and `tok_coverage.py`.

`benchmarks/phase64/MVE_PREREG.md` seals five gates: KD vs CE on held-out BPB, end-to-end/resume pipeline, recall stability, KD→QAT transition stability and throughput. It also records two voided attempts and the fixes, including monotonic logit windows, divergence aborts and deterministic resume sampling.

## Known testing lessons

- A planted control must be on the same axis as the claim. E64's first run compared attention kernels across binaries and voided itself; the corrected run asserts `CONFIG attn=serial` and reproduces the full ladder.
- A metric can be a floor/ceiling and therefore be uninformative; free-running floors in E18/E19/E20/E65 are reported but not treated as progress.
- A result file must not become its own control. E65 run 1 was malformed for this reason; E66's runner deliberately separates stage artifacts and scoring.
- A tolerance must match the discontinuity/dispersion of the measured path. Top-k selection amplified a ~1e-7 perturbation to ~1e-3 in E64.

## Gaps

- No obvious pytest/unittest project-wide suite is the authority; verification is script/gate based.
- E63 `G-E63d` remains void/owed: the clean absolute rate of the 10B carved-int8 artifact has not been measured.
- H1 has quality/trainability evidence at 8 layers but not a completed 10B trained model or a full free-running recovery.
- E66 has apparatus and preregistration but no published result/probe in the current tree.

