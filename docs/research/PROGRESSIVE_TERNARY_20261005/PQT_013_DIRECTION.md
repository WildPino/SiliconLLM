# PQT013 direction: matched alphabet/granularity on original expert contexts

6 October 2026. Prospective design only. No new input read, candidate fit,
controls, implementation qualification or dispatch. This does not modify
PQT012's frozen ranks, matrices, methods, gates or execution protocol.
PQT012 is now [completed and adjudicated](PQT_012_RESULTS.md): no candidate
passes the joint gates; complete1,013-entry raw/source/client evidence is
byte-exact at actual Gita9ddda48. This distinct matched-alphabet screen remains
worth one bounded test; it is not a refit or a revision of those negative results.

## Decision question

Does the literature-matched rowwise asymmetric three-level alphabet improve
preservation of the original pretrained Switch expert function over a matched
rowwise symmetric alphabet and the existing group64 baseline, under the same
natural calibration/evaluation and complete storage cap? This is an original-
function diagnostic on four selected experts, not a whole-model QMoE replication.

Use the already qualified PQT012 source capsule and encoder11 experts81/87/18/91,
without new extraction or favorable expert selection. All inputs are consumed
research evidence and do not establish new-domain or pretraining exclusion.
A promotion beyond the diagnostic requires fresh, independently frozen final
confirmation and whole-model/native evaluation. PQT012 outcomes have been
adjudicated. Prepare this distinct mechanism under its own frozen protocol and
controls; no numerical launch is implied by this direction document alone.

## Primary mechanism checked

Read the pinned official [QMoE quantizer](https://github.com/IST-DASLab/qmoe/blob/9110baa9466f2a7d8590e3c5dc3a5e11f7446604/quant.py)
and [compensated rounding](https://github.com/IST-DASLab/qmoe/blob/9110baa9466f2a7d8590e3c5dc3a5e11f7446604/gptq.py).
These files are Apache2.0; no copied code or numerical reproduction is added
by this direction. Quantizer lines21-25/37-55 use row endpoints, zero wins
half-level ties, and optional symmetry equalizes endpoint magnitudes. Rounding
lines39-108 uses fixed row parameters, block128, damping.1, no activation order
by default, and progressively compensates uncommitted columns. Its exceptional
singular-matrix fallback is an algorithmic difference that must be explicit.

This is not generally an affine equally spaced grid. `{-a,0,+b}` permits
independent magnitudes. Count two row parameters even if a symmetric arm can
share one. QMoE's capacity1024, propagated full-model contexts and special-token
filter differ from the original capacity64 natural context contract here.
Preserve original capacity and routing; call the result literature-informed.

## Minimal discriminating matrix

The implementation preregistration should separate effects instead of attributing
every gain to asymmetry: group64 symmetric direct/progressive; rowwise symmetric
direct/progressive; rowwise asymmetric direct/progressive. The matched row arms
use the same endpoint selection, curvature normalization, damping, column order,
phase schedule and post-ReLU candidate inputs for WO. Direct and progressive
versions share initial row levels. No residual, optimizer or scale fitting after
observing development outcomes is part of this pure-code question.

If special-token calibration exclusion is added, predeclare an additional
matched symmetric/asymmetric progressive pair and its exact eligibility rule
before fitting. Natural evaluation must keep all fixed rows, including special
tokens. Eligibility cannot be inferred from hidden output or validation error;
input-token metadata must be qualified and retained first. If metadata or
minimum calibration coverage is inadequate, omit that arm prospectively with
a documented reason rather than silently choosing another filter.

Row maximum endpoints differ from group64 least-square scales. Compare the
matched row symmetric/asymmetric pair to isolate alphabet shape; use group64
arms as deployment baselines, not a causal asymmetry-only comparison. Keep
granularity and coefficient SSE descriptions beside every result. Do not
silently add clipping, a grid search, learned scales or coordinate passes.

## Before any numerical execution

Write and freeze a complete implementation/execution protocol: exact arm list,
endpoint handling including all-zero/one-sided rows, tie rules, Hessian formula
and normalization, damping floor, failure policy, seeds, actual operation
counters, packed codec and every stored byte. Prefer a hard numerical admission
failure to an unrecorded identity-Hessian fallback. If a fallback is studied,
it must be an explicit counted arm or qualification event, not a hidden success.

Reuse the existing four-expert natural split contract and document-cluster
metric conventions only after specifying their exact code identities. Seal all
artifacts before any numerical development/held-out reads. Preserve the1%RMS
and35%FP16 absolute screen; any relative improvement signal is secondary and
cannot promote a candidate failing the absolute goal. Existing consumed data
are diagnostic, not fresh final confirmation. Declare a finite fitting/worker/
audit/output stop and exact scalar independent codec/compensation controls.

The auditor must verify the actual deployed row levels, codes, original-context
identities, per-expert predictions and complete storage. Matching an unpacked
floating simulation without exporting asymmetric codes is insufficient. Native
asymmetric execution is a separate integration question after a quality pass;
two level magnitudes cannot be passed to a symmetric kernel unchanged.

## Bounded interpretation

If matched row asymmetry improves reconstruction but still misses the absolute
gate, record that mechanism signal without promotion. If all admitted formats
fail, the result closes this alphabet/granularity hypothesis at the declared
coverage and cost, not all asymmetric quantization. A positive expert result
justifies further whole-model/native evaluation. No full large-model QAT or
unlimited hyperparameter search is implied. Follow the review's finite closure
rule: justify any remaining high-information test or close the investigated
scope negatively when evidence and resource limits warrant it.
