# Deferred idea: adaptive union of exact neuron certificates

PROPOSED/DEFERRED. No observation, index export or native call. Goal ACTIVE/INCOMPLETE.
Written before506's donor result; deferred after its14/15 quality outcome.
An exact acceleration preserves I8 source IDs and cannot repair the failed
donor-relative prose-edit criterion. First diagnose this information loss.
This proposal changes
certificate granularity using retained source information, without new centres,
mask widths, fitted functions or another kernel grid.

## Why this question follows the information evidence

500..503 finite overlapping masks omitted nonzero source terms on new inputs.
504's637 source-derived centres provided sound certificates, but one fixed
1536-neuron mask had to certify EVERY omitted row simultaneously; natural
coverage was6.394780%.505 and506 instead consult actual hidden zeros after
computing complete source WI. The next uncertainty is whether individual row
certificates, combined across the SAME existing centres, avoid enough WI work.
This is not a rerun of504's half-width regional recipe or a larger centre grid.

For each selected original parent, keep the complete source WI/WO fallback and
all original scales. A certified zero row can be omitted; every uncertain row
is computed exactly. This preserves the source function even outside the data
distribution. Coverage and cost can fail without losing that guarantee.

## Exact integer geometry

Original WI input q and retained centre a are768-dimensional signed A16 codes;
w is one original signed I8 WI row. Define integer quantities

```
A=a.a, Q=q.q, t=a.q, R=w.w, d=w.a, Delta=A*Q-t*t >=0.
```

For A>0, real orthogonal decomposition along a gives

```
w.q = d*t/A + w_perp.q_perp,
|w_perp.q_perp| <= sqrt((A*R-d*d)*(A*Q-t*t))/A.
```

If d and t have opposite signs and `d*d*Q > R*Delta`, then w.q is strictly
negative. Expanding the squared upper-bound condition cancels d*d*t*t and
gives precisely this predicate. No estimated sign or statistical assumption.
Zero q or zero w yields zero exactly. Equalities are left uncertain; all
other cases retain original evaluation. Source positive row scales/A16 alpha
preserve the sign; original ReLU replaces the negative result with+0.

For each query, union the rows certified by ANY retained centre of its parent.
The union remains sound because every member has at least one sound witness.
All nonzero original hidden values therefore remain, retaining the original
hidden maximum/A16 alpha/codes and exact zero-only column WO from505/506.
No copied child enters the parent softmax; original winner and normalized mass
remain explicit original costs.

All products fit unsigned128bits: A,Q<2^40, R<2^24, d*d<2^64,
R*Delta and d*d*Q<2^104. The preliminary implementation must use Python integers,
not overflowing NumPy I64 products. A prospective C path uses explicit128bit
comparison. Centre-query dot and norms themselves fit signedI64. General floating
source rows/other activations do not inherit this integer/ReLU proof.

## Stored redundancy turns predicates into an index

For each existing centre, sort negative-d rows and positive-d rows separately by
the exact rational ratio d*d/R. Compare ratios by integer cross products;
ties use original neuron ID. For t>0 search the negative list; t<0 the positive
list. The true predicate occupies a suffix, found with exact binary search.

Store a3072-bit suffix mask at EVERY64 list positions, plus sorted neuron IDs
and squared dots. Read the checkpoint at the next64 boundary after the cut,
then add at most63 individually listed certified rows. OR masks across centres.
No unproved row enters a checkpoint read. Keep a separate unconditional mask
for R=0.64 is a fixed storage/index layout choice, not a tuned scientific width.

Redundant suffix masks store repeated zero-membership information. They add no
donor capacity, but may reduce active computation. Memory, comparisons, centre
dot/norm work, bitmap reads/unions, remaining WI rows/scales and complete original
fallback weights must all be charged. Selection of the parent is not free.

## One bounded preliminary screen to implement and freeze

Use ALL original128 parent slots and ALL17540 qualified bank11 inputs/UID roles,
including11721 development and5819 consumed. Reuse504's637 centres, their actual
source row dots/norms and prior admitted source-hidden/code arrays as immutable
data. Fit no new centres and acquire no new source functions. Empty parent0 stays
stored with complete fallback and has no empirical coverage claim.

1. Freeze exact source/input/centre joins, sorting, encoding, controls, work
   decisions and a small actual dependency closure before numerical execution.
   Bind only actual dependencies of this screen; no compiler/Torch/model call.
2. Build ONE physical index. Byte-reconstruct all sorted lists/checkpoint masks;
   retain per-parent source contribution and explicit storage size.
3. Compute all query cuts/unions with exact predicates. Retain ALL certificates,
   chosen masks/remaining widths and work counts. Every omitted source hidden
   value/code must equal zero; all retained values/scales stay identical.
4. Independently reconstruct centre-query dots/norms, exact sorting/cuts/unions,
   physical encoding and all finite-domain work summaries. No main imports,
   source model call, full source-function replay or native timing.
5. Report development, consumed, natural, teacher, per-expert/rare and every-book
   views, retaining every fallback. Consumed data are a diagnostic domain, not
   fresh donor-relative quality.

Primary proposed screening decisions, to freeze BEFORE any observations: consumed
and natural mean addressed WI coefficient+centre/index bytes <=.75 of original
WI bytes; natural p95<=1; every consumed book mean<=1; full original bank plus
index<=1.20 of original bank bytes. This is a preliminary information/work
criterion, not a latency prediction. Also report exact scalar comparison and
bitmap work; do not turn dot-count ratios into token/s.

Price main/audit <=180s each, <=2GiB host peak and <=128MiB new outputs, using
existing local assets. Determine exact output/dependency maxima before freeze.
Stop at first arithmetic/data fault; preserve it before a numbered repair.
Economic failure is a complete negative result: close this union/index recipe
before C or fresh whole-model work. No centre/radius/checkpoint/threshold grid.

If work passes, price/freeze a separate actual C function-and-cost test with
continuous foreign-process identity logging; only then consider new whole-model
fresh quality on the same artifact. Existing506 timings cannot supply its rate.
Useful larger n, CPU LUT parent winner AND mass, real DRAM and actual additional
families/scales remain independent goal requirements.
