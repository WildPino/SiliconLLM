# Next variable: exact wide-lane integer accumulation

14 October 2026. PROPOSED / NOT IMPLEMENTED / NOT MEASURED. The existing
24-layer cost now has full stored audit PASS and closure qualification:
26.97095 ms resident median against the 20 ms useful-50 budget.
The benchmark excludes generators, FFN,
attention and head, so faster recurrence is required but will not alone
prove the final architecture or useful rate.

The current accepted integer_readout.c reduces every 128-column chunk
horizontally into an int32, then extends it to int64. Each of 2048 output
rows performs 24 horizontal reductions. The scalar full-dot bound requires
int64, but this particular reduction frequency is not required by the
mathematics. Changing accumulation layout is a new exact implementation
variable, distinct from another failed precision/LUT/rank grid.

Proposed candidate: accumulate PMADDWD lanes over 1024 source columns,
then sign-extend each of the eight int32 lanes to int64 vectors. Sum the
three wide chunks and perform one final int64 horizontal reduction.
With abs(q)<=364, abs(r)<=32767, each int32 lane covers 128 scalar products:
128*364*32767 = 1,526,680,064 < 2^31. The full dot is bounded by
3072*364*32767 = 36,640,321,536 < 2^53, preserving exact F64 conversion.
These bounds apply to the admitted q/r domains, not arbitrary int16 values.
Use separate independent lane accumulators where useful, without float
accumulation or any change in ordered scale multiplications/BF rounding.

Before implementation verify the bounds as integers. Keep the original
readout source and qualified DLL untouched. A separate C translation unit
must use the unchanged state-row code and replace only the readout include,
with an explicit code identity check and source/DLL/compiler freeze.
No new learned weights, precision, routing, tokenization or source forward.

Qualification plan to freeze before timings:

1. Adversarial lane saturation/sign/tie/zero controls, including all-max q/r,
   against independent exact integer arithmetic. Reject on the first mismatch
   while preserving its receipt; do not relax any exactness criterion.
2. All24 canonical source packets and all frozen continuation intermediates:
   every dot/raw/BF exact to both original qualified readout and independent
   int64 reference. Full recurrence prefixes/state/rings likewise exact.
3. Paired original/candidate replay on the same bound four histories and
   positions; preserve sequential layer order, loaded owners and budgets.
   Alternate arm order under a frozen rule to reduce time-order confounding.
   Measure resident and parameter-evicted modes, component readout and whole24,
   wrapper overhead, full parameter bytes and CPU sharing. Do not reuse old
   timings as the denominator of a claimed speedup.
4. If exactness and physical benefit pass, verify all actual known-corpus
   bridge recurrent traces, not only four cost histories, before admitting
   a replacement in the quality bridge. Existing source GPU continuity gap
   remains; a future fresh end-to-end GPU quality run requires new binding.

Selection gates: all integer/raw/BF/state exactness mandatory; paired median
whole24 must improve by >=5% and no split/length stratum median regress by
>5% for an acceleration admission. Report whether whole24 crosses 20 ms
and how much budget still remains for excluded operators. A speed gain that
still exceeds 20 ms is useful implementation evidence, not goal completion.
Reject a faster microkernel if it worsens the whole24 workload.

If no substantial gain emerges, the active-byte/full-width budget remains
the obstacle: choose a donor-derived channel/layer/shared-core or conditional
generator change with own-model quality gates. Do not persist with minor
layout grids merely because they preserve bits. Useful conditional n,
compact core, same-artifact final CPU quality/useful50 and multiple donor
families/scales remain the goal; exact acceleration is an enabling step.
