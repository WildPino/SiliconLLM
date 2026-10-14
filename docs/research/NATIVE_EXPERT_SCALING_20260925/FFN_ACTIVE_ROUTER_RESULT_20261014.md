# Source remainder-sensitive routing: qualified numerical failure

14 October 2026. **COMPLETE / full numeric audit PASS / no admitted FIT-selected
candidate. Goal ACTIVE / INCOMPLETE.** The prior interrupted audit observation
is retained; no exit code or cause has been invented for that worker.

The resumed audit reused the existing producer and frozen binding. It did not
repeat fitting, predictions, source queries, GPU work or RESERVED access.
All 576 predictions, exact projected-only neuron IDs, cell IDs, unit masses,
144 source-proxy controls, FIT derivative coefficients, direct derivative
Gram, finite differences, input moments/covariance, eigen witnesses, balanced
metric, biorthogonal factors and case/domain/selection metadata passed.
Before/after input and output hashes passed. Maximum F32 prediction discrepancy
was 2.638920094526002e-8; maximum duality discrepancy 4.4968791045055065e-15.

The selector changes the input geometry from energy PCA to a FIT source
remainder-derivative metric balanced by the input second moment. This metric
is a per-neuron weighted derivative surrogate, not the score gradient or the
joint output-loss geometry. Full source projections are evaluated only after
IDs have been fixed, as part of the diagnostic reference evaluator.

| Site | Selected count | FIT-selected rank | FIT mean / worst | DEV mean / worst | Decision |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 512 | 512 | .098412530 / .127617496 | .137283700 / .178026976 | Selected candidate fails DEV mean |
| 12 | 2048 | none | rank 512: .124525 / .140929 | rank 512: .186774 / .239840 | No tested rank passes FIT |
| 23 | 32 | 64 | .095835675 / .187636042 | .105606825 / .172211771 | Selected candidate fails DEV mean |

The smallest FIT-eligible rank rule and mean <=.10 / worst <=.20 / DEV domain
<=.20 gates remain unchanged. Site 23 ranks 256/512 pass local DEV curves but
cannot be promoted through DEV. Rank 128 DEV .10002605329889486 fails .10.
All 2048 input directions survive the frozen relative cutoff. Support condition
numbers are approximately 1.6414e8 / 4.0234e6 / 2.7156e9 at sites 0/12/23.
F64 witness success does not establish deployable quantized factor precision.

Compared with PCA at rank 512, DEV means improve from .169438/.224801 to
.137284/.186774 at sites 0/12; at site 23 the FIT-selected rank 64 improves
.110001 to .105607. This qualifies the geometric improvement and the remaining
selection/generalization failure within this fixed recipe. It neither rejects
all conditional encoders nor admits a cheap selector or whole-model replacement.

Source freeze b34cd94; binding 58c33ec. Producer exit 0 / 595.687 s / OS peak
1,933,545,472 bytes. Resumed audit PID 21332, creation 1791965189.9385552,
exit 0 / 645.781 s / OS peak 2,309,419,008 bytes. Every declared input/output
pin, protected file, command receipt and process identity was checked again
in closure. The E family contains 579 files / 3,460,561,182 bytes. Inherited
four unused NVIDIA pin changes remain explicitly recorded; no current GPU
continuity claim is made.

- [Protocol](FFN_ACTIVE_ROUTER_PROTOCOL_20261014.md), [binding](ffn_active_router_binding_20261014.json).
- [Producer](ffn_active_router_result_20261014.json), SHA256 aef9eccc6b4fa2f2e66820e0fb4288000eb4f368b5e4f23b2bbdb04d61015189.
- [Complete audit](ffn_active_router_adjudication_20261014.json), SHA256 26a60b0ae635c9028c305735e5021365a594dda6730b83c29b1f3395d36ef731.
- [Closure](ffn_active_router_session_closure_20261014.json), SHA256 7a860566ff5cdc7b918480c4de424c44ac89cea89c2bd24ffd11c1f89eec0272.

Next measure the actual 24-layer recurrent backbone under the frozen
[cost protocol](NATIVE_24_COST_PROTOCOL_20261014.md), before further rank/cell
expansion. Its full-width known-corpus quality bridge is the reusable control.
The final CPU engine, same-artifact useful >=50 tokens/s, actual DRAM, useful
conditional capacity growth, multiple families/scales and sustainable
conversion costs are all still incomplete.
