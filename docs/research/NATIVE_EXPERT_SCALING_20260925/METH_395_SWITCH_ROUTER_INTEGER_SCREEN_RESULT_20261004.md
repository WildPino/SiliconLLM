# METH-395: full-dimensional I8/A16 router/refinement grid rejected

Frozend46fddd; physical source/capture/prior/export preflight PASS. Command exit0
fully consumed;6.328s/max RSS318,308,352B,3,569,568 retained code/scale bytes.
ALL5 apparatus gates PASS:384 trace identities/shapes,24 actual source F32
router segments/full manifest, independent full-weight score/selected-ID
controls, I8/F32 representation roundtrips, all Python integer-dot controls.
Original payload hashes inherited393; actual router/source segment hashes fresh,
payload size/mtime unchanged. NumPy2.4.6/OpenBLAS1 recorded with library SHA.

Fixed shortlist0/1/4/8/16, original full-dimensional row-I8/A16 scores, exact
candidate refinement plus ALL-candidate mixed normalization. No rank truncation
or training. None passes ALL24 bank/modes on either independently pretrained
source under unchanged394 local identity/probability bounds. Byte proxy<=half
passes for every variant; coefficient work1+m/n is NOT reduced.

| n /m16 | Passing bank/modes | Selected-ID mismatches | Worst bank/mode probability p95 /max | Weight-byte ratio |
| --- | ---: | ---: | ---: | ---: |
|256 |2/24 |574 |2.52% /14.89% |.313802 |
|128 |22/24 |212 |.83% /1.42% |.376302 |

At128 only decoder.block.1 router teacher/natural identity fails m16:113/99
mismatches; all probability bounds pass. At25616/24 fail identity,13/24 p95
and4/24 maximum probability. Fixed m8 at128 passes16/24; m4 passes10/24.
No pooled override or threshold/grid adjustment. Full bank/mode/m0 controls
and errors in raw. Local original selected identity is sensitive even when
selected probabilities remain close; changed routes cannot inherit quality.
This does not measure downstream harm or reject every mixed-precision scheme.

Raw `meth395_switch_router_integer_screen_result.json`, SHA256
`e1a9b0fbbb99c5797ca16b8e7fffc48b3079cabe50812bdbc41a84cdf9143eaf`.
20min/8GiB guards held, no model/native timing/download/GPU/T4 overlap.
Original F32 routers remain stored with additional I8 codes/scales. Addressed
byte proxy excludes inputs/normalization/selection/dispatch and is not physical
DRAM or actual speed. Integer mathematical dot bounds and356 arithmetic are
reused, not a new C numerical/whole-quality/rate qualification.

Decision: close unchanged fixed full-dimensional I8/A16/refinement grid; no
native quality/rate promotion. Adaptive certified selection, per-bank fallback,
input-aware fitting or shared-vector centering would be NEW variables requiring
prospective protocols and NEW whole quality, not repairs to this result. Keep
qualified original full-F32 routers and374/389 binaries/rates authoritative.
Next prioritize an actually pretrained source with>256 choices/applicability
and source-specific normalization, rather than polishing this same-bank router
indefinitely. Actual n/DRAM/LUT/family/~100B goal still open.
