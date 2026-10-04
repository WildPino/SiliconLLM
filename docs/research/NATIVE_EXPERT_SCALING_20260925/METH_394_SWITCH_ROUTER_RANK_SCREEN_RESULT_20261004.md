# METH-394: source-only low-rank router/refinement grid rejected

Frozenf73fd9d, committed controller/protocol/capture/export preflight PASS.
Command exit0 fully consumed,12.344s/max RSS311,562,240B,11,107,776 retained
F32 factor bytes. NumPy2.4.6/OpenBLAS1 actual thread readback/library SHA.
ALL4 apparatus gates PASS:384 trace identities/shapes,24 source F32 router
segments/full manifest, independent full-weight score/selected-ID controls,
all source-only F32 SVD factor roundtrips. Complete payload hashes inherited393;
actual router segment/source hashes fresh, original payload size/mtime stable.

Fixed ranks8/16/32/64 x candidate shortlists1/4/8/16, EACH twelve banks and BOTH
teacher/natural modes, on both independent original sources. Full approximate
score normalization retains ALL candidates after exact shortlist refinement.
None of16 fixed variants qualifies on either source; every variant has zero
passing bank/modes under the frozen joint local criteria. No postscore grid,
threshold, pooled-bank override or fitting to responses.

| Source /r64,m16 | Identity agreement range across bank/modes | Absolute relative selected-probability p95 range |
| --- | ---: | ---: |
|256 |27.51%-89.51% |33.01%-72.34% |
|128 |63.02%-100% |4.56%-23.77% |

At256 r64,m16 ALL24 bank/modes fail identity, p95<=1% and max<=5%; coefficient
proxy passes. At12822/24 fail identity,ALL24 fail probability limits and<=half
addressed coefficient proxy. Even retained source-weight energy73.37%-96.88%
at256 and87.48%-98.28% at128 does not preserve competition/normalization on
actual inputs. These are local diagnostics on consumed cohorts, not complete
changed-model quality or downstream harm; factorization was of the classifier,
not a low-dimensional cap on expert outputs.

Raw `meth394_switch_router_rank_screen_result.json`, SHA256
`b8719b4d45bf0b814499111084de5daadea5bdb8cc8990bb044735f1b47ffe11`.
20min/8GiB guards held, no overlap/native timing/original model/download/GPU/T4.
Source-only F32 factors and full original routers remain separately counted;
coefficient/byte proxies are not measured cache/DRAM or rate.

Decision: close this unchanged source-only low-rank/refinement grid. An input-
aware trained factorization would be a new variable, not licensed by energy
retention. Next consider preserving full-dimensional scores with the already
independently verified I8/A16 primitive, plus exact candidate refinement. That
changes precision/byte traffic rather than rank or normalization domain; it
requires new local gates, native math and NEW whole-quality/rate acceptance.
No new useful choices/LUT/physical DRAM/cross-family/~100B qualification.
