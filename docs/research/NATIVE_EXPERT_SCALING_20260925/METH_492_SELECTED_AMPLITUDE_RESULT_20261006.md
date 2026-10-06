# METH492: constant selected amplitude independently excluded

6 October2026. Goal ACTIVE/INCOMPLETE. Sole main/independent audit/finalizer
complete. ALL5 main/ALL8 audit/ALL5 final admission gates PASS. The uniform
constant-amplitude class fails on1499 development-exposed bank/expert cells.
No transformed model, task quality or new model speed is claimed.

## Question and exact certificate

Original Switch n128/12banks/D768; all1536 bank/expert slots. Source479
canonical pF32, UID/ownership/occurrence fields,159414development,
79458consumedval,238872UID and387036occurrences. Condition on each original
winner's own cell, unlike491's root aggregate. No model/FFN/score/softmax
replay, fitting or SVD. Candidate construction uses development only.

With rho=101/100 exactly, one real c preserves abs(log(c/p_i))<=log(rho)
uniformly iff c belongs to [p_max/rho,rho*p_min]. Thus:

```
ideal constant exists iff p_max/p_min <= 10201/10000.
min_c max_i abs(log(c/p_i)) = .5 log(p_max/p_min).
```

Extrema sourceF32 bits/earliest UID and exact rational endpoints are retained.
Two extreme states suffice to exclude the constant class on the entire cell.
No rounded log or average-loss surrogate enters this proof. This new full
selected-amplitude allowance does not alter any historical root gate.

For nonempty intervals the fixed candidate is the SMALLEST positive F32
inside, found through exact bisection. An unavailable scalar retains a
sentinel/explicit state, never a physical zero or a borrowed fallback.
All source p are valid; no clipping/epsilon/excluded source UID.

## Complete outcomes

| Bank | Ideal constant excluded | F32 constant eligible | No development cell |
| --- | ---: | ---: | ---: |
| 0 |125|1|2|
| 1 |128|0|0|
| 2 |125|3|0|
| 3 |123|2|3|
| 4 |124|3|1|
| 5 |115|9|4|
| 6 |123|4|1|
| 7 |127|1|0|
| 8 |127|0|1|
| 9 |128|0|0|
| 10 |127|0|1|
| 11 |127|0|1|
| TOTAL |1499|23|14|

1499 of1522 nonempty development cells are excluded. No ideal-eligible but
unrepresentable-F32 cell occurs in the donor; the fixed synthetic control
for that case passes. The23 eligible cells are all rare:16 have one
development UID,4 have two,2 have three,1 has four. They cover only34
development UIDs. Four of14 development-empty cells appear in validation.
Empty/unexposed cells are not verified useful experts.

| Role | UIDs | PASS | Source p below candidate range | Above range | No scalar | Invalid source p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development |159414|34|0|0|159380|0|
| Consumed validation |79458|3|10|17|79428|0|
| ALL |238872|37|10|17|238808|0|

Only64 UIDs have a development-selected scalar:34development/30validation.
27 of those30 validation states fail the fixed scalar's amplitude gate.
No scalar was fitted on the1499 excluded cells: their unavailable rows are
not counted as measured errors of an invented constant. The exclusion is
already certified by their exact development extrema.

## Independent validation

ALL1536 extrema, ties, rational intervals and scalar choices independently
recovered; all238872 saved UID source/candidate bits and states checked.
Source479 source-fields match every occurrence's original p bits, metadata,
canonical UID/bank/ID/ownership role/mode/accepted flags and occurrence counts.
All39unique/72occurrence/4608book/9216sourceID/144accepted reports, five state
counts and first UIDs rederived; each occurrence family retains387036 total.

Verifier imports no main/math helper. SourceF32 fractions are decoded through
struct/Python float.as_integer_ratio; scalar selection uses a nearest-F32
initial guess plus exact inequalities/predecessor, independently of main's
bisection. Every UID gate uses crossed rational inequalities, independently
of main's accepted-bit intervals. Eight fixed boundary/subnormal/empty/F32-gap
fixtures checked. No main/control/native/fit/SVD replay.

## Actual commands, costs and retained first faults

[Scientific protocol](METH_492_SELECTED_AMPLITUDE_PROTOCOL_20261006.md) frozen
50941db. First metadata patch context fault changed no files and is retained.
First builder failed before any numerical main/control: installed `_struct`
is builtin, not a separate DLLs/_struct.pyd. [Metadata-only R1](METH_492_BINDING_R1_PROTOCOL_20261006.md)
frozen f7d8ba9 binds actual python312.dll/builtin origin. Only operational
BIND path changed; main/math/verifier/finalizer/Windows recipes unchanged.

| Stage | Actual handle | Outcome/resources |
| --- | --- | --- |
| Original sole builder |95942f/session53639/c5b803|exit1;21.031s;37916672B; first fault retained |
| Sole R1 builder |aeac73/session12861/17960e|exit0;20.704s;38735872B |
| R1 binding freeze |6030d4e|fresh hashes/source/runtime binding |
| Sole main |3918c3/session95700/714362|exit0;27.313s;150077440B; native0 |
| Main actual Windows |ac9ee7|exit0; available/zero events |
| Sole independent audit |6e62fc/session72228/fa9e4a|exit0;27.797s;120582144B; native0 |
| Audit actual Windows |50f563|exit0; available/zero events |
| Sole finalizer |5cacff|exit0;ALL5 |
| Next source metadata only |146ffb|exit0;1.625s;25370624B; no vector calculation |

MainPID28796/create1791267482.0161233; auditPID21932/create1791267553.9281046.
Combined builders41.735s<=90s; R1 inherited remaining68.969s. Main180s256MiB,
audit300s256MiB; all resources pass. Main/audit fresh hashed bytes
9891491057/9891496153. Main source scans944240304B. No foreign-work wait;
CPU0/BLAS1, no new resources. Inquiry times include hashes/I/O/guards, not
model throughput or physical DRAM. No old scientific namespace rerun.

Main UID wire6688440B; output inventory6694043B; RAW6197944B. Both stages
remain within24MiB directory+8MiB document caps. The original protocol's
128KiB reserve wording is replaced in the actual frozen implementation by
checking the COMPLETE serialized document against8MiB, including terminal
metadata. No observed threshold changes. Guards remain active through the
write; separate terminal-resource receipts link RAW SHA and later wall/peak
snapshot. Their own small serialization follows that snapshot; the actual
terminal handle/last guards cover it. Reported resources use these later
receipts, not the earlier RAW resource snapshot.

## Identities, decision and next

- [RAW](meth492_selected_amplitude_result.json), SHA256 `bfb8d2af090e9bfc82601b77281e47cc37b973699507addc604e1e409e03e4ae`.
- [RETENTION](RETENTION_492_20261006.json), SHA256 `fe6c6c5a01027c524f513517ea6c7d5d30b11aa42c9405548fb1da2d4181c86d`.
- [ADMISSION](ADMISSION_492_20261006.json), SHA256 `368a74840f0f3c8d3e656c24cc73247ae4c0eb07e1806c6bc5abfd499f0e09c0`.
- [Binding](meth492_r1_binding.json), SHA256 `e1128e2a54874d41e956ff2edd0dca78dfff33530c93e55e26dce82a3fdb88b1`.
- [Completion/metadata registration](meth492_completion_registration.json) retains all actual handles and preserved bytes.

**Decision:** close uniform constant selected-amplitude preservation on this
development domain. This does not refute direct G_e approximating p_e F_e
with changed features/readouts, nor infer task loss when F is small/zero.
No useful retention or fresh generalization follows from the23 rare positives.
Do not sweep scalar budgets/IDs or promote a constant candidate.

[Whole algebra/next](METH_492_WHOLE_ALGEBRA_AND_NEXT_20261006.md) chooses direct
weighted-function transfer. [493 eligibility](METH_493_WEIGHTED_FUNCTION_TRANSFER_ELIGIBILITY_20261006.md)
and [fresh source metadata](meth493_weighted_target_metadata.json) identify
saved source vectors for the whole bank11 pilot; its compiler/learning/native
artifact remain unimplemented. Full goal ACTIVE/INCOMPLETE.
