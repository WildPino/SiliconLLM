# R4.1: pre-first-call resource observer cadence

Complete actual original-reference/native R4 binding is retained, SHA
d0542cc8a5b36a50f792bc3d79e19abeef86c160b30b38cd9d04b14f1513cb93,
676.016s/150,425,600B peak/56,558,893,434B hashed,2408 immutable native files.
No R4 model/quality/audit call has begun; no R4 output directory exists.

Static review: R4's additional combined-output inventory loop was accidentally
unthrottled in every hash-chunk/watchdog guard, unlike the original1s inventory
cadence. Enumerating2408 old native files per4MiB chunk adds unnecessary filesystem
work. Set additional output inventory to1s cadence; memory/wall guard/watchdog
remain0.1s. All file/peak/wall/quantity/scientific limits and thresholds unchanged.
No scientific parameter or learned result changes. Current binding path is fresh
meth506_r4_1_binding.json; original R4 binding remains immutable.

The pre-first-call metadata derivation replaces ONLY the operational wrapper SHA,
adds its source/supplement/source-binding record, and inherits ALL previously
actually hashed file identities. It does not claim to rehash bulk inputs itself.
The R4 quality/audit admission revalidates EVERY inherited catalog file before
numerical imports, exactly as before. No new benchmark/export/cohort/native replay.
