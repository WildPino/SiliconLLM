# First broader-data attempt: terminal, preserved

9 October2026. Freeze b4da7b3f698adbbe0c856a3dfed4fe00d985e8da,
bindingf5de2bca,tool session8169 terminal exit1. Launcher26888/worker5460.
Original broad data adoption FAILED; there is no selected corpus/quality claim.

Both pinned shards downloaded and match their published LFS SHA/extents:
test105,418,621B in2.938s;train222,842,864B in6.984s. Reuse these files; no new
network acquisition is necessary. Original result namespace remains untouched.

Launcher stopped at11.219s because child Python spawned an unlisted conhost.exe.
Held worker peak34,471,936B/exit1. The original failure path killed only the
worker; the Arrow reader completed afterward without the worker's held exit
receipt. Its PID/through-exit peak are unavailable; do not invent those fields.
Process inventory afterward contains no Python/native/model/compiler job.

Reader progress records ALL54,948 test and115,991 train rows,all structures
eligible,2,369 train rows excluded by first-user/test grouping. It then FAILED
the test quota:everyday-conversations has only5 distinct first-user groups,
below8 required. Its first fault atphase select/28.125s is retained. The grouping
collapses dialogue histories sharing a generic greeting and does not provide
the intended dialogue identity. This is a selection design failure, not model
quality or proof of low dialogue capacity.

Byte archives of original worker/reader/launcher accompany the original binding.
Repair1 changes group identity to the normalized complete ordered user-turn
sequence,retains exact prefix identities/old-query exclusion/global selected
disjointness/all source quotas,allows ONLY pinned system conhost and ensures
observed own descendants are stopped on future launcher failure. It re-reads
existing Parquet bytes for a NEW grouping variable; no loss/model/network replay.
See [repair protocol](CHATBOT_BROAD_DATA_REPAIR1_PROTOCOL_20261009.md).
