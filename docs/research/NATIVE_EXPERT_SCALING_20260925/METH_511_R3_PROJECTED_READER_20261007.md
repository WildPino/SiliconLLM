# METH511-R3: bounded projected source reader

R2 cohort first query failed before returning any row or constructing any case:
DuckDB internal512MB limit cannot allocate a second256MiB sort/string buffer.
Actual process5.5s/225,603,584B peak, zero model/native calls. RAW/fatal/progress
and actual tool exit1 preserved; no completed model/whole observation to replay.

R3 computes physical row metadata (count/min/max/distinct==1243/0/1242/1243)
without reading the text column into a sort, then projects text only for the
unchanged first96 ranked candidates plus two fixed consumed tokenizer controls.
Order by physical row applies only to this bounded candidate pool. Reader/
auditor internal memory setting1536MiB fits inside the unchanged2GiB cohort/
4GiB audit OS bounds; actual watchdog peak/time/output limits remain required.
Every returned row must belong to the predetermined set exactly once. Selection
eligibility/seed/first24/4096chars/four32token windows/all model IDs unchanged.
All1243 physical IDs are independently checked by aggregate metadata; source
text is not unnecessarily materialized for unselected books. No model scores.

Metadata rebind reuses the completed R2 full input SHA receipts after retained
Win32 read/share-read handles and current stat equality; it never calls them a
new full SHA pass. Freshly hash changed committed R3 sources/derivation/protocol/
original first-fault RAW. Original donor ZIP checksum remains explicitly retained
with ALL3320 actual canonical tensor hashes before first inference. Same native
binaries/payloads/environment/k8/576 calls,384 donor wrappers, all quality/
economics criteria, accepted numerators/phase bounds/18GiB outputs. Source reader
operation repair does not change the scientific comparison.

Rebind<=180s/512MiB, zero scientific/model/native/compiler/cohort queries. All
original/R1/R2 faults remain; only previously uncompleted cohort and never-started
native/donor/audit stages run in R3. Goal ACTIVE/INCOMPLETE.
