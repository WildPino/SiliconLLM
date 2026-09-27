# METH-84: viewed-set single-FFN-layer precision rescue diagnostic

**Question.** METH-83's grouped-Q4 FFN core has only 68.469% donor
prompt top-1 with the E128 adapter, versus 96.332% for the BF16
composition. METH-81's exact ledger leaves 24,260,096 bytes below
the ideal 560 MB allotment. Restoring all three FFN matrices of one
layer to BF16 costs 19,203,072 additional ideal bytes, yielding
554,942,976 bytes/token. Test whether this narrowly affordable
precision exception could rescue prompt ranking.

**Fixed diagnostic.** Use the now-viewed METH-83 24 prompts and exact
donor, METH-56 E128 adapter, METH-82 Q4 core and hashes. First
reproduce 4,045/4,199 BF16+E128 and 2,875/4,199 Q4+E128 prompt
matches against the BF16 donor. Starting from the Q4 model, restore
the donor's original BF16 gate/up/down FFN matrices in exactly one
layer at a time, in increasing layer order 0–23, score all prompts,
then restore that layer's Q4 reconstruction before proceeding.
Record each layer's matched positions, payload charge and difference
from the Q4 baseline. No optimizer, new data, or test-set query.

**Decision boundary.** If any single-layer arm reaches ≥95% pooled
donor top-1 within ≤560,000,000 ideal bytes, it becomes a candidate
for a separately frozen, new-source quality test. Otherwise reject
single-full-layer BF16 restoration as a sufficient repair on this
viewed set. These 24 results are a mechanism diagnostic and cannot
promote any checkpoint or reverse METH-83's failure, regardless of
outcome. Report every arm rather than only the best. The result says
nothing about multi-layer row selection, a different Q4 rule, native
speed, or large-E quality.

Local RTX 3060 only; ≤10.5 GiB GPU allocation, ≤20 GiB RSS and
≤5 minutes. No T4.
