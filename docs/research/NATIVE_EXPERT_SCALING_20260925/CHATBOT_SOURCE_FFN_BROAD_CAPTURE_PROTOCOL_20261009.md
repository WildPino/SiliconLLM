# Missing broad source FFN operands: fixed acquisition protocol

9 October 2026. Freeze before source calls. Previous turn is PROGRESS: completed
fixed local learning and saved mean/variation controls changed the next action.
Local absolute fidelity fails; 91-94% of final-site squared-error gain is mean
correction. The selected new variable is coverage across the retained 12 domains.

## Scope, reuse and decision

Pinned Falcon-H1-1.5B-Instruct revision80ebc50d/source SHA1fb78851, original
411 parameters/D2048/L24/FFN4608/BF16 casts/MUP/SiLU. Reuse qualified source SSD
temporary-storage schedule, chunk128 and eager attention. No optional kernels.
No deployment operator, active row count, original engine body or source weight
change. This is offline scaffolding toward faithful ternary functions.

Existing corpus48/8808 labels/24FIT+24DEV/12 domains, source replies/logits
already retained. Reuse four completed hidden cases/542 labels/8,880,128B with
their successful receipt and source-logit bit witness. Acquire NEW missing x/y
at sites0/23 on44 cases/8266 label positions, following corpus order. Force
original prompt/previous cached reply ID, observe only last-position FFN x/y.
Verify all newly obtained logits against retained BF16 bits and finite values.
Preserve IDs/absolute positions/dtypes/hashes and completed cases on fault.
Atomic payload files followed by atomic per-case receipt identify usable data;
an incomplete case is not admitted from orphan partial files.

Total raw hidden payload144,310,272B; NEW135,430,144B. New verified logit
coordinates541,728,842; old35,521,054 reused, total577,249,896. No reply
generation/optimizer/native/T4/RESERVED. Model parameter identities/versions
must remain unchanged; bound source file hashes checked before/after by launcher.

PASS requires all44 new +4 retained cases complete, exact expected byte/count
arithmetic, bit-exact source rows and resource/input gates. A source mismatch
stops immediately; retain first differing logits and completed packets.
Successful transport permits broader local function recovery; it is not another
quality observation or proof of conversion/capacity preservation.

## Price, caps and isolation

Reuse measured four-case capture100.047s family/91.391s worker and original broad
source1091.453s family. Noninitial longest capture case256 labels costs36-37s;
44 missing cases/8266 labels suggest roughly1200-1500s including prefills/import/
hashes. This is an estimate, not guaranteed throughput. Largest two contexts
already captured and reused; all new forcing lengths<=1268, original maximum1507.

Fixed family1800s, worker reserve60s, OS8GiB, GPU allocated10GiB/reserved11GiB,
outputs256MiB/log4MiB, worker cores0..5/launcher11. No concurrent Python/compiler/
engine job except the exact unrelated publisher exemption. Source/format/finite
checks every label; deadline/resource checks in worker and held-handle launcher.
Stop rather than enlarge caps on failure. No automatic restart/replay of completed
source cases; any repair binds retained case receipts and original failure first.

Bind all inherited source/runtime/foreign extents, 48 original logit packets,
16 retained hidden packets, completed original capture receipt, this protocol
and new worker. Shared launcher adds a distinct broad schema, preserving old
four-case validation; the old launcher extent is explicitly superseded only.
Original capture/calibration/recovery code remains immutable and reusable.
