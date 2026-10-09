# Missing-only delta seed completion after field-name check fault

9 October2026. Goal ACTIVE/INCOMPLETE. Original freeze83dd2b2/bind3499a1a0,
launcher31320/worker17984/create_time1791569625.5206437/exit1/session47670 CLOSED.
Held26.734s;worker10.266s/OS12587532288B/GPU622361088/650117120B. No child/update/
snapshot/native query. Five actual1507x1024 F32 postconv FIT fields are durable.
Five rank32 seed calculations passed buttheir U arrays/ledger were not serialized.

Fault: master verification compared `layers.0.organs.x_proj` against a dictionary
key `layers.0.x_proj`,thus wrongly required the intentionally seeded whole tensor
to equal its parent. A later updated-U check had the same namespace mismatch.
Preserve original worker/protocol/binding/log/firstfault/X namespace unchanged.

New standalone finish worker changes only these master-to-wire lookups and
adopts the five saved X fields instead of repeating GPU feature capture. It
loads actual26 (no prior optimizer update existed),checks all92 masters/Adam/RNG,
computes the missing five U arrays using exact same helper/parameters andsaved
X,then persists them andcontinues original selected initial native preservation/
one actualstep27/durability/export/three native endpoints. This is five repeated
closed-form seed solves required by missing arrays,not an inference/update replay;
do not claim bit identity to unsaved oldU arrays. Saved original scalar rank/cross
witnesses remain evidence ofold five successful solves only.

All original rank/energy/F32cross/norm limits andnative/update/quality scope
unchanged. No grid/random fallback/native-math change. All unchanged master
coordinates including primary x_proj rows must match actual26 bitwise after
seeding. Explicit new coordinate mapping permits only x_proj[16:48,:].
One actual fullFIT optimizer update maximum;partial/complete27 snapshot before
later evaluation andno restart/replay. New V columns nonzero gradient/change at
allfive sites;newU gradient0/rows exact infirst step withV0 expected.

New held family3560s/reserve300s;combined upper bound26.734+3560<3600s.
OS32GiB/GPU10allocated/11reservedGiB unchanged. New outputcap12GiB minusall6
old namespace file bytes;combined stored namespace<=12GiB. Logs4MiB/CPU settings/
child holding/allowed binaries same as [original protocol](ORIGINAL_DELTA_SEED_PROTOCOL_20261009.md).
Bindings include old45 inputs/failure metadata/log/allfive X/new code/protocol,
pre/post extents/SHA. Record both family costs andfirst fault;resource scope is
not retroactively a successful original exit. No source/RESERVED/T4/accepted-speed/
useful chatbot/useful-n/physicalDRAM/family admission.

Freeze new complete worker/binding/protocol before new observation:

```text
original_delta_seed_finish.py --bind --out <finish binding>
original_delta_seed_finish.py --launch --binding <finish binding> --binding-sha <SHA> --freeze <commit> --directory <fresh finish namespace> --out <finish result>
```
