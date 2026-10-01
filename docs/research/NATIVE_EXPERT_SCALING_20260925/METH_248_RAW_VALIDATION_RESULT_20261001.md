# METH-248: continuous children fail consumed generalization too

Frozen at `e62664c`;session44196 completes,exit0. All176 original physical
coefficient/bias replays exact,raw fit SSE matches METH-244,all128 factored
per-window scores/energies exactly reproduce METH-247. Error-ledger
identities close;no new fit strength/rank/source/validation data.

| Consumed normalized SSE | Raw FP64 readout | Actual rank32 factors |
| --- | ---: | ---: |
| E16 |**.00009494904**|.0001176413|
| E160 |.0001213229|.0001202956|
| Rotated E160 |.0001680529|.0001354738|

Raw E160 loses **27.7768%** to raw E16,despite49.53% fit gain. Raw paired
P05/P95 **-2.709631e-5/-2.567049e-5**. Absolute1% and rotated10% pass;
E16 useful-count/positive gain fail. Thus correcting final readout encoding
alone cannot solve this frozen recipe's consumed count loss. Rank32
truncation slightly improves child error while harming parent error here;
it masks some raw-child loss rather than causing the whole gap.

Ledger normalized distortion/cross term:
E16 **2.314958e-5/-4.573646e-7**,
E160 **4.284588e-5/-4.387310e-5**,
rotated **6.507705e-5/-9.765611e-5**. Maximum closure1.785e-16. Distortion
is not an additive loss in isolation;its error cross term matters.

**Decision:** change continuous function/hierarchy before codec-only count
attempts. This does not prove data overfit is the sole cause:raw children
still inherit **quantized source-projected priors compiled from stored
parents**;raw parent final readouts are unencoded. The diagnosis removes
final output factorization,not every ancestral quantization/source reset.
Do not call these fully FP64 ancestors or the original donor.

[Raw result](meth248_raw_validation_result.json),SHA256
`dda4fc408cf4b7b51fa2909d0e5d8cec9df999a227266d4823b3ff4b26315e7a`.
Runtime74.110s,RSS4,573,847,552bytes,GPU peak3,844,792,320bytes,local3060/
six threads,no T4. All176 fit replays and128 windows/ledgers retained.
No new checkpoint,native-bank promotion,fresh/full quality or accepted rate.

Next: [parent-anchored latent hierarchy](PARENT_ANCHORED_LATENT_HIERARCHY_PROPOSAL_20261001.md).
Protect the already learned parent function across inputs,not only its
value at a child center;first a continuous diagnostic,then separately
price/encode any useful private correction. No rank/tau retry.
