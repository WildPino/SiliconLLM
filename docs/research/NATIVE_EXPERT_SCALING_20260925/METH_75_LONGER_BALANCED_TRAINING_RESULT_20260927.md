# METH-75: longer balanced training loses donor retention at both E

**Decision.** The fixed update-256 continuation of METH-71 fails its
new-prompt non-inferiority gate at both E128 and E1280. The E1280 bank
continues to distribute routes broadly and improves raw BPB, but its
donor-position top-1 falls further than the E128 control. Neither
checkpoint is eligible for a new external audit or native promotion.

The [protocol](METH_75_LONGER_BALANCED_TRAINING_PROTOCOL_20260927.md)
was committed at `c372c5d`. The
[manifest](meth75_longer_dev_manifest.json), SHA-256
`4d0fe7a52942e547864be5d04939aa2a8c3959736dcdbb0f942aacb20ffb0850`,
was committed before continuation. Its 24 prompt rows are excluded
from all old development sets, METH-70 raw training, and the entire
simulated METH-71 256-update raw draw stream (508 unique rows out of
512 draws). The [runner](../../../benchmarks/donor_adaptation/s1/meth75_balanced_continuation.py)
restores update-64 model, optimizer and RNG states; every actual raw
row/offset and chat draw at updates 65–256 is checked against that
frozen stream. Both arms use the same draws and unchanged METH-71
objective and hyperparameters.

| Fixed new-prompt measure | E128 | E1280 |
|---|---:|---:|
| Update-64 donor top-1, same new prompts | 3,717/3,807 = 97.636% | 3,746/3,807 = 98.398% |
| Update-256 donor top-1 | 3,677/3,807 = 96.585% | 3,676/3,807 = 96.559% |
| Decline versus same-arm update 64 | −1.051 points; **fail** ≤1-point loss | −1.839 points; **fail** |
| Raw BPB, update 64 → 256 | 0.968051 → 0.961239 | 0.969470 → 0.962478 |
| Raw ΔBPB against donor at 256 | −0.010017 | −0.008777 |
| Minimum changed B slots per layer | 128/128 | 1,280/1,280 |
| Minimum selected slots per layer | 128/128 | 1,197/1,280 |
| Worst maximum/mean route load | 4.35× | 22.78× |
| Peak allocated GPU | 2.581 GB | 10.231 GB |
| Full runtime including hash/save | 397.8 s | 508.3 s |

The [E128 result](meth75_e128_result.json) was committed at
`d073996` before the E1280 run. Its checkpoint SHA-256 is
`549572cd9bf60bf2e640976aad87c3895ad1e0b4f25f3a265b405353602f099d`.
The [E1280 result](meth75_e1280_result.json) was committed at
`6d8591d`, checkpoint SHA-256
`dc7518c84e29a5941cec19275fc970d7637e68b254a02beb268facdb592178e2`.
Both saved results contain every continuation update, microbatch
CE/KL/balance value, gradient range, per-prompt comparison and
per-layer route load. Exhaustive pair-score checks match product-key
top-four on evaluated prompts. No T4 was used.

The balance term remains active during continuation; for example,
E1280's summed axis-balance value on a raw microbatch is 14.05 at
update 65 and 4.41 on the first raw microbatch at update 256.
That observation does not isolate why top-1 declines. The same
continuation also improves raw BPB, so raw likelihood alone is a
poor substitute for donor-relative chat behavior. The next training
rule should be fixed before new evidence, explicitly control
retention as expert utility grows, and use fresh development and
external sources. The METH-71 update-64 checkpoint remains the
better of these two checkpoints on the new prompt set, but its
METH-72 and METH-74 promotion failures still stand.
