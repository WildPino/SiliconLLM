# METH-67: the METH-66 prompt set alone does not explain the E128 failure

**Decision:** the new CPU-offloaded METH-66 E128 recipe needs an
initialization/optimizer parity check against the earlier E128 recipe.
The previously frozen METH-55 and METH-56 checkpoints both clear 95%
on the same now-viewed METH-66 prompt set, whereas METH-66 E128 did
not. This diagnostic cannot promote any checkpoint or reopen the
METH-66 stopped gate.

The [protocol](METH_67_VIEWED_DEV_RECIPE_DIAGNOSTIC_PROTOCOL_20260927.md)
and [runner](../../../benchmarks/donor_adaptation/s1/meth67_viewed_recipe_diagnostic.py)
were fixed before the read-only evaluation. The
[raw result](meth67_viewed_recipe_diagnostic_result.json), SHA-256
`77f8646fe72b1437a5b6d8219881b5bff16a4f5d247978d5f07d1c5e2b278e03`,
binds the original checkpoint hashes and frozen METH-66 development
manifest.

| Checkpoint on viewed METH-66 prompts | Donor top-1 matches / 3,810 | Raw ΔBPB | Worst route load |
|---|---:|---:|---:|
| Original METH-55 E128 update 16 | 3,669 = 96.299% | −0.001552 | 21.86× |
| Original METH-56 E128 update 512 | 3,678 = 96.535% | −0.014485 | 15.74× |
| New METH-66 E128 update 16 | 3,577 = 93.885% | −0.001879 | 22.02× |
| New METH-66 E1280 update 16 | 3,615 = 94.882% | −0.002020 | 71.06× |

Both historical checkpoints match exhaustive product-key routing on
the viewed inputs. Their result shows that this prompt set can support
≥95% for E128; the new offload recipe changed initialization,
optimizer behavior and the training draw schedule/raw exclusion pool.
This comparison cannot identify which change caused the gap. The next
method step is to reproduce METH-55 E128 initialization and one
dense-AdamW update on the *same* sample sequence while keeping factor
storage/selected-row transfers in CPU RAM, then test larger E under a
newly frozen development set. This viewed set is only diagnostic.
