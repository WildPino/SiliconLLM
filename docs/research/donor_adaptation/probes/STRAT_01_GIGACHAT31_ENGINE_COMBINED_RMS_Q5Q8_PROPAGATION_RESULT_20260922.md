# STRAT-01 GigaChat 3.1 combined RMSNorm + K-B propagation result

**Result:** `COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES`

**Scope:** block-0 composition of the already measured double-RMSNorm and
Q5_0×Q8_0 K-B runtime semantics; frozen `prefill8` and `cached7p1` schedules;
no production, Rung 2C, later-layer, quality, generation, RAM, or rate claim.

## Canonical evidence

- Accepted artifact: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Captured candidate:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_combined_rms_q5q8_20260921/`.
- Canonical offline adjudication:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_combined_rms_q5q8_offline_adjudication_repair1_20260922/`.
- Adjudication SHA-256:
  `0a0aac6a58fe4bbf37d414d80d5681afdf357c7fda846596bea10685092c1b77`.
- Run-manifest SHA-256:
  `8c2bc3f03d4755dbc1de8548211bdf0e8a9ac792d4905aa2232ffd1a113f7869`.
- Execution accounting: one source candidate donor graph, zero new donor
  graphs, zero new reference graphs, `errors=[]`, and no reference producer
  rerun.

## Result

All 24 Rung-2A tensor comparisons, all three cache checkpoints, all twelve
prefill-versus-cached continuity checks, and all six dense-FFN comparisons pass
their frozen gates in both schedules. The failure list is empty.

The combined path preserves the frozen K-B Q8_0 and absorbed-Q hashes in both
schedules. `q_nope_absorbed_perm-0` reaches NRMSE
`4.802399901907019e-8` and normalized maximum
`8.816135785418076e-8`, compared with upstream-double NRMSE
`3.1511401613621e-4`. Downstream `kqv_out-0` is
`1.4223848767537198e-7`, and terminal `ffn_inp-0` is
`1.6235944693498375e-5`, comfortably inside its tighter `0.001 / 0.005`
gate.

The decisive dense projections now pass:

| tensor | combined NRMSE | combined normalized max | upstream-double NRMSE | accepted-float NRMSE |
|---|---:|---:|---:|---:|
| `ffn_norm-0` | `2.3612375112384717e-5` | `3.108848345891282e-5` | `6.922862039534381e-4` | `7.914143404266998e-4` |
| `ffn_up-0` | `4.02014425233629e-4` | `4.2690695118983e-4` | `2.6264924085599465e-3` | `3.0612619849613985e-3` |
| `ffn_gate-0` | `3.4697064936483897e-4` | `2.951011547638659e-4` | `2.3147870586418297e-3` | `2.6914051074363363e-3` |

Both schedules have the same values. Cache NRMSE is
`2.027999173672203e-4` for both final checkpoints and
`2.0451901014432078e-4` for the cached prefix; all pass. Every continuity
NRMSE and normalized maximum is exactly zero.

The Q8 census confirms that the changed semantics are live: attention changes
45/48 blocks and 45/14,016 bytes, while final FFN normalization changes 46/48
blocks and 372/14,016 bytes. All planted/source controls pass, including
swapped-target rejection, the two upstream double sites, the one final double
site, and the combined K-B sites.

## Preserved VOID chain

The source candidate completed but its original runner record is VOID because
it compared the accepted upstream baseline to the current, necessarily changed
`engine.c` hash. That record remains immutable (adjudication
`8eca798bb38706b2d9f0a56ebebf5934f035200c5ae0e983ae4e801f56219e66`,
manifest `4964efe45dbbde5b7fd80d957b25279c8099f267c6b40e2aac573858d232db5e`).

The first offline attempt is also preserved as VOID: it passed the historical
source check but unpacked `(tensors, metadata, cache)` as if metadata were the
cache, producing `KeyError: 'cached7p1/final'`. It executed no graph
(adjudication
`7e6fc537a177334d4588e01545d06d00378a89a3874d7c1e0c82609f93abb03a`,
manifest `c2f638010b7b39763bfd7667271b6926e48741204351c43ae6abe9ff2f9b1dc0`).
The canonical offline record binds both VOID stages and corrects only those
adjudicator defects; it does not recalculate candidate tensors.

## Interpretation and next gate

The measured block-0 failure was compositional: neither double-RMSNorm nor
K-B Q5_0×Q8_0 alone closed the dense projection gates, while their composition
does. This closes the diagnostic propagation coordinate and forbids repeating
the RMSNorm, K-B operator, or combined cells.

The next distinct coordinate is a separately frozen production integration and
confirmation of these exact semantics in `engine.c`. It must preserve the
accepted attention/V-B repairs, validate the full block-0 path against the
existing immutable references, and make no Rung-2C or end-to-end claim until
that changed production path passes. No `SPEED_LEDGER.md` entry is due.
