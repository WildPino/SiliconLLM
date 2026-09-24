# STRAT-01 GigaChat 3.1 layer-2 KV RMSNorm offline-recovery protocol

**State:** FROZEN BEFORE RECOVERY IMPLEMENTATION OR EXECUTION

## Why the scientific record is VOID

The sole scientific invocation completed all six C arms and wrote a complete
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION` report, but the Python
adjudicator ended `VOID_LAYER2_KV_RMSNORM_CROSS_INPUT` with:

`unexpected ValueError: cannot reshape array of size 4096 into shape (8,6144)`

The defect occurs only while constructing descriptive metrics for a normalized
prefix shaped `[8,512]`: it reused the downstream helper whose per-token view
is hard-coded to `[8,6144]`. Downstream judgments, exact replay checks,
production-C replay, frozen-metric replay, planted controls, input mutations,
and label-swap rejection execute before that final dictionary expression.

The raw record remains immutable and VOID at SHA-256
`2c628c3a9503cf882c2e6ce42a6c8211a5767b6269dc0694b148623da383aaaf`:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_20260924/`

It records one diagnostic invocation, zero donor/reference graphs, commit
`a1ee4f61f8b10f763108f4e37830e642f7b859f6`, and C binary SHA-256
`8aa61ddb65aae9f4a1a8ee7c8009cee8609d56b956c8faf009ec7f5085ca8235`.

## Authorized repair

Create a separate offline recovery that:

1. binds the raw adjudication and every report/output hash without modifying
   the raw directory;
2. requires the exact raw status, error, commit, binary, invocation count,
   zero-graph accounting, report state, inputs, descriptors, and arm labels;
3. reruns the original downstream adjudication unchanged;
4. computes prefix metrics with the same aggregate formula and an `[8,512]`
   per-token reshape, for description only;
5. reruns every exact replay, frozen metric, planted control, mutation, and
   label-swap check;
6. writes a new `offline_recovery1` adjudication with zero new diagnostic
   invocations, zero model access, and zero graphs.

No C binary, GGUF parser, model weight, donor/reference producer, or output arm
may execute. Commit the recovery source and tests before the one offline run.

## Frozen facts the recovery must verify

- captured and computed reference prefixes both have SHA-256
  `0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91`;
- their downstream outputs both have SHA-256
  `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e`;
- captured and computed C prefixes both have SHA-256
  `e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9`;
- their downstream outputs both have SHA-256
  `81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`;
- the two control downstream hashes are
  `d88cfb70238acd7ad88a19464a1e2f6c8c3078490134b61c75726736a518c6db`
  and `58af021a2c2dce43cb50c9ba2a0266d374cb9268553d92f2a1659597c302c86f`.

## Decision rule

Apply the original frozen decision rule without alteration:

- `LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT_RECOVERED` if computed
  reference input fails downstream;
- `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED` if computed
  reference input passes and computed C input fails;
- `VOID_LAYER2_KV_RMSNORM_OFFLINE_RECOVERY` for any identity, provenance,
  accounting, replay, metric, control, or schema failure.

The `_RECOVERED` suffix distinguishes the canonical offline adjudication from
the immutable raw VOID. Prefix metrics cannot affect the decision.

## Non-claims

Recovery does not add evidence arms, repair layer 2, execute KV-A projection,
or make quality, RAM, rate, tokenizer, generation, or later-layer claims.
