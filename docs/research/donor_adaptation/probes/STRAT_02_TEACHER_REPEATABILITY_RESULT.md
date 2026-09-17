# STRAT-02 F32 teacher repeatability — result

**17 September 2026. Outcome: `PASS_REPEATABILITY`.** The
[frozen brief](../briefs/BRIEF_STRAT_02_TEACHER_REPEATABILITY.md) required
one fixed calibration document, two consecutive full F32 scores, and two
first-64-token logit forwards before the W4-v2 gate. The raw write-once run
directory is
[`strat02_teacher_repeatability_20260917_142728`](../../../../benchmarks/donor_adaptation/density/results/strat02_teacher_repeatability_20260917_142728/).

The independent audit found `supervisor_result.json` with
`PASS_REPEATABILITY`, `exit_code=0`, and 25 resource samples. The worker
recorded one verification pass over all 11 pinned shards, with every shard
hash matched; the verified shard names, sizes, and mtimes remain in
`worker_result.json`. The run used the pinned
`allenai/StdMoE_1b14b_1T_Preanneal` revision
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`.

## Frozen input and protocol

Only the first row of the calibration JSONL was used, in original order:

| Field | Observed value |
|---|---|
| Source document ID | `file:data/external/the_stack_python/cpython/Lib/test/test_sqlite3/test_userfunctions.py` |
| Category | `code` |
| UTF-8 bytes | `8,192` |
| SHA-256 of text | `dd1ca41399ccc61f42f1a8984c35c704d6c7449025a1f1e21337d70e42a63b4a` |
| Heldout opened for scoring | `false` |

No raw text, payload token IDs, or logit tensor was persisted.

## Repeatability observations

| Check | Repeat 1 | Repeat 2 | Result |
|---|---:|---:|---|
| Payload tokens | 1,788 | 1,788 | exact match |
| UTF-8 bytes | 8,192 | 8,192 | exact match |
| NLL bits | 2933.250114861314 | 2933.250114861314 | exact match |
| BPB | 0.3580627581617815 | 0.3580627581617815 | `abs(delta)=0.0 <= 1e-7` |

The two scores used `chunk_size=128`, EOS-prefix scoring, and no truncation.
For the separate first-64 payload check, the logits shape was
`1×64×100352`; all values were finite, `max_abs_error=0.0`, and
`elements_outside_tolerance=0` under `atol=1e-5, rtol=1e-6`.

## Resource and supervision record

The worker was supervised at the frozen five-second interval. The observed
maxima/minimum were:

| Metric | Observed |
|---|---:|
| Samples | 25 |
| Worker max private commit | 56,917,733,376 B |
| Worker max working set | 44,838,477,824 B |
| Minimum available physical RAM | 23,645,814,784 B |
| Worker exit code | 0 |

No resource cap or wall limit was crossed. The preserved worker stdout and
stderr logs are empty, and their write-once files are retained in the raw
directory.

## Artifact integrity

These are the SHA-256 values recomputed from the current raw files:

| Artifact | SHA-256 |
|---|---|
| `supervisor_result.json` | `89390013e926fddfad5c3bd651b29f1ae861de2bb98996d297300343cf65d41f` |
| `teacher_repeatability_scores.jsonl` | `00fe00ad6e015dd02467c6ac96b8c58ccaab2b8d944291e1129f9d23793d73e` |
| `supervisor_manifest.json` | `64cbbd5b00c96f77b432fdabeb5b2588d49759304b8c5f302ba088239e594d1f` |
| `supervisor_log.jsonl` | `f106e3ff89468fa624db8135448b63e1ebdb334eaa8bd6b71e84ad007c111ca6d` |

## Decision and limits

`PASS_REPEATABILITY` authorizes use of the pinned F32 teacher as a stable
oracle **for this repeatability control** in W4-v2. It does not add a new
heldout BPB measurement, establish W4 quality, attribute any W4 error, prove
compatibility with `benchmarks/phase60/engine.c`, or make a `tok/s`/50-tok/s
claim. No T4 was used.
