# PQT-SRC-004: synthetic full Switch execution qualified on CPU

6 October 2026. **SYNTHETIC CPU APPARATUS PASS; RESEARCH INCOMPLETE.**
[Protocol](PQT_SRC_004_PROTOCOL.md), [independent audit/retention](PQT_SRC_004_AUDIT_RETENTION_001.json),
[exact Git proof](PQT_SRC_004_GIT_VERIFICATION.json).

## Qualified behavior

The complete tiny encoder-decoder, including all dense/sparse blocks and tied
embedding/readout, produces identical retained traces through the unmodified
official eager path, resident SDPA MATH and SDPA with ephemeral experts. This
holds for all32 prospectively defined CPU F32 cases: capacities1/64, batch1/3,
encoder length7/13, decoder length5/9, full/right-padded masks. The independent
standard-library audit checks every retained logit, encoder/decoder hidden
state, sparse router mask/probability/logit and dispatched expert input/output.
Maximum independently recomputed relative RMS0; all argmax and route masks exact.
No original pretrained weights, corpus, GPU or fitting enters these tests.

The SDPA path uses scale1 and the original relative/additive bias contract.
It explicitly rejects training, non-F32 execution, caches, offset positions,
head masks/pruning and requested attention weights. It is qualified for full
teacher-forced encoder-decoder calls without cache; generation/cache behavior
has not been admitted. CPU equivalence does not admit the T4 backend/runtime.

Checked loading validates every weight's NPY shape/dtype, full file hash,
original raw hash and finite values. Core parameters remain resident/unchanged;
four embedding/head aliases share one Parameter. All experts start and finish
on meta, with at most one pair resident (1,024 synthetic coefficients), loaded
only on actual nonempty official dispatches and immediately released. Independent
retention matches every logged expert read to the frozen checkpoint and actual
load/release/dispatch sequence. Each case makes four full-model calls: eager,
resident, streamed, repeated streamed;128 calls per successful round.

An independent router/FFN equation verifies capacity per sequence and zero
outputs for eight dropped tokens, relative RMS6.049753566879122e-8.
Round003 passes23 malformed controls, including changed bytes/layout/dtype,
nonfinite weights, missing namespaces, alias errors and unsupported calls.
Partial-pair loading failure and an exception during the original FFN both
release weights and leave every expert on meta. Numerical repeat identity is
checked by the executable worker; the separate audit checks repeated read/
dispatch logs. Repeated numerical tensors are not separately retained.

## Runtime, attempts and measured costs

Owned scientific environment: Python3.12.10, Torch2.6.0+cpu, NumPy2.1.3,
Transformers4.57.6, one thread. Fifteen added dependencies have exact PyPI wheel
size/hashes; all preexisting versions remain unchanged. Installed Switch model/
config source hashes match the consumed official release binding. Complete
active dependency requirements, distribution METADATA/RECORD identities and
locations are recorded in the qualification report. No main environment changes.

| Attempt | Outcome | Measured seconds | Peak process RSS bytes |
| --- | --- | ---: | ---: |
| Runtime001:14 pinned wheels/offline install | Installed identities pass; later dependency check finds missing Windows colorama | 73.875 setup /69.062 install | Not instrumented |
| Premature import probe during live install | ModuleNotFoundError; no model construction/function, same setup observed to terminal | 0.3146903 command duration | Not instrumented |
| Dependency completeness check001 | tqdm requires colorama on Windows; no model function | 1.0393588 command duration | Not instrumented |
| Runtime002:colorama0.4.6 only | Exact pinned dependency completion;14 packages preserved | 3.984 setup /2.875 install | Not instrumented |
| Synthetic001:32 cases/19 negatives | Numerical gates pass; elapsed timer ends before manifest hashing | 47.110 before manifest | 290,541,568 |
| Synthetic002:extended controls | Live guard rejects other checkout's Python; zero model functions | 0.8537079 complete worker /0.484 nested | 28,037,120 |
| Synthetic003:unchanged extended source | Fresh guard passes;32 cases/23 negatives, failure cleanup pass | 30.3765057 complete worker /29.500 nested incl manifest | 301,842,432 |
| Independent stdlib audit/retention001 | Every trace equation, load/dispatch and raw identity pass; no forward replay | 8.578 | 45,277,184 |

The initial import probe was issued before confirming the install terminal;
[observation retained](PQT_SRC_004_RUNTIME_PROBE_FAILURE_001.json). It did not
restart installation. The [dependency failure](PQT_SRC_004_DEPENDENCY_FAILURE_001.json)
was corrected prospectively by adding only colorama. Extended round002 stopped
at resource admission while another checkout's Python was live. After its
authoritative process disappearance, round003 used unchanged executable source
and a new exclusive directory. Keep all these attempts and costs; do not merge
them into one successful-duration figure. Complete synthetic001 process duration,
runtime setup RSS peaks and authoring/Git-command costs are unmeasured; no inferred
native speed or scientific quality benefit. The timing controller now enforces
the180-second complete-worker timeout and retains stdout/stderr/report identity.

Runtime acquisition20,297,943+45,094 bytes, including metadata. Qualification003
outputs4,237,450 bytes before its report. Its lazy source performs917 checked
reads/1,777,024 raw bytes at capacity1 and925/1,793,408 at capacity64, including
core loads and both streamed passes; four additional expert reads/8,192 raw bytes
exercise failure cleanup. NPY framing and full-file hash checks add I/O beyond
these raw coefficient counts. This is synthetic source cost, not pretrained
Switch throughput or physical VRAM measurement.

## Exact retention and resumption

Seven deterministic ZIP_STORED archives retain every original synthetic tensor,
index, input, report, controller log and historical executable source, plus raw
runtime metadata/install logs. Runtime wheels remain outside Git with pinned
download URLs/full identities. No synthetic numerical artifact is discarded.
Archives13,784,826 bytes;10,062 raw entries/12,380,660 raw bytes. Git proof at
0b42fa411784d690816d423054e43c5542dd7694 checks20 actual committed files/
15,363,027 bytes and every ZIP entry against its original byte identity. The
manifest and audit bind each historical source revision, including earlier
control and runtime versions; later edits do not rewrite the earlier evidence.

Qualified executable source3e019e37aa0c3e6be2c524f8615e5aee22faa9c0;
stdlib auditor ab131b93d976c7a455afdc5bcafcdc7fa3c6b28e.
Owned source: benchmarks/progressive_ternary/switch_lazy_float.py and
scripts/progressive_ternary/setup_src004_runtime.py, qualify_src004_execution.py,
run_src004_qualification.ps1, audit_retain_src004.py, verify_src004_git.py.
Completed result namespaces underresults/progressive_ternary/PQT-SRC-004:
runtime001/002, qualification001/002/003, controller002/003. Preserve them;
never overwrite/restart a completed run. Lossless raw evidence is stored under
pqt_src004_evidence/audit_001.

Next qualify this exact operator/loading hypothesis on the pinned T4 runtime
with separate synthetic inputs and numerical gates. Then freeze a separate
original float masked-span context acquisition protocol: source/tokenizer/corpus
identities, prospective optimization/verification splits, routing/coverage and
discard rules, full encoder-decoder conditioning, source reference equivalence,
and realistic complete-source/scratch/I/O/VRAM/time stops. No original natural
context has yet been acquired. Current synthetic tests do not prove original
expert function fidelity, useful ternary capacity, preservation or native speed.
Private dispatch still needs fresh account-owner/quota/all-kernel admission;
long T4 monitoring is delegated while the coordinator waits dormant. Native CPU
timing remains pending explicit owner availability. Keep the full goal active.
