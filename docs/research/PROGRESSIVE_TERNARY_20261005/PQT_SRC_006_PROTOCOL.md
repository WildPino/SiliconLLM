# PQT-SRC-006: bounded ZIP-to-tensor source qualification

Preregistered6 October 2026, before controls or new original payload reads.
Question: can the original-checkpoint ZIP reader supply checked F32 tensors
directly to the unchanged ephemeral-expert load(name, device) contract, with
bounded temporary buffers and no NPY checkpoint duplication? This is source
qualification only; no natural corpus, pretrained model forward, fitting,
quality measurement, local GPU job or native timing is admitted.

Keep switch_zip_source.py and switch_lazy_float.py unchanged. Bind their exact
historical qualified Git bytes, the new adapter/worker source commit and this
protocol before each numbered operation. Own namespaces only; main originals
read-only. Use the existing isolated scientific runtime Python3.12.10,
Torch2.6.0+cpu and NumPy2.1.3, one thread, no gradients. Live process admission
must pass before scientific CPU controls or original reads. Preserve first
failures and numbered repairs; no overwrites or silent retries.

The adapter must validate full pinned pickle/header/storage namespace using the
qualified restricted parser before any tensor reads. Each requested tensor must
be full contiguous little-endian F32, offset0, exact shape/byte count and SHA256,
finite before exposing any tensor. Return a fresh owned byte buffer; count each
actual read, bytes and metadata/payload budget, including repeats. No full-shard
Torch load, mmap, persistent tensor cache or NPY output. Compare alias-group
shape/identity at admission and recheck archive stat identity before and after
each request and closure. Archive whole hashes remain historical; qualify
actual selected payloads, not a new full-checkpoint scan.

Bound source tensors<=128 MiB, payload chunks<=1 MiB, metadata reads<=2 MiB;
only one temporary source buffer at a time. Instrument maximum temporary source
buffer, chunk and finite-check mask sizes, successful tensor reads and failed
attempts. These are adapter-owned temporary statistics, not caller-held tensor
or physical device memory measurements. CPU returned tensor ownership survives
source closure; GPU transfer is not qualified by these CPU controls.

Synthetic controls use exact actual torch.save fixtures and independent
weights_only Torch loads/NumPy bytes. Require positive weight/alias/signed-zero
and repeated reads exact; actual expert-pair handoff and partial/FFN failure
cleanup through the unchanged manager must pass. Reject wrong hash/shape/dtype/
namespace/alias, nonfinite, offset/noncontiguous view, corrupted ZIP/CRC,
unexpected pickle global, modified archive, byte/time/buffer caps and reentrant
read. Save complete fixtures, source bindings, actual returned positive arrays,
raw input/output/control logs and resource scopes. No original payload read
until committed controls and complete retained bytes are independently verified.

Controls each<=180s/1 GiB process RSS/64 MiB outputs; source metadata<=8 MiB,
payload<=128 MiB including repeated small tensors. Count actual synthetic FFN
calls; original pretrained functions remain zero. Controller records complete
child elapsed/exit/timeout, separate from nested before-manifest/report timers.

After controls pass, one original qualification may read exactly SRC003's15
consumed tensors totaling118,358,016 raw bytes, no others. Bind full
PQT_SRC_003_BINDING.json and its three pinned archive/header identities. Require
all3,320 headers exact, each returned payload exact to the prior admitted NPY
and tensor SHA, finite/shape/F32 match and unchanged archive stats. Original
operation<=180s/1 GiB RSS; metadata<=8 MiB/payload<=128 MiB, and per tensor128
MiB. Save complete small original result/source/cost evidence and per-tensor
identities, referencing retained SRC003 NPYs rather than duplicating them.
Do not interpret this selective budget as a bound on complete float capture.

Retain every synthetic raw byte losslessly with exact manifest and Git proof;
verify original reports and source closures too. Qualification requires complete
controller, worker, independent raw/source/retention checks and Git verification.
A successful result admits only the checked CPU source contract and selected
original values. T4 provider deployment and whole-model original float capture
still require prospective complete I/O/scratch/VRAM/time and data/reference
admission. Native timing awaits the separately pending owner window.
