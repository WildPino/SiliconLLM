# Selected509: ONE top8 I8-to-source-F32 head refinement

PROPOSED only. Goal ACTIVE/INCOMPLETE.508 source-head/state intervention complete
and independently verified:20 original differences recovered,3 introduced,
31->14 disagreements on996 common-history positions. No new whole quality/rate.

## Why

Full F32 source head reads98.7MB logical coefficients versus24.7MB old I8 head;
current SAMEartifact speed margin is small and unqualified. Conditional F32
precision on a small vocabulary candidate set might retain508's readout benefit
at low extra cost. Old394/395 address expert ROUTER winner AND mass, not this
vocabulary head. New source/precision/candidate state from506/508 is explicit.

## Single prospective local recipe

Use ALL96 native generation wires and ALL996 common positions from qualified508;
reuse its full-vocabulary source-head winners as consumed oracle, no repeated
full head evaluation. Freeze main/independent auditor/actual dependency/input/
source extent closure before any new selected-row source-function query.
No Torch/PyArrow, whole checkpoint, donor NPZ/corpus/model/C/compiler calls.

For each position sort old I8-head logits descending, then ascending original
vocabulary ID for ties. Choose EXACTLY8. F32 tied scale on captured native final
state; selected original F32 rows dot in F64, cast result to F32, overwrite only
these8 logits. Unselected old I8 scores remain. Argmax whole hybrid array with
lowest-ID ties; also retain selected-only rerank winner and any unrefined tail
overtake. Retain all8 IDs/source scores/native scores, full source oracle rank,
hybrid/refined winner/ties, recovery/new-change flags, each case/book/outlier.

Frozen controls: ALL996 full source-head oracle IDs in top8 AND ALL996 hybrid
full-vocabulary winners equal qualified508 source-head winners. These are
finite local criteria, not a general guarantee. Reproduce31 old disagreements,
20/3 full-source counterfactual outcomes from retained508. Independent rank
count and lexsort/partition implementation, source extent/rows, state/ties,
hybrid winner/metrics reconstruction. If a criterion fails, close THIS top8
recipe before native promotion; no k grid/tuning or threshold edits.

Proposed main/audit each<=180s/2GiB/32MiB (small binding<=180s/512MiB).
Logical selected source work996*8*768=6,119,424 MACs,24,576B extra F32 coefficients
per position, plus old I8 head cost. Selection scan/reread/coefficient gathers
have real cost still to measure. Actual input/catalog/source maxima and CPU
thread/affinity/process/deadline/output receipts precede observation.

## Next decision and limits

If all local criteria pass, implement ONE engine head branch/control and measure
actual cost, then new own-state generation/task quality with original donor and
same-artifact >=50 accepted IDs/s. Full correctness on consumed prefixes does
not ensure downstream trajectory/rare-input/held-out coverage. Head NLL uses
the hybrid approximate logits, not source full-head logits. Other readout/body
errors and508's three introduced choices remain explicit.

If top8 fails, preserve all misses/tail overtake and consider a source-derived
adaptive error bound/fallback as a DIFFERENT future hypothesis. No numerical
certificate implemented or claimed yet. Exact WI acceleration/256 expansion
remain deferred pending whole-quality path. No compact-core/useful-n/DRAM or
cross-family promotion from this head experiment alone.
