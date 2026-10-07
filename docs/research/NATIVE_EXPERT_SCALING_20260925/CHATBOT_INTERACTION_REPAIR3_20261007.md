# Canonical API container repair3, before FIRST valid ID-array comparison

Repair2 actual executora428a1/exit1 repeats only failed first fixture; worker2712/
create1791379011.834785, OS peak253370368B through exit. The first-case assertion
still fails; its newly added diagnostic then raises TypeError because canonical
output is a BatchEncoding, not a list. No raw worker failure JSON was written;
the actual worker log and launcher fault retain the full trace/actor/resources.
Other7 fixtures/stopping probes remain unentered. No completed fixture admission.

Bound local primary source `tokenization_utils_base.py` lines3015/3074 establishes
Transformers5.13.1 changed `apply_chat_template` default to return_dict=True.
Our comparison incorrectly tested BatchEncoding equality against three ID lists.
Those earlier "token-ID mismatch" assertions are INVALID as evidence of actual
ID differences; they show an apparatus container error. The lost arrays cannot
be retrospectively asserted equal or unequal. Source tokenizer incompatibility
is therefore not established, and no golden/criterion may be changed.

Repair ONLY API representation: explicitly request return_dict=False and require
list of integer IDs before unchanged ALL-case array equality. Preserve pending
vectors before assertion, which are now JSON-serializable. Source loader,
rendering, independent BPE mathematics, eight goldens, four stop criteria and
prospective120s/512MiB budget remain unchanged. Failed first fixture is computed
again; no completed fixture/model/native/source-response replay. Freeze before
this FIRST valid ID-list comparison. No lowering quality/threshold/expected IDs.

Reuse existing package/input trees and source semantics; new binding updates
only worker source SHA and appends this repair, prior binding and failure/log.
Then execute all originally frozen fixtures once, stopping any real mismatch.
Continue toward complete compact Qwen converter after genuine contract closure.
