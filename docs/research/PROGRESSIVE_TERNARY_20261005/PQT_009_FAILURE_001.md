# PQT-009 first failure: insufficient eligible news evaluation windows

6 October 2026. **Invalid/incomplete experiment; no quality adjudication.**
Private acct3 kernel 137249305/version 1 terminated ERROR at
2026-10-06T01:11:52.719971Z. Exact privacy/image/T4 metadata remained qualified.
The first `failure.json` reports ValueError: `insufficient eligible independent
evaluation windows`, internal elapsed 2028.146326 s. Traceback locates the
failure in the news branch of `breadth_selection.evaluations`, after WikiText
selection and after every final representation froze. No evaluation model
outputs were calculated and no independent audit process ran.

The fixed news test pool did not provide sixteen eligible 129-token windows
after the cumulative consumed/prefix/calibration exclusion rules. The guard
stopped without shortening windows, changing seeds, relaxing exclusions or
silently accepting fewer contexts. No rejection-count ledger was saved when
selection failed; therefore the exact count and individual rejection causes
are not available and must not be invented. This is an input-selection failure,
not a measured failure of the breadth hypothesis.

Installation returncode 0, process elapsed 16.412078 s. Experiment returncode
1, elapsed 2038.600950 s, no bootstrap timeout error. The final phase event
reports five final representations frozen and narrow predecessor numerical
reproduction passed, with combined fitting/export 1809.718723 s. These are
unqualified execution records; without independent audit they do not prove
scientific validity or preservation. Do not interpret the retained online or
final fitted-set metrics as independently verified findings.

Terminal-only single retrieval and retention verified 75 files /
1,525,735,033 bytes, all sizes/hashes exact. All 22 actual embedded source
files match committed scientific source
`6630edb8abbf3711530247d6dbb6c1380102c5d1`, bundle `7c066ed`. Forty-eight small
raw files / 15,956,277 bytes are retained under `pqt_009_evidence/`; original
large artifacts at `results/progressive_ternary/PQT-009/remote_001/`.
See [retention](PQT_009_RETENTION.json), actual failure/traceback/bootstrap and
the immutable fetch manifest. Preserve the first failed namespace indefinitely;
never repush its reference or refetch into it.

A temporary delegated-monitor service interruption was resolved by checking
the same handle at 01:00:53.483697Z (RUNNING, exact metadata) and continuing it.
No remote job mutation/restart occurred. Monitor records 001..013 and parent
status_002 preserve the observations. This interruption did not cause the
selection failure.

## Numbered repair boundary

Prepare PQT-009-R1 as **evaluation/audit only**, reusing byte-exact frozen parent
payloads, calibration states, target ledger, counters and provenance. Do not
repeat optimization, change latent weights/scales, choose a different checkpoint,
add steps or use newly selected outputs for adaptation. The parent artifacts
must pass complete source/input/representation/state audits before any result
can be used. Count original fitting plus transfer/repair/evaluation/audit costs.

Prospectively admit the same immutable AG News repository/revision's train
partition as the new news **evaluation source**; its provider label does not
make it fitting data. It has not been used for this chain's calibration. Retain
and reconstruct old test-source rows separately and exclude their token/prefix
identities in the new partition; row IDs are partition-specific. Keep 129-token
prediction, 32-token prompts, sixteen eligible rows, all original strict gates
and separate WikiText/news decisions. The WikiText selection from the failed
attempt was token-only, not evaluated; explicitly record that history rather
than claiming it was untouched. New evaluation selection, code/input hashes,
resource bounds and independent audit require a frozen numbered protocol first.

Use the existing isolated private-dataset operation to transport only verified
frozen parent evidence when repair is ready. Do not assume errored notebook
outputs can be mounted directly, and never publish the artifacts. Current
PQT-009 has no audit or preservation decision. The full goal stays active;
local native timing still awaits the explicit owner CPU reservation.
