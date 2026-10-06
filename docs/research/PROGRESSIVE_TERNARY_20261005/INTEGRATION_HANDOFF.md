# Documentation integration handoff

6 October 2026. **Ready for review; not applied to the active checkout.**

| Identity | Value |
| --- | --- |
| Target branch | `research/native-expert-scaling` |
| Target commit used as base | `d4a602fcbbd4e2507a40d24690d0902d39d79aa7` |
| Prepared branch | `research/progressive-ternary-integration` |
| Prepared worktree | `<integration-worktree>/SiliconLLM` |
| Preserved full source/raw branch | `research/progressive-ternary` |
| Completed source/raw commit | `0f906f495551c24829a130bf772efb713d3ba533` |
| Latest scientific source | `e218e3e36d44ab4a34de091f5ccad05c0a9267db` |
| Latest exact raw evidence commit | `3bbc58fa7f36a5d4174718a69609f3036ec1160d` |

## Scope to review

Import the documentation snapshot from the prepared branch, not a full merge
of `research/progressive-ternary`. All new files are inside
`docs/research/PROGRESSIVE_TERNARY_20261005/`; the only existing-file edit is an
additive completed-research paragraph in `docs/research/RESEARCH_INDEX.md`.
No runtime, benchmark, experimental helper, model, root `AGENTS.md`, root Git
attribute or dependency changes are part of this integration.

The [manifest](INTEGRATION_MANIFEST.json) binds every selected document to its
original Git blob/SHA256 and its separate snapshot identity. In39 historical
records only machine-specific worktree-root labels are aliased; source evidence
and measurements remain unchanged. The [README](INTEGRATION_README.md) explains the
scientific conclusion and documentation/archive boundary. The
[omitted inventory](INTEGRATION_ARCHIVE_INVENTORY.json) locates retained raw
files on the original branch. Publication of the full evidence is a separate
decision; this preparation makes no remote push and no private service write.

The active checkout was observed with uncommitted changes in the research
index and two donor-adaptation files. The prepared index is based only on the
committed target version; it does not copy or overwrite those in-progress
changes. Keep the current owner's work authoritative when reviewing a future
merge. The target commit may advance after preparation, so use fresh status
and compare against the live target before applying anything.

## Coordination and application

Preparation can coexist with the active work. For actual application, the
current checkout owner should reach a stable step, commit intended tracked
changes and pause file edits during review/merge. Do not automatically stash,
reset or commit unrelated generated files. Long-running computation need not
be terminated merely to prepare or read this documentation; the owner decides
when its file-writing state is safe for application.

The following commands are for the coordinated application stage, **not
operations already performed**. Run them in the target checkout after its
owner has prepared that state:

```powershell
git branch --show-current
git status --short --untracked-files=no
git status --short -- docs/research/PROGRESSIVE_TERNARY_20261005
git diff --name-status research/native-expert-scaling...research/progressive-ternary-integration
git diff research/native-expert-scaling...research/progressive-ternary-integration -- docs/research/RESEARCH_INDEX.md
git -c commit.gpgsign=false merge --no-ff --no-commit research/progressive-ternary-integration
```

Expected target branch: `research/native-expert-scaling`. Tracked edits should
be committed before the merge, and no untracked destination file may collide
with the imported namespace. The staged merge must change only the two paths
listed above. If the index conflicts, preserve the latest native-scaling
content and insert the completed ternary-research paragraph once. Do not
replace the index wholesale with the old prepared version.

Review the staged paths/content and whitespace, then commit the reviewed merge:

```powershell
git diff --cached --name-status
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
git -c commit.gpgsign=false commit -m 'Integrate progressive ternary research documentation'
git status --short --untracked-files=no
```

Review the linked conclusion/literature/result entry points after application.
No scientific experiment, CPU/GPU benchmark, dependency installation or native
model test is required for this documentation-only change. Do not rerun
completed fits/fetches or promote an artifact on the strength of the merge.
The active owner can then resume its original work. Preserve the full research
branch and this prepared branch until the merge/provenance has been verified.

## Completion of preparation

The [preparation verification record](INTEGRATION_VERIFICATION.json) proves original/snapshot identities and
the explicitly bounded root-label transformation, relative
links, original-index preservation and the exact changed-path whitelist.
Its measured checks are metadata/document operations, with no numerical
model execution. The branch can be reviewed now; actual application remains
outside this preparation task.
