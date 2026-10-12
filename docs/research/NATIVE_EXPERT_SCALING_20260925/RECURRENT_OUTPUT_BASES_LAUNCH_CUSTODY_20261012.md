# Launch attachment bookkeeping fault retained

Source/protocol were committed in e409079 before binding. Binding and launch
JSON were completely written before the run; their command/freeze/limits were
consumed unchanged. Initial `git update-index` failed because PowerShell grouped
two path concatenations into one argument. The subsequent run started with tool
session30075 before the launch JSON Git attachment was corrected.

Corrected only the explicit path array in the Git command; no source, input,
binding, command, output namespace or gate changed. Launch/binding were attached
in the immediate follow-up commit. This records the sequencing fault without
claiming that the Git attachment preceded process creation. Numerical custody
still depends on the immutable binding, pinned source freeze and full input
hashes before/after, which the run and independent audit verify.
