# Operational repair1: pre-worker overlap

FIRST launcher at Git7f74818511a15ff4b4275fabab2f883cec830508 stopped with
AssertionError overlap PID9276 python.exe, before a worker was created, after
.157s of launcher work. Original launcher_failure.json is retained unchanged.
No original result directory, log, worker instance or scientific observations
were produced. Inventory showed a foreign uv/maturin scopa_maestro editable
build (PIDs31596/9276); leave it untouched and wait for it to terminate.

Once no foreign compute overlap remains, ONE repair1 uses the SAME original
5048f5070460ab79a7284670497c0ac0ba93fe0bc2ab2dd8289a100cca8429e4 binding,
same collector bytes, inputs, formula, directions, h, tolerance and limits.
New directory/results namespace chatbot_null_curvature_repair1_20261008;
no numerical retry because the first attempt acquired no labels. Never
overwrite the original failed namespace or change the overlap guard.
All source/runtime/input hashes still verified by the actual launcher.

The audit binder now explicitly takes --curvature for the actual sealed raw
path instead of assuming the unobserved original result. This administrative
path change is frozen before FIRST observable; no audit formula/tolerance
change. FIRST audit results namespace chatbot_null_curvature_repair1_audit_20261008.
Terminal repair1 administrative checker covers the original launcher plus
both repair1 launcher/worker pairs (five known instances), typed UTC and OS
fault positive controls. Scientific decision/pipeline gates remain unchanged.
