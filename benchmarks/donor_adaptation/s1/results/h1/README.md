# H1 session artefacts

The two T4 bundles are NOT here and are not in git: 1.33 GB each, and the standing rule
is that weights live in exactly ONE place on disk.

    D:\_ktmp\h1_s1_final\h1_trained_s1.npz   seed 1717, fresh carve,  ~195 steps (inferred, see addendum J.4)
    D:\_ktmp\h1_s2_final\h1_trained_s2.npz   seed 2718, resumed from s1, 195 steps -> CUMULATIVE ~390

`h1_trained_s2.npz` is the artefact the gate is read from.

`h1-qat-run-session-1.log` is absent because it is 0 bytes on disk: the kaggle library writes
logs with open(path,"w") and no explicit encoding, which crashes on Windows/cp1252 when the
log carries a tqdm glyph.  Session 1's completed step count is therefore unrecoverable from
the artefacts.  Fixed for later sessions with PYTHONUTF8=1 alongside PYTHONIOENCODING=utf-8.
