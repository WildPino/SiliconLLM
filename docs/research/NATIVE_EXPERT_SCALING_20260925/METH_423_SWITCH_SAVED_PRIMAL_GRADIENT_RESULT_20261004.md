# M423 result: local backward conforms; differing loss primals explain mismatch

Frozen3b74720 then0618f56 BEFORE first numeric outcome (explicit F32 cast at
native logit gradient interface). First/only diagnostic exited0, ALL5 gatesPASS.
Raw SHAa8ef145bd2c0fa496d5bf7dbfbdfb3b349b5061502ba3546efa4c6c56c5f2424.
Fresh whole22.36GB source payloads/manifests,422 failure/256 gradient archive,
418 raw and first same-context native captures, committed helpers. Source128
first teacher.book0.case0/expert25/rank8/seed547, unchanged detached source256
target. No new point, data, rank, seed, threshold or optimizer update.

Native zero-correction forward every endpoint/full32128 head remains byte-exact.
Independent literal NumPy LOCAL A/C backward on captured native primals has
relativeL2 **0.0 /0.0** against actual autograd. This verifies those declared
approximate local rules on this point, not true derivatives through native
rounding/top1 or whole-gradient conformance on every case. No signed-zero byte
claim for gradient arrays from a norm-zero comparison.

| Fixed counterfactual | A vs global smooth | C vs global smooth |
| --- | ---: | ---: |
| Native ReLU mask + native loss primals |0.004939194124061269 |0.0048021180248012905 |
| Native ReLU mask + smooth loss primals |1.0095095822404155e-6 |2.9339000173929254e-6 |
| Smooth ReLU mask + native loss primals |0.004939194124061269 |0.0048021180248012905 |
| Smooth ReLU mask + smooth loss primals |1.0095095822404155e-6 |2.9339000173929254e-6 |

NO changed ReLU rows. Full native/smooth logit relativeL2 only
5.61790814698254e-5. Changing only the head logits used for the loss derivative
removes nearly all global-gradient discrepancy; remaining captured native
primals held fixed. Actual declared local implementation therefore conforms,
while comparing those gradients against a global dequantized path evaluates
loss at a different point. Counterfactuals are diagnostics, not repaired model
or fitting licenses. Original422 mixed/global1e-3 criterion remains FAIL.

20.485s/max1106522112B RSS/peak/22382756174B hashes/770742B output, within2min/
2GiB/8MiB. Complete native/smooth logits/up values/A/C gradients and all four
counterfactual arrays retained with full archive hash in raw. CPU only; no
GPU/T4/network/new corpus/native artifact/quality/rate/useful capacity claim.

Next NEW424 protocol: evaluate a matched-primal local smooth continuation with
detached node offsets, so independent finite differences test the declared STE
at the SAME native values. It must separately establish whole coupled gradients,
offset/forward fidelity, nonzero initialization and independent negative controls.
Native rounding/top1 remain nondifferentiable; any anchored local continuation
is explicitly an approximation. Alternatively a fully coherent dequantized
training surrogate needs separate whole native-fidelity/gradient qualification.
Do not loosen422's bounds or infer globally smooth equivalence from local rules.
No training/selector/capacity artifact before new numerical and bounded-fit
protocol; original donor-relative quality/SAMEartifact accepted50 still required.
Goal active/incomplete; useful n/RAM/LUT/realDRAM/another-family/~100B remain open.

Final retention check:411 unique files/833790106B freshly hashed, ALL384420
complete baseline archives,419-423 raw/helpers/protocols and retained gradient
archives exact. Original374/389 binary hashes unchanged. No model job; two
exact previously authorized publisher daemons preserved.
