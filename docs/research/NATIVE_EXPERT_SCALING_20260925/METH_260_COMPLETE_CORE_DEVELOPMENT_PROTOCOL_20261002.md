# METH-260: frozen full-model consumed-development screen

Bind259 export result SHA
`39f796651522a2af907f4123756b9c7ab1e838057debbb142db5c1e02765b663`,
require all gates and same actual artifact SHA
`3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71`.
Require archived export script and all used helpers unchanged. Bind original
METH-121/122 manifest, parent/child training/checkpoints, pinned local BF16
donor file and METH-194 original control result SHA
`9063af7040e063070a3f590e5be19001ffb48f8b007a2375db8dd17aa656ec1b`.
All24 document text/token and prompt-token hashes exact. This cohort has
already been consumed many times: this is development, never fresh quality.

## Three complete model arms, no fitting

1. Original BF16 donor, conditional bank disabled.
2. Original BF16 donor plus original centered E1280 bank, enabled.
3. Complete saved259 unique-bank/source-core candidate with bank enabled,
   loaded from config/weights/maps in its actual archive, no source fallback.

Same24 documents/prompts,8 per code/prose/technical category. Full exact
BF16 tied head for all probability/ranking scores; no shortlisting in BPB
or donor-top1 reference. Replay all original194 donor/E1280 document NLL
and per-prompt matching counts exactly before candidate observations.
One current candidate, no precision/bounds/rank/router/training adjustment.
Score complete model hidden propagation, not isolated layer functions.

After candidate scoring, fixedK64 head proposal on each candidate prompt
hidden state using archived int8 codes/FP16 scales decoded BF16. Require
no omitted full-head top1 and no exact-row reranking mismatch on these
finite prompts. This is not a universal shortlist/generation certificate.

## Frozen decisions

Candidate pooled BPB minus both original BF16 controls<=.01; each category
minus both<=.02. Candidate donor-top1 agreement relative to BF16 E1280
must lose<=1 percentage point pooled and<=2 points in each category.
Also exact old controls/complete archive loader/finiteK64 inclusion and
exact-rerank gates. These are the original214 thresholds, not chosen from
new candidate scores. Diagnostic paired source bootstrap10,000 draws,
seed214214 (reused fixed summary helper), P05/P95 versus both controls;
bootstrap is descriptive, not an additional decision threshold.

All gates pass only licenses a separately frozen genuinely new independent
source/fragment-disjoint manifest, then its full prediction/generation/
task/blind gates before native whole-model/alias lookup/LUT/DRAM and
same-artifact>=50 accepted batch1tok/s. Failure closes this fixed candidate;
no new-source screen or native quality/rate promotion based on a passing
subset. Original214 cohort is also consumed and cannot become fresh again.
No new n-semantic count gain, arbitrary RAM scaling or10B/100B/second family
claim from this small dense source plus existing conditional bank.

Local RTX3060/six threads, deterministic/highest/TF32 off,20min after
imports,20GiB RSS/10.5GiB GPU. Original source locally cached; no acquisition,
T4 or new data. Preserve per-arm partial and failure stages. Freeze before
running once:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth260_complete_core_development.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth260_complete_core_development_result.json
```
