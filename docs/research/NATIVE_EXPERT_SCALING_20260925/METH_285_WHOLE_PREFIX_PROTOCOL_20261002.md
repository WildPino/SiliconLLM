# METH-285: whole complete-archive prefix/cache numerical smoke

## Purpose and prior binding

284 qualifies actual phase60 binding/all6144 source/conditional operators,
but its complete attention/norm/RoPE/residual/head forward was never invoked.
Exercise that SAME implemented equation next,without changing weights,
arithmetic or route policy.285 model header is verbatim284 source before
its main. The new phase60 `SILICON_COMPLETE_I16_FULL` personality supplies
only the batch-prefix/cache assay entry;all forward operators stay exact.
Original284 source/results remain preserved.

Same original276 archive SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`;
284 pass SHA `615e4854b8e9300f660c66f513152e4ed63b78e41f6d521400f784bede0ca18a`;
278 source manifest SHA `7fc74c2ff172bcca5d71de685f1e230240d05789e11b0bb3b9290b64a4c8e053`.
Require all284 gates,same archive and all76 project helper hashes. No new
dataset,new weights/fitting,output selection or source fallback. Full-prefix
GPU reference uses276's own loader/BF16 head/no cache/TF32 off/deterministic
Torch/six threads/CuBLAS4096:8.264's failed cached GPU recipe is not reused.

## Fixed inputs and prospective gates

Use entire278 prompt-ID sequences at fixed indices0,1,8,9,16,17,two each
code/prose/technical_general. These sources are consumed diagnostic data.
No source selection based on native outputs. For each prompt compare its
LAST eight real positions,not generic chat-header positions.48 positions
per arm. Two actual stored-archive arms:bank disabled (compact-core-only
ablation,NOT original donor) and bank enabled (complete E1280 candidate).
GPU computes each full prefix and then its last8 hidden/full-head logits.
CPU sequentially prefills the entire prefix and retains those same states.
All execution weights and config remain the original artifact.

Initial numerical smoke gates BEFORE any native outputs:

- Same archive SHA/all725 typed fields/no source or checkpoint-weight fallback.
- Finite native hidden/full-head logits everywhere.
- Hidden-vector relativeL2 maximum<=.05 and full151936-dimensional logit
  relativeL2 maximum<=.05,each arm over48 positions.
- Top1 agreement>=.95,each arm over48 positions (at least46 matches).
  Lowest token-ID ties on both sides. This coarse smoke licenses further
  quality checks;it is NOT the final donor-relative BPB/top1 quality gate.
- For every six-source/two-arm case,independent CPU cache allocation rebuilt
  from the complete prefix gives BYTE-exact final hidden and full-head logits
  versus the original sequential run. This guards CPU cache/data flow,not
  CUDA cached/full-prefix equivalence or universal contexts.
- After rebuilding,erase all K/V history and rerun only the final real token.
  This deliberate stale-history fault must change final hidden OR logits
  for every case,so the exact checker detects it. Retain resulting numeric
  error as descriptive. No weights or IDs change in this negative control.

Retain all96 positive native/reference hidden/logit rows,12 negative rows,
IDs/cache flags/top1/error arrays and provenance. Do not enlarge thresholds,
regrade or tune after observations. Positive failure stops before full
quality/rate;localization or a corrected execution recipe requires a new
prospective record. Pass licenses complete native prediction/generation/
task/semantic and K64/rate qualification under their own fixed apparatus.
No pass bypasses same-artifact donor-relative gates or the>=50 requirement.

## Cost and reproduction

Expected GPU<2minutes,CPU1-5minutes. Hard GPU12minutes after imports,
20GiB RSS/10.5GiB CUDA;CPU compilation/forward/cache/negative<=600seconds
AFTER GPU synchronization/reference completion,no overlapping model job
or performance benchmark. Native plus reference<160MiB,no download/T4.
No accepted timing/prefill/rate interpretation of this qualification run.
Source-only compile succeeds before model/native numerical observations.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth285_whole_prefix_qualification.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth285_whole_prefix_qualification_result.json
```

Save terminal result/failure/handle;never duplicate a live launch after an
observation timeout. Code/protocol freeze commit must precede execution.
Native quality/K64/accepted>=50,useful large-RAM n,real DRAM/LUT costs and
cross-family10B/100B all remain open;old259/264/274-275 stops unchanged.
