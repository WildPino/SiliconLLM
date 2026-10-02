# METH-265: same-artifact full-prefix generation reference

## Decision and prior evidence

METH-263 passes independent full-model prediction on the saved259 archive.
METH-264 then stops on 2/48 cached/full choices in the original BF16 donor,
before generating any complete continuation or candidate output. Preserve
that stop and its zero-mismatch threshold. This protocol is frozen after
those source-only partial choices were observed, before any265 continuation.

Change only reference decoding arithmetic: at every step recompute the
whole prefix with `use_cache=False`. The purpose is to establish finite
generation health and K64 behavior for the fixed complete candidate.
This reference path is not an efficient production decoder. A pass only
licenses separately frozen tasks/blind semantic assessment; actual native
cached quality and >=50 accepted batch1tokens/s remain mandatory.

## Fixed inputs and apparatus

Bind263 raw SHA `7bcb8774bf335762e5e9b89dee29223bf1fa2bfea8d59480ed0621fa1c4d7821`,
264 stop SHA `c016596e419476e7033fc95e774b06bc184e54f8f6ceaf08989dcc54e3865029`,
and the same26124 prompts/source IDs/tokenizer,259 complete archive and
source/parent/child/helper hashes. No weight fitting, source filtering or
prompt replacement. Three arms: original BF16 donor, original BF16
centered E1280, and actual saved unique-bank E1280 candidate. Source arms
use pinned original weights; candidate loads every weight from259 archive.

Use unpenalized greedy BF16 full tied head, lowest-ID ties, config EOS,
128-new-token cap. No temperature, repetition penalty or new wrapper.
Retain all raw IDs/text and per-document progress. For every candidate
generated state require fixedK64 proposal inclusion and exact-row rerank
to agree with the full head. Full head chooses the actual token. Any
miss/mismatch stops before tasks/native promotion. This finite check
cannot certify universal proposal coverage.

After all72 continuations, compare candidate against both controls:
pooled EOS count >=control-2; early non-EOS termination<16 and repeated
8gram3x counts <=control+1, pooled and every category. These are unchanged
215/264 health criteria. Failure closes this fixed full-prefix candidate
path; pass alone does not establish factual or semantic preservation.

## Cost and command

Local RTX3060, six hostthreads, deterministic/highest/TF32off; 70min
after imports,20GiB RSS/10.5GiB GPU; no T4 or downloads. Failure and
partial outputs are preserved. No CPU speed measurement while running.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth265_full_prefix_generation.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth265_full_prefix_generation_result.json
```

Large-RAM useful n, CPU routing/LUT/alias lookup/real DRAM cost, native
whole-model quality/rate and other-family/10B/100B transfer remain open.
