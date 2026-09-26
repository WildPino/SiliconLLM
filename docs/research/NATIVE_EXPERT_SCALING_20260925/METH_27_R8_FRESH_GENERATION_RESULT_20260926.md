# METH-27: stored R8 composition fails generation gates

Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth27_r8_fresh_generation.py --manifest docs/research/NATIVE_EXPERT_SCALING_20260925/meth27_r8_fresh_generation_manifest.json --manifest-sha256 94abd7544d4f887e476286774b2ce51bbc0c5c68f079907b88b3984589d62ec6 --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth27_r8_fresh_generation_result.json`.
The [protocol](METH_27_R8_FRESH_GENERATION_PROTOCOL_20260926.md)
fixed 24 prompts from the METH-25 selection, 256 input IDs and up to
128 greedy output IDs. The stored core/adapter hashes match METH-24/19;
all 169 matrices were rebuilt from stored int8 codes/scales. The
original BF16 donor, stored-R8 donor and stored-R8 adapter ran in one
RTX 3060 model instance. Full continuation IDs and decoded texts are
in the [machine result](meth27_r8_fresh_generation_result.json).

| Repeated 8-gram ≥3× | BF16 donor | R8 donor | R8 + E128 adapter |
|---|---:|---:|---:|
| Code (8 prompts) | 8 | 7 | 6 |
| Prose (8) | 8 | 7 | 6 |
| Technical (8) | 3 | 2 | **4** |
| Pooled (24) | 19 | 16 | **16** |

No arm emitted a premature non-EOS continuation under 16 tokens.
All three arms agreed on the first continuation token for all 24
prompts; later trajectories diverged. The R8+adapter arm generated
3,037 tokens and reached EOS on one prompt. Its technical increase
from 2 to 4 loops fails the frozen relative allowance of at most
one extra per category. Sixteen pooled and six code loops also fail
the separate ≤4 pooled/≤2 code absolute-usefulness screen. The
PyTorch BF16-reconstruction run took 417.9 s, peaked at 2.283 GB
allocated GPU memory and ended at 3.369 GB RSS.

Two technical failures added by the adapter illustrate the issue:
the Chinese SIG Node excerpt repeats a Pod/Host lifecycle sentence;
the MDN `transition-delay` example repeatedly extends the same CSS
block. The donor itself has severe repetition on these kinds of
continuations, so this test establishes an adverse *relative*
change on the fixed subset, not a broad claim about the adapter or
the dataset. The prompts had already been used for METH-25 top-1
comparison, but not for generation or factor/quantizer selection.

**Decision:** hold native promotion of this exact pair. METH-25's
document/ranking pass does not imply useful generation. PIQA on the
same stored bytes will distinguish task retention from this
generative weakness; any generation repair needs new training data
and new fixed prompts, with no retuning on these 24 outcomes.
