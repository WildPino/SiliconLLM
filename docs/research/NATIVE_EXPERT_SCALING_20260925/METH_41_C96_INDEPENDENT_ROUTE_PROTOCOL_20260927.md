# METH-41: independent quality audit of the frozen C96 route

**Uncertainty.** METH-38's rank-64 int8 sketch with 96 exact
candidates passed next-token and route-recall diagnostics on the
prompts used to select that budget. METH-37's 64-candidate route
failed its top-1 quality gate. Test whether the frozen C96 rule
preserves document, ranking and relative generation quality on
source-disjoint inputs when it actually routes the saved R8+E128
model. This is a quality gate for the existing E128 route, not
a large-E or native-speed experiment. METH-40 independently showed
that a 273,547-expert exhaustive sketch scan has little CPU margin.

The candidate rule is fixed: the METH-36 stored fp32 basis and
row-scaled int8 rank-64 sketches retrieve the top **96** of 128
rows; candidate-only fp32 fine scores use the elementwise
multiply-and-sum path in `ShortlistExperts` (`rescore_mode=sum`),
then top-4/softmax gate the same E128 adapter at amplitude 0.50.
The exhaustive arm uses the unchanged fp32 router. Do not use
exhaustive scores to choose or gate C96, and run both arms on their
own model and generation trajectories. Source BF16 donor and
stored R8 core are the same artifacts as METH-37: source
SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
core `c307fa48015fd01ccc43ab6a90debe42c10245ac4e45b926fc3a6ee2572afa27`,
adapter `3147b2cf4fd3af671d5bf6c6451829ec32acfadaca2a0272be0b123e3873e1ca`,
sketch `133287eb87cf993d9498fd18c1efafbf703bf10a29cc757a5a5ec73cadbb674d`,
tokenizer fingerprint `4efeeb9382a77a06`.

The [frozen manifest](meth41_fresh_c96_manifest.json) has SHA-256
`8fff79f2502612d1f49083d1c48dd36d2718acec6335d676ccb5b8ded726365f`.
It selects eight documents each of code, prose and technical text
before model scoring. Code and technical spans come from Git
revision `9fb7791` (the existing `benchmarks/donor_adaptation/s1`
Python and top-level `docs` Markdown, respectively). Prose comes
from the two pinned external held-out JSONL files. The
[selector](../../../benchmarks/donor_adaptation/s1/meth41_fresh_c96_manifest.py)
excludes METH-17/19/25 source IDs and overlaps of the first,
middle and last 256 bytes with Qwen calibration/held-out text
and prior selected spans. It also screens overlap within this
set. The external code/technical rows failed that screen during
source preflight; they were not used. These inputs were not used
to choose rank64, C96 or the candidate fine-score rule, but the
older source pools and model development are not universally
independent.

Verify the bound artifact SHA hashes, metadata/shapes, tied head,
and 169 stored core matrices/121 control tensors. Reproduce the
METH-37 exact-route top-1 stream on its 24 old prompts as an
apparatus check before scoring these new inputs. Score the BF16
donor, exact-route stored R8+E128 and C96 route on full 24
documents with the METH-17 EOS-prefix/512-target/512-context
procedure. Compare 24×256 next-token top-1 IDs for exact versus
C96. Generate up to 128 greedy tokens from each arm on the
fixed 256-token prefixes; save all IDs and text and count
repeated-8-gram≥3×, early non-EOS under 16 and EOS.

**Prospective paired gate:** C96-minus-exact pooled BPB ≤+0.001;
each category ≤+0.002; category-stratified 20,000-draw
one-sided 95% bootstrap upper ≤+0.002 (seed 414141);
top-1 agreement ≥99%; and at most one extra repeated-8-gram
continuation or early non-EOS continuation pooled and in each
category versus exact. Separately, donor-relative C96 BPB must
be ≤+0.01 pooled/code and ≤+0.03 prose/technical. Passing
these gates would qualify only this E128 route for further
native testing. The parent R8+E128 candidate still fails
absolute generation usefulness (METH-27), and a route pass
does not repair it. Any failure keeps C96 unpromoted and
redirects work toward changed route/training or index design.

One local RTX 3060 run. The model scoring phase has a 20-minute
wall limit, 10.5 GiB peak allocated GPU memory and 20 GiB RSS;
selection/reconstruction stops on hash, overlap or apparatus
failure. No T4 job. Record runtime, raw paired results and
hashes, then update INDEX/METHOD with the actual decision.
