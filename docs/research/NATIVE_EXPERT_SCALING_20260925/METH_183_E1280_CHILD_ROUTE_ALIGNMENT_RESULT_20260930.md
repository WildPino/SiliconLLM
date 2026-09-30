# METH-183: learned E1280 child selection has functional alignment

**Decision: pass the frozen mechanism diagnostic.** For the already
quality-gated, mean-preserving Qwen2.5-0.5B-Instruct E1280 bank, the exact
learned child route gives lower pooled document BPB than every nonzero
cyclic permutation of the ten children. A paired source bootstrap also
places the fifth percentile of mean-rotation disadvantage above zero.
This supports retaining a content-coupled learned route in the next
expert-count method. It does not show that a third tier can be balanced,
trained, or transferred to GigaChat 10B.

The [protocol](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_PROTOCOL_20260930.md)
was committed at `a349ead` before implementation and execution; the
[runner](../../../benchmarks/native_expert_scaling/meth183_e1280_child_route_alignment.py)
was first committed at `d321e05` and corrected at `67e77d8` before the
valid run. It binds the METH-121 manifest SHA-256
`7f35f2253850f3a19e0bf517d2e088d2d08c88c0bc68cc09413781ddac6f9366`,
METH-107 checkpoint SHA-256
`15a14b8476936e83cf91a479b05f8d8e83f4ffd094186f3d4dfededdb138d520`,
METH-56 E128 parent, pinned BF16 donor and tokenizer, and the exact
METH-122 mean-centered B transform. It retains each parent choice and
top-four gate **from the exact trajectory in every causal window**, then
maps its local child `j` to `(j + shift) mod 10` for shifts 0–9. Each
map is a bijection, so child traffic counts are permuted within each
fixed parent. The changed child output may alter later hidden states,
but replay keeps later parent IDs, gate scores and original child IDs
fixed too. Weights, A factors, token windows, and scorer are unchanged.
The exact arm reproduces all 24 stored METH-122 document nats with
**zero** absolute difference.

| Frozen measure, 24 previously consumed documents | Result | Gate |
|---|---:|---|
| Exact pooled BPB | 1.184887492 | reference |
| Nine shifted pooled BPB values | 1.185063382–1.185407237 | exact beats each: pass |
| Mean shift minus exact BPB | +0.000330135 | diagnostic |
| Paired 10,000-draw bootstrap fifth percentile | +0.000160145 | positive: pass |
| Category mean shift minus exact: code / prose / technical | +0.000178937 / +0.000249931 / +0.000561537 | diagnostic |

The [raw per-document result](meth183_e1280_child_route_alignment_result.json)
has SHA-256 `e9ae3ba02ecb342fc202146664cd034f5d33ccc3e1f28f4978e256504b16c7f5`.
Run from the repository root with:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth183_e1280_child_route_alignment.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth183_e1280_child_route_alignment_result.json
```

The valid local RTX 3060 run took 108.640 seconds with six host threads,
4.123 GB peak allocated GPU memory and 3.882 GB process RSS; it created
less than 1 GB of output. The first invocation with the system Python
stopped before imports because that interpreter lacks `psutil`; the
repository `.venv` was then used. No T4 was used.

An initial completed run used the `d321e05` implementation, which shifted
children during complete model forwards but let later parent IDs and
gates change in response. That violates the frozen comparison. Its
[raw artifact](meth183_e1280_child_route_alignment_unfrozen_parent_invalid.json),
SHA-256 `8c320592855b6e380ea1d7c14b062f3cb73e5a3ef20ffbbf78652d0df6655ffd`,
is retained as **invalid for this protocol**; none of its decision
statistics are used. The corrected runner captures exact parent IDs,
gate scores and child IDs separately for each window and replays them
for all nine rotations. The code correction did not change the protocol,
sources, gates or thresholds.

These 24 METH-121 sources were already used in the original external
quality adjudication. This is a diagnostic of *why* that quality-valid
E1280 bank works, not another independent quality test. Its BPB gains
are small and the category contributions uneven. METH-179's failed
route alignment belongs to a different, hash-routed E12,800 bank and
does not contradict this result. The next E12,800 proposal should keep
content-coupled learning while controlling load on source-held-out
contexts; it needs its own training, fresh quality and actual CPU route
cost before promotion. In parallel, a pretrained-to-native full-model
path still needs a quality-valid compact core and same-artifact speed.
