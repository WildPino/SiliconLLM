# METH-185: direct pretrained FFN channel selection loses too much output

**Decision: reject this direct channel-selection rule.** On 6,144 real
BF16 Qwen2.5-0.5B-Instruct E1280 pre-MLP states, retaining even 2,048
of 4,864 SwiGLU channels gives 16.49% median and 20.85% 95th
percentile FFN output relative L2 error. Both miss the frozen 1%/5%
screen by wide margins. The rule uses full exact gate/up activations
to choose channels, so merely approximating this same ranking with a
cheap router cannot remove the measured truncation error.

The [protocol](METH_185_PRETRAINED_FFN_CHANNEL_SPARSITY_PROTOCOL_20260930.md)
was committed at `78aa225` before execution, and the
[runner](../../../benchmarks/native_expert_scaling/meth185_pretrained_ffn_channel_sparsity.py)
at `53238be`. The donor revision is
`7ae557604adf67be50417f59c2c2f167def9a775`, source SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The METH-125 state file SHA-256 is
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
Each layer uses the original BF16 gate/up/down matrices, and K=4,864
reproduces the unmasked output bitwise on every state. The selector
ranks `abs(SiLU(gate) × up) × ||down column||₂`; the static control
ranks the same column norm without activation.

| Retained channels K | Dynamic median relative L2 | Dynamic 95th percentile | Static median relative L2 | Ideal selected BF16 FFN bytes/token |
|---:|---:|---:|---:|---:|
| 128 | 67.72% | 79.85% | — | 16.515 MB |
| 256 | 59.14% | 70.48% | — | 33.030 MB |
| 512 | 47.81% | 57.82% | 91.54% | 66.060 MB |
| 1,024 | 33.33% | 40.96% | 85.03% | 132.121 MB |
| 2,048 | **16.49%** | **20.85%** | 73.09% | 264.241 MB |

The [raw result](meth185_pretrained_ffn_channel_sparsity_result.json),
SHA-256 `cb60d11b8e981d9ffadbb0a76a6ac2bc7c598a22eb8e83799488363c7abb72d1`,
contains all per-state errors and per-layer medians. None of the
K=512/1024/2048 arms passes the joint gate. Dynamic selection does
beat the static norm list by at least 2× median at K=1024 and 2048,
so state dependence matters, but the retained function is still far
from the specified component fidelity. The byte column is the
arithmetic best-case stream of selected BF16 gate/up/down rows; the
measured runner actually computes every gate/up channel and therefore
does not save those bytes or establish speed.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth185_pretrained_ffn_channel_sparsity.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth185_pretrained_ffn_channel_sparsity_result.json
```

The local RTX 3060 runner took 1.969 seconds after Python imports,
including source binding. Peak GPU allocation was 101.050 MB and
process RSS ended at 1.683 GB; no T4 was used. The 256 states per
layer were captured from 24
previously consumed external prompts. This is a component mechanism
diagnostic, not a source-held-out full-model quality result, native
DRAM trace, accepted-token rate or 10B/100B transfer.

The protocol title's "upper-bound" term was clarified after the run
without changing the rule or gates: access to exact activations favors
the selector, but ranking individual contribution norms is **not**
the mathematical optimum over all K-channel subsets because channel
outputs can cancel. This result rejects the concrete norm-ranked
top-K transformation at K≤2,048. It does not reject trained sparse
FFNs, other partitions, or a jointly adapted core. The next compact
path must change the representation or train the core and conditional
capacity together; direct channel omission is insufficient here.
