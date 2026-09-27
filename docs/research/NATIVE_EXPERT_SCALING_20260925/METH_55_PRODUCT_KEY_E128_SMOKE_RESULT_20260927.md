# METH-55: joint E128 product-key smoke passes the retention screen

The [precommitted protocol](METH_55_PRODUCT_KEY_E128_SMOKE_PROTOCOL_20260927.md)
starts from the same BF16 Qwen0.5B-Instruct donor as METH-44, with
24 fresh rank-64 product-key routers (8×16 E128) and distinct rank-8
residual experts. Every expert B factor begins at zero, giving **zero
maximum absolute logit difference** from the donor on both checked
raw and chat prefixes. Initial held-out raw BPB also matches the donor.
The candidate top-four pair search agrees with exhaustive 128-pair
search on random vectors before training and on actual hidden states
in the terminal chat evaluation.

The [new development manifest](meth55_product_key_dev_manifest.json)
uses seed 555555 and 24 rows excluded from the 256 chat-training rows
and METH-44/47's 48 prior development rows. On these 3,811 prompt
positions after 16 fixed updates, top-1 agreement is **3,658/3,811 =
95.985%**. Held-out raw BPB is 0.9697036 versus donor 0.9712552,
ΔBPB **−0.0015516**. B-factor gradients are nonzero in all layers
at every update (smallest norm 6.23e-4); router gradients are nonzero
from update 2 (smallest norm 3.59e-6), as expected with zero B init.
The minimum changed B slots per layer is 125/128. On the terminal
chat prompts, 108–128 expert pairs per layer receive at least one
selection. The worst layer's maximum load is **21.76×** its mean,
so broader expert use and balance remain unresolved.

All preregistered 16-update gates pass: ≥95% prompt top-1,
ΔBPB≤+0.05, ≥16 changed slots/layer, finite nonzero gradients and
the route oracle. The checkpoint is eligible for a separately frozen
longer continuation. This screen cannot establish semantic
conservation: METH-44 passed a similar early screen, then its first
long continuation failed chat retention. It also cannot establish
large-E trained quality or native throughput.

The [result JSON](meth55_product_key_e128_smoke_result.json) has SHA-256
`aa9a44d4e740b5224ee003855ddd8c66277c11ba77fb730a88f33aeb081cfdd8`.
The local [checkpoint](../../../benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth55_product_key_e128_smoke.pt)
has SHA-256
`2d652709c6730e2b4f3a1a543834d7eec7409ef0f374738ae24d57ae2a5b5bd3`.
The [training script](../../../benchmarks/donor_adaptation/s1/meth55_product_key_e128_smoke.py)
SHA-256 is
`dea5e17eafc6f751f89e236ec571e38e1207286f9fafb4f03083ef0c436d5a5e`;
the [router module](../../../benchmarks/donor_adaptation/s1/meth55_product_key_experts.py)
SHA-256 is
`4ea635b942a6f5330b77894a6fe6ab7bc3ab2aa8785694005ed5beba740d8eb0`.
Run command:

```powershell
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth55_product_key_e128_smoke.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth55_product_key_e128_smoke_result.json --checkpoint benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth55_product_key_e128_smoke.pt
```

Elapsed 38.56 s, peak GPU allocation 2,582,093,312 bytes, process
RSS 3,206,803,456 bytes on local RTX 3060 (PyTorch 2.6.0+cu124).
The checkpoint is kept locally because it is substantial binary
state, while its hash and source identity are recorded in JSON.
No T4 was used.

**Decision:** continue this *specific* joint geometry under a new
frozen longer-retention protocol with stronger donor KL, new
development rows and explicit routing-load telemetry. Then run a
source-grounded external semantic audit before exporting to native C.
