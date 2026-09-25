# METH-02: Granite 4.0 H Tiny base metadata screen

**Question.** Is an already sparse/recurrent pretrained donor a more tractable
first conversion case than the measured GigaChat Q4 baseline? This is a
metadata-only **candidate screen**, not a model run, quality result, format
payload measurement or engine compatibility pass. No weights were downloaded.

## Pinned sources and reproducible calculation

The official [IBM model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base)
reports a 7B-total/~1B-active hybrid with 36 Mamba2 and four attention layers,
64 experts/top-6 and a shared expert, under Apache 2.0. The pinned source
revision is `f95c8e83b06c12f877486f18d2e2d109ab252bd0`; its `config.json`
SHA-256 is `bda8fd574ace7d968d82397f59ea6b9a702a077bbeab279a65b9dad7386a82c6`.
The official [IBM GGUF repository](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base-GGUF)
is revision `27f75056a8aabab6377df1d9dd4f37c9b1fbe14e`. Its Q4_K_M file
is 4,230,969,888 B, published SHA-256
`7aff4ea112f1310a654ff39f2df3bc7191afee59b8f258822557de59720f5ac0`;
its BF16 GGUF is 13,891,199,808 B. The three source BF16 safetensors shards
sum to 13,878,142,256 B. These are **repository metadata**, not local hashes.

The existing [`donor_inventory.py`](../../../benchmarks/donor_adaptation/donor_inventory.py)
`analyze_one` function was applied to the pinned config and the same repo's
API safetensors count; no new analyzer was written. Reproduction from the
repository root (requires network access to the pinned official files):

```python
import json, urllib.request
from benchmarks.donor_adaptation import donor_inventory as inventory
repo = "ibm-granite/granite-4.0-h-tiny-base"
revision = "f95c8e83b06c12f877486f18d2e2d109ab252bd0"
with urllib.request.urlopen("https://huggingface.co/api/models/" + repo, timeout=15) as h:
    api = json.load(h)
assert api["sha"] == revision
with urllib.request.urlopen("https://huggingface.co/" + repo + "/resolve/" + revision + "/config.json", timeout=15) as h:
    config = json.load(h)
counts = api["safetensors"]
row = inventory.analyze_one(repo, config, {
    "safetensors_total_params": counts["total"],
    "safetensors_params_by_dtype": counts["parameters"],
})
print(row["status"], row["crosscheck"], row["params_total_computed"],
      row["params_total_advertised"], row["active_params_per_token"], row["partial"])
```

This read took under two seconds locally and no GPU/T4 time. It uses the
current API only to verify that `main` still resolves to the pinned revision;
if `main` changes, fetch the pinned revision metadata instead of silently
using a new checkpoint. The model card's rounded active count is not used in
the arithmetic.

## Result and choice

The analyzer returned `OK`, `AGREE`, no partial warnings:
**6,938,917,440** computed total parameters versus **6,939,037,248** in the
safetensors metadata, and **1,465,350,720** active large weights/token. The
config specifies 40 MoE blocks, 64 routed experts/top-6, a 1,024-wide shared
path, tied output/input embeddings, and 36 Mamba2 plus four attention mixers.
The existing analyzer reads its 512-wide routed expert size from
`intermediate_size`, explicitly noted in its output.

At **hypothetical uniform W4**, the active large-matrix payload is
732.675 MB/token, or 18.317 ms at the 40 GB/s yardstick before scales,
state, routing, compute or sampling. It exceeds the 560 MB/14 ms streaming
design allotment by 172.675 MB. The official Q4_K_M file's 4.231 GB size
does **not** identify its active bytes/token; a tensor-header ledger is still
needed. Mamba2 is architecturally relevant to the native SSM target, but
different operator semantics prevent treating it as an exact native port.

**Decision:** Granite H Tiny is a conditional *screening candidate* for the
first hybrid-source branch, not a selected conversion checkpoint. Its lower
total size and recurrence warrant an actual GGUF per-organ header inventory
before acquiring weights or budgeting training. If the actual active format
still exceeds the target, a stated transformation and paired quality test are
needed. Neither the model card nor this analyzer result demonstrates
donor-relative quality, ≥50 C tok/s, or a transfer method that generalizes.

**Follow-up:** [METH-03](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md)
completed the header inventory: the official mixed Q4 file addresses
913.314 MB/token under the stated no-cache model. The hypothetical uniform
W4 figure above remains a preflight, not the actual format result.
