# METH-03: Granite H Tiny Q4_K_M active payload

**Question.** Does the actual official Q4_K_M layout give the pretrained
recurrent/sparse candidate a plausible direct path to the 14 ms streaming
allotment on the Ryzen 5 3600X? This is GGUF tensor-header arithmetic, not a
quality, C compatibility, DRAM-traffic or decoder-rate measurement. It follows
the [metadata screen](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md).

## Artifact and reproducible method

The artifact is the official IBM
[Granite 4.0 H Tiny base GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base-GGUF)
at revision `27f75056a8aabab6377df1d9dd4f37c9b1fbe14e`, file
`granite-4.0-h-tiny-base-Q4_K_M.gguf`, 4,230,969,888 B. Its published LFS
SHA-256 is `7aff4ea112f1310a654ff39f2df3bc7191afee59b8f258822557de59720f5ac0`.
Only its first 8,388,608 B were fetched; the full weight payload was **not**
downloaded or locally hash-verified. The server returned HTTP 206 with
`Content-Range: bytes 0-8388607/4230969888`. The local prefix SHA-256 is
`591f7b0af4794e781b401bb5bb85e5576415350cf415590db000dca8925548e7`.

From the repository root, the prefix can be fetched again with:

```python
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

revision = "27f75056a8aabab6377df1d9dd4f37c9b1fbe14e"
url = ("https://huggingface.co/ibm-granite/"
       "granite-4.0-h-tiny-base-GGUF/resolve/" + revision +
       "/granite-4.0-h-tiny-base-Q4_K_M.gguf")
with urlopen(Request(url, headers={"Range": "bytes=0-8388607"}), timeout=60) as response:
    assert response.status == 206
    assert response.headers["Content-Range"] == "bytes 0-8388607/4230969888"
    prefix = response.read()
assert len(prefix) == 8388608 and prefix[:4] == b"GGUF"
assert hashlib.sha256(prefix).hexdigest() == "591f7b0af4794e781b401bb5bb85e5576415350cf415590db000dca8925548e7"
Path("results/native_expert_scaling/granite_h_tiny_q4_header_8m.bin").write_bytes(prefix)
```

The [ledger tool](../../../benchmarks/native_expert_scaling/gguf_active_ledger.py)
reads the complete tensor descriptors in that prefix with pinned llama.cpp
`gguf-py` revision `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`:

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\gguf_active_ledger.py --header-file results\native_expert_scaling\granite_h_tiny_q4_header_8m.bin --profile granite_h_tiny_base --gguf-py C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4\gguf-py --expect-file-bytes 4230969888 --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth03_granite_q4_active_ledger.json
```

It enforces the 666-tensor Granite inventory, 40 layers, 64 routed experts,
top-6 selection, tied output/input embedding, and the four attention-layer
positions from the pinned config. It charges six slices of each stacked routed
tensor, all shared, mixer and router tensors, and the **whole tied embedding**
for the output projection. It checks that the 3,582,240 B payload offset plus
4,227,387,648 B of declared tensor payload equals the published file size.
Its 1,465,470,528 active elements/token agree to within 119,808 elements
(0.0082%) with the independent config analyzer; the small difference is
control tensors. The header-only path was cross-checked on the local GigaChat
Q4 file: all organ, active-element and stored-byte results exactly matched
the full-file parse. See the write-once [raw JSON](meth03_granite_q4_active_ledger.json).

The header fetch and parse took seconds of wall time, no GPU/T4 work, full
weight download, model execution or C benchmark. Local header prefix is under
`results/native_expert_scaling/` and is not needed in Git because the pinned
URL, response checks and prefix digest reproduce it.

## Result and decision

| Organ | Active compressed MB/token |
|---|---:|
| Routed experts | 342.835 |
| SSM matrices and controls | 298.969 |
| Tied output head/embedding | 126.444 |
| Shared expert | 114.278 |
| Router | 15.729 |
| Attention | 14.561 |
| Other controls | 0.498 |
| **Total** | **913.314** |

The file mixes Q4_K and Q6_K weights with F32 controls. The active compressed
payload is **913,314,048 B/token**. At 50 tok/s, reading those addressed
payload bytes once requires **45.666 GB/s**. At the measured 40 GB/s aggregate
DRAM yardstick in [PHASE64_BUDGET.md](../../PHASE64_BUDGET.md), that alone is
**22.833 ms/token**, before state, activation, compute, dequantization,
routing and sampling. The 14 ms/560 MB streaming design allotment needs a
**353.314 MB/token (38.68%)** reduction from this layout. Actual DRAM traffic
can differ with cache residence or repeated access, and effective kernels can
deliver a different rate. Neither effect was measured in this cell.

Routed experts and SSM account for **641.805 MB/token** together. Even an
ideal exactly 2-bit treatment of just their large matrices would leave only
about 547.5 MB/token, a 12.5 MB margin before bit-scale/padding overhead or
quality loss. The earlier [W2 PTQ failure](../donor_adaptation/probes/STRAT_02E_W2_BF16_EXPERT_SCOUT_RESULT.md)
concerned another donor and format; it does not decide this one. A credible
candidate therefore needs a more generous structural/precision budget and
an explicit paired quality test, followed by integrated C timing.

**Decision:** direct Granite Q4 execution does not satisfy the declared
40 GB/s, 14 ms streaming design budget. Keep it as a candidate source only if
a specific transformation can remove at least 353 MB of active payload with
margin for overhead and demonstrate donor-relative quality. Its Mamba2 mixer
is not the native SSM operator, so a C port alone would also require an
operator implementation or a measured substitution. This cell neither
selects a conversion checkpoint nor establishes model usefulness or ≥50 C
tok/s. No T4 work was started.
