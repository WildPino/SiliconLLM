# HANDOFF — find where the 1.5B fp16 NaN enters, on the GPU

updated: 2026-09-04   status: **READY. ~10 GPU-minutes. Decisive.**
written by: the Adapter/Principal

> **Read this file and nothing else. No prior conversation needed.**
> Repo `D:\_THINGS\Progetti\SiliconLLM`, branch `research/donor-adaptation`, at or after `95cdf33`.

---

## 1. What happened, in one paragraph

The S1 scale-arm run completed on Kaggle and its 1.5B GPU point came back with
`baseline_bpb = NaN`. The probe's own `C1_IDENTITY` control caught it — *"FAIL — HARNESS IS WRONG,
STOP"* — but the **A1.2 anchor gate certified PASS anyway** and the ladder continued to 7B/14B.
Both bugs behind that are now **fixed and pushed** (`95cdf33`); what is *not* known is **where the
NaN comes from**, and that needs a GPU. This handoff is that one job.

## 2. What is already ruled out — do not re-check these

Two hypotheses were tested locally, on CPU, over the same shared eval slice
(`ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`):

| hypothesis | test | result |
|---|---|---|
| **fp16 overflow of the FFN intermediate** | `s1/fp16_range_diag.py` — fp32 pass, per-layer max\|h\| vs the fp16 limit | **REFUTED.** worst \|h\| = **4541.5** vs max **65504** = **14.4× headroom**, 0 values over the limit in all 28 layers |
| **fp16 arithmetic anywhere in the stack, or sequence length** | `s1/fp16_first_nan.py` — full fp16 forward **on CPU**, every one of 284 leaf modules hooked, at 128 **and** 512 tokens | **REFUTED.** Everything finite, both lengths. fp32 and fp16 agree to ~4 significant figures on every large tensor (e.g. `layers.26.mlp.down_proj` 5733.3 vs 5736.0) |

> **Conclusion: the NaN is not a property of the numbers. It is a property of the GPU execution
> path.** Same weights, same slice, same dtype, same sequence length — finite on CPU, NaN on T4.

## 3. One anomaly worth a glance while you are there

The run manifest records, on **Tesla T4, capability 7.5**:

    "bf16_supported_achieved": true

**T4 is Turing and has no native bf16.** Recent PyTorch returns `True` from
`torch.cuda.is_bf16_supported()` on Turing because it counts emulation. That flag is *reported*
by our code, not *acted on* — but if anything in the stack (torch 2.10.0+cu128, transformers)
branched on it, that is a candidate. Check whether any tensor in the failing run is bf16.

## 4. The job — one kernel, one GPU, ~10 minutes

Run `s1/fp16_first_nan.py` **on the GPU**, across attention backends, and report which one goes
non-finite and at which module. The script already prints the first non-finite module plus the six
before it; it needs only a device and a backend loop.

```python
# on Kaggle, single T4 is enough -- 1.5B fp16 is ~3.1 GB
import os, torch, itertools
os.environ["S1_RESULTS"] = "/kaggle/working/nanhunt"
# for each of these, load the model and run the walk from fp16_first_nan.py:
#   attn_implementation = "sdpa"    <- what the failing run used by default
#   attn_implementation = "eager"   <- the reference path
# and, with sdpa, each backend forced in turn:
#   torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
#   ... .EFFICIENT_ATTENTION
#   ... .FLASH_ATTENTION
```

**Report, per configuration:** finite or not; if not, the module index and name of the **first**
non-finite output and the six modules before it; and the dtype actually in use on the parameters
(`{p.dtype for p in model.parameters()}`) so §3 is answered at the same time.

**Also run one control:** the same walk in **fp32 on the GPU**. If fp32-on-GPU is also non-finite,
the problem is not the dtype and not the attention backend, and that is a different and more
serious finding.

### Pre-registered reading of the outcome

| what you find | what it means |
|---|---|
| `eager` finite, `sdpa` non-finite | an SDPA backend on sm_75 is the culprit. Fix: pin `attn_implementation="eager"` for the whole ladder and re-run. Cheap, and it does not touch the measurement |
| one specific SDPA backend non-finite, others fine | pin the working backend explicitly rather than relying on the dispatcher's choice |
| all fp16 configurations non-finite, fp32-on-GPU finite | it is fp16 **on this GPU**, not fp16 as such (CPU fp16 is clean). Then A1.3's "fp16 on GPU" is not achievable on a T4 for this donor and **that is itself the finding** — report it, do not work around it |
| fp32-on-GPU also non-finite | not a dtype problem. Stop and report; something else is wrong |

## 5. What NOT to do

- **Do not re-run the 1.5B anchor or the ladder yet.** The two silent-degradation bugs are fixed,
  which makes the failure **loud**, not absent. A re-run before this diagnosis just fails louder.
- **Do not "fix" it by clamping, rescaling, or upcasting inside the probe.** This probe measures
  activation statistics; changing the arithmetic changes the object under test. A1.3 allows fp16 on
  GPU and fp32 on CPU and nothing else.
- **Do not touch `l1_keep_count` or `check_anchor`.** Both were repaired in `95cdf33` and both now
  carry planted controls. If either fires, that is them working.
- **Read the JSON, not the logs.** Kaggle duplicates subprocess stdout 2–3×.

## 6. Why this is worth 10 GPU-minutes

The whole scale arm exists because every FFN-sparsity result this programme owns was measured on
**one** donor at **one** size, and the only published measurement on the size axis runs the *other*
way. Until 1.5B reproduces on the GPU path, A1.2 cannot certify anything and 7B/14B are not
interpretable. This is the last thing between here and a real scale trend.

## 7. Context you may want but do not need

- The two repairs, in `95cdf33`: `check_anchor()` now hard-fails on any NaN/Inf/missing value and on
  fewer than 20 finite comparisons, and carries `selftest_anchor_gate()` — 7 planted cases, run in
  `main()` before any model loads. The **old** code fails 3 of those 7, reproducing the field
  failure exactly (`worst = 0.0` over 51 comparisons, verdict PASS).
- `l1_keep_count()` now raises `FloatingPointError` on non-finite input. Previously one NaN in a row
  drove `(cs < NaN).sum() + 1 = 1`, and `clamp_(1, F)` turned that into "keep 1 neuron" — achieved
  sparsity `8959/8960 = 0.9998883928571428`, which is exactly what the manifest reported on layers
  1–27, matching to the last digit.
- Budget so far: ~0.40 GPU-h on acct1 against a 30 h/week ceiling.
