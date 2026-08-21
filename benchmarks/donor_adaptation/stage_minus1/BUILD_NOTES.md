# Stage −1 pre-registration — build notes

**Builder**, branch `research/donor-adaptation`, 2026-08-20. Nothing committed, nothing pushed,
no install run, no arm run. `-ffast-math` not used anywhere.

## What was built

| file | what it is |
|---|---|
| `prereg.yaml` | the pre-registration as a **populated artefact**, 354 required leaf fields, every one either a concrete value or a string starting `UNPINNED:` + reason |
| `pin_slice.py` | downloads the eval corpus at its pinned revision, computes sha256 / byte range / token geometry / **B**, and **writes the values back into `prereg.yaml`** |
| `check_prereg.py` | the mechanical guard: reads `prereg.yaml` against a **schema embedded in the script**, exits non-zero listing every ABSENT / UNPINNED / MALFORMED field |
| `exercise_check_prereg.py` | fires the guard in both directions and writes `exercise_output.log` |
| `check_prereg_output_real.log` | the guard's output on the real file |
| `exercise_output.log` | the bidirectional exercise transcript |
| `pin_slice_report.json` | machine-readable sidecar of the 16 values `pin_slice.py` pinned |

**Why the schema lives in `check_prereg.py` and not in `prereg.yaml`.** If the required-key list
were derived from the artefact at runtime, deleting a key would delete its own requirement and the
guard would stay silent — the exact failure it exists to catch. The contract lives outside the thing
it checks. Exempting a field requires a visible diff in `STRUCTURAL_EXEMPTIONS` (currently empty),
never a sentence in prose.

## The guard, exercised in both directions

`exercise_check_prereg.py` → **all 5 assertions PASS** (`exercise_output.log`):

| case | fixture | exit | result |
|---|---|---|---|
| 1 | the real file | 1 | 0 absent, **34 unpinned**, 0 malformed, 320/354 pinned |
| 2 | `donor.licence` **deleted**, `donor.gated` set `UNPINNED:` | 1 | 1 absent, 35 unpinned — **both named by path** |
| 3 | all 34 UNPINNED replaced by values | 0 | `COMPLETE — 354/354 pinned`, silent |

The guard caught a defect in **itself** on first run: eight fields under `environment.precision_flags`
were keyed with literal torch attribute paths (`torch.backends.cuda.matmul.allow_tf32`), and a key
containing a dot is unaddressable by a dotted-path resolver — reported ABSENT while the value sat in
the file. Keys renamed to underscore form with a `torch_attribute_map` recording the real names, and
a **MALFORMED** class added so no future dotted key can hide the same way.

## What `pin_slice.py` measured (B1, open through three reviews — now values)

Every field below was written into `prereg.yaml` by the script, not typed by hand.

- corpus `wikitext` / `wikitext-2-raw-v1` / **test**, revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- join `"\n\n".join(test['text'])` — **read from `pt2_llm/data.py@9e943e6`**, not chosen
- sha256(joined) `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83` — 1,296,370 bytes
- sha256(**evaluated** bytes) `70a209ab2213ab3302b7ad1ded8354eb6f6b0845e330c795e0a300ac370ed8e4` — byte range `[0, 1296016)`
- 299,078 tokens → **146 windows** × 2048, 70-token tail dropped, 298,862 target tokens
- context 2048, **stride 2048 (non-overlapping)**, tail dropped, no per-window BOS — all read from `pt2_llm/eval_ppl.py@9e943e6`
- BOS **verified empirically**: the Qwen2.5 tokenizer has `bos_token_id = None` and prepends nothing (note: `config.json` declares `bos_token_id: 151643` — config and tokenizer disagree, and the tokenizer is what runs)
- fast/slow tokenizer parity **PASS** on the first 200k chars (the method builds its tokenizer `use_fast=False`; `pin_slice.py` needs the fast one for offsets)
- decontamination: **1/80,995 shingles (0.0012%)** vs the calibration source, **0/80,995** vs the pinned P62 code-val — both far under the 0.5% threshold
- **B = 4.334308 bytes/token** (target region), 4.334555 whole-stream, agreeing to 0.006%

### The finding that follows from B, and it is load-bearing

**B is 4.334, not 4.0 — and the anchor table was never a single-B object in the first place.** The
LLaMA SentencePiece tokenizer gives **B = 3.7965** on the *same bytes*, a **14.2%** difference from
Qwen's 4.334. Six of the seven PT²-LLM rows are LLaMA models and must divide by 3.7965; the Qwen3-14B
row must divide by a Qwen-family B. Recomputed per-tokenizer:

| model | b/byte at its own B | (at the old B = 4.0) |
|---|---|---|
| LLaMA-13B | 0.221 | 0.210 |
| LLaMA-65B | 0.239 | 0.227 |
| LLaMA-2-70B | 0.242 | 0.229 |
| LLaMA-7B | 0.264 | 0.251 |
| LLaMA-2-7B | 0.284 | 0.270 |
| **Qwen3-14B** | **0.316** | 0.342 |
| LLaMA-3-8B | 0.630 | 0.597 |

Frontier range **0.221 – 0.630**. Note what moves: the Qwen row was the argument that killed Δ\* = 0.30
(0.342 > 0.30). Tokenizer-corrected it is **0.316** and sits *mid-pack*, not second-worst. The
conclusion (cut Δ\*) still stands — a threshold cannot separate three causes with one bit of output —
but **the specific number it was argued from was off by 8%**, and any future threshold argued from the
uncorrected column would be arguing from a wrong number. This is the same class as the "fourth
instance" finding, one level deeper.

## What is still UNPINNED, and why — all 34, exactly as the guard reports them

**Blocked on a decision nobody has made (25 fields).** `bit_width_sweep.per_point_config.{4bit,3bit,2bit}.*`
— 21 fields — plus `ternary_1p58.scale_dtype`, `calibration.extended_arm.{composition,content_hash}`,
and `controls.control_1a.{anchor_paper,anchor_model,numeric_tolerance}`.

- **The five-point sweep cannot be run with PT²-LLM.** `quantize.py@9e943e6` restricts
  `low_quant_method` to `['atq','atq-itf','atq-aga','ternary-init','fp16']`. **There is no 4-bit,
  3-bit or 2-bit path — it is a ternary-only method.** Any 4/3/2-bit point needs a *different*
  method, which makes the curve a multi-method comparison — precisely B7's defect, now confirmed
  from the code rather than suspected from the papers. Principal's call: label every point with its
  method, reduce the stage to {bf16, ternary} + the 4-bit control-1a anchor, or find one method
  family spanning 4→1.58 bits. **Not a Builder decision.**
- **`ternary_1p58.scale_dtype`**: the released code does *fake* quantization — dense float tensors
  written back, no packed format ever serialised. The storage dtype of (α, μ) is a **reporting
  choice the artefact cannot settle**, and the paper's "1.58 bit" headline silently omits it. Both
  consequences are pinned so the x-axis cannot move quietly: **1.835 bits/weight at fp16 scales,
  2.085 at fp32** (group size 128 read from the code, per-(row,group) affine pair).
- **`calibration.extended_arm.composition`**: "composition mixed, not code-only" is not a
  specification (F2, claimed closed in prose, never written). I will not invent a corpus composition.
  Until it is pinned the extended arm may not run; the **anchor arm is fully pinned** at the paper's
  own 128 × 2048 WikiText-2-train configuration.
- **`control_1a.anchor_paper` / `anchor_model` / `numeric_tolerance`**: review #1's B3 required
  "name the paper, table and row"; three revisions later it is unnamed. A tolerance for an unnamed
  anchor would be post-hoc by construction — the thing B3 rejected. What I *did* pin: the tolerance's
  **form** (absolute PPL band) and a **defensible default of 0.10 absolute PPL** for a 7B-class
  WikiText-2 number, for the Principal to adopt or replace.

**Blocked on the install / a forward pass (5 fields).** `method.cpu_only_feasibility_smoke_test.status`,
`controls.control_4.architecture_numeric_probe_value`, `environment.harness_identity_recorded.{torch_version,cuda_runtime_version,cudnn_version}`.

**Blocked on the run itself, correctly (2 fields).** `measured_prerequisites.BPB_donor.value` and
`PPL_donor.value`. These are *outputs*. They are UNPINNED with that reason and the guard will keep
saying so until a run fills them.

**Control 1's numeric tolerance — promised three times, and here is a number.** Control **1b**:
`|PPL_ours − 11.39| ≤ 0.35` absolute PPL on the ternary leg, `|PPL_ours_fp16 − 5.68| ≤ 0.06` on the
baseline leg. 0.35 is **derived, not chosen**: the paper's own calibration-size ablation on
LLaMA-2-7B (64→11.92, 128→11.56, 256→11.35) shows a 0.36 PPL move for a 2× calibration change, so a
same-configuration reproduction must land inside one such step. Control **1a**'s tolerance stays
UNPINNED because its anchor is unnamed — that is a real gap, not a formatting one.

Every other control now carries a number: lesion ladder (ε grid of 10, fires iff the non-redundant
anchor rung exceeds *h* at ε ≤ 0.05, monotone within 1·SE, with a pre-registered response if it does
not fire); exact comparator (identical leg **exactly** 0.0, differing leg ≥ 1e-6); named refusal
(1 refusal / 0 refusals, binary); calibration-influence (**k = 3**, `≥ 3·SE`, expected direction
stated); device parity (**τ = 0.1·h**, a formula now and a number after the replicate step, and it
does **not** inherit the struck 0.007); precision probe (flags off ≤ 3e-7 vs fp64, flags on ≥ 1e-4,
both legs must land or the instrument has not been shown to fire).

## Task 4.1 — control 1b and the 12 GB ceiling

**The premise is wrong, and the code says so.** PT²-LLM **never holds the whole model on the device**.
`quantize.py@9e943e6` and `pt2_llm/eval_ppl.py@9e943e6` both run **layer-sequential**: the model
lives on CPU, one decoder layer at a time is moved to the GPU (`layers[i].to(dev)` … `layers[i] =
layer.cpu()`), and only the `inps`/`outs` activation buffers stay resident. That is how the paper
quantized a 70B model on one A800.

Peak VRAM for LLaMA-7B is therefore **one layer + two activation buffers**, not 13.5 GB:

- one LLaMA-7B decoder layer at fp16 ≈ **0.40 GB**
- calibration buffers: 128 × 2048 × 4096 × 2 B × 2 = **4.3 GB**
- eval buffers: WikiText-2 test is **341,469 tokens** under the LLaMA tokenizer (measured by
  `pin_slice.py`, not estimated) → **166 windows** → 166 × 2048 × 4096 × 2 B × 2 = **5.6 GB**
- peak ≈ **6.0 GB** including `lm_head` and workspace

**Verdict: control 1b is viable on the RTX 3060 unchanged.** Nothing is dropped, offloaded further,
or 4-bit-loaded. Cost: 13.5 GB download (1.3 TB free), 13.5 GB CPU RAM (80 GB available), and an
**estimated 3–5 h** for the quantize pass plus ~30–60 min for the two eval legs — the paper's 32 min
on an A800, scaled by a 5–8× slower card. **That is an estimate, not a measurement**; the one-layer
smoke test is what converts it. Fallback if wrong: pure CPU (`--device cpu`, same layer-sequential
code, 80 GB RAM), estimated 20–60 h, not memory-blocked. **A 4-bit load of the anchor is rejected** —
control 1b reproduces a published fp16→ternary number, and loading the baseline at 4 bits destroys
the quantity being reproduced.

## Task 4.2 — the LLaMA-2 gate

**Checked, not bypassed.** Unauthenticated HTTP only; no authentication attempted. An `HF_TOKEN` is
present in this environment and I **did not** use it to probe the gate.

- `api/models/meta-llama/Llama-2-7b-hf` → HTTP 200, **`gated: "manual"`**, sha `01c7f73d…`
- `meta-llama/Llama-2-7b-hf/resolve/main/config.json` → **HTTP 401**
- `meta-llama/Llama-2-13b-hf` → likewise `gated: manual`

**Plainly: LLaMA-2 is not accessible from this environment** without the owner accepting Meta's
licence on HuggingFace. If left unresolved, control 1b against the LLaMA-2-7B row is
**UNVALIDATED-AGAINST-PUBLICATION**, and that must be reported with every number the stage produces.

**Resolution adopted, and it needs ratification.** Control 1b is re-anchored to **LLaMA-7B** via
`huggyllama/llama-7b` @ `4782ad278652c7c71b72204d462d6d01eaaf7549` — **verified `gated: false`**.
LLaMA-7B is **a different row of the same table of the same paper at the same bit-width**
(5.68 → 11.39, 2.005×). Control 1b therefore remains a reproduction of a published number on the
published model, and the UNVALIDATED declaration is **not** triggered. This is a substitution, not a
workaround of the gate.

**Rejected, and recorded rather than silently taken.** Ungated third-party mirrors of LLaMA-2 exist
(`NousResearch/Llama-2-7b-hf`, `gated: false`) and would restore the exact LLaMA-2-7B row. Obtaining
licence-gated weights through a mirror is a **licence question, not a technical one**, and it is the
owner's call, not the Builder's.

## Task 5 — environment

**Current state:** `torch 2.12.0+cpu`, `torch.cuda.is_available() == False`, Python 3.12.10,
`transformers 4.57.6`, driver **610.47**, 80 GB RAM, 1.3 TB free. `datasets` and `accelerate` are
**not installed**.

**Both GPUs are visible**, and the enumeration is the trap the device pin exists for:

| nvidia-smi index | name | UUID | VRAM | cc |
|---|---|---|---|---|
| 0 | GeForce **GTX 1660** | `GPU-35e31090-c6c3-a989-adfc-540f03c7d9e5` | 6144 MiB | 7.5 |
| 1 | GeForce **RTX 3060** | `GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221` | 12288 MiB | 8.6 |

The 3060 is index **1**, not 0. An index-based pin or a `get_device_name(0)` assertion would land on
Turing. `prereg.yaml` pins `CUDA_VISIBLE_DEVICES` **by UUID** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
plus `device_count() == 1`, capability `(8,6)`, UUID read-back, a materialised-parameter device
assertion, and `device_map="auto"` forbidden.

**Recommended install — NOT RUN:**

```
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0+cu126
```

Wheel verified to exist: `torch-2.12.0+cu126-cp312-cp312-win_amd64.whl`, sha256
`194f5bd0721b968e769777b8ab4dbe51dd7ffdfdf295db045093b94a1b9765bb`. `+cu130` also exists;
**cu128 and cu129 have no 2.12.0 cp312 win_amd64 wheel**. cu126 is the conservative choice for
sm_86, and driver 610.47 (a CUDA 13.x driver) is backward-compatible with a 12.6 runtime; cu130 is
the fallback if a kernel-launch failure appears. Pinning **the same 2.12.0** keeps the CUDA backend
the *only* variable the install changes.

Then, separately: `python -m pip install datasets accelerate sentencepiece protobuf`.

**The method needs its own venv.** `requirements.txt@9e943e6` pins `torch==2.4.0`,
`transformers==4.44.2`, `tokenizers==0.19.1`, `accelerate==0.32.1`, `datasets==2.19.2`. Its
quantize/eval path passes `position_embeddings` into decoder layers — a 4.43-refactor API changed
again since. Running the method against `transformers 4.57.6` without a compatibility smoke test is
pre-registered as forbidden.

**`torch.cuda.is_available()` is not sufficient post-install verification.** Required before any
arm: `device_count() == 1` under the UUID pin, capability `(8,6)`, UUID read back, a materialised
parameter's device asserted, and control 7 fired in both directions.

## Things I did not do

- Did not touch `docs/research/*.md` or `gemv_donor_bench.c`.
- Did not commit or push.
- Did not run the install.
- Did not authenticate to HuggingFace or probe a gate with the token in the environment.
- Did not invent values for `control_1a`'s anchor, the extended calibration composition, or the
  4/3/2-bit configs. They are UNPINNED with reasons, and `check_prereg.py` reports them every run.
