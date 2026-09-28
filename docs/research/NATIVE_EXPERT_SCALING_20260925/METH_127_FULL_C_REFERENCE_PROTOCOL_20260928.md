# METH-127: full native Qwen plus centered E1280 reference composition

## Uncertainty, prior and decision

METH-121/123 retain BF16 donor-relative quality and METH-126 runs the
centered E1280 factors exactly in a C component, but no native program has
executed attention, KV cache, dense donor core, routing, factors and logits
together. The existing Qwen C runtime has attention/RoPE/KV and a dense FFN;
the historical phase60 `engine.c` has no Qwen attention. This experiment adds
the factor bank to the existing Qwen runtime as a **full-path reference**, not
as the final efficient `engine.c` conversion. The FP32 donor core is expected
to miss 50 accepted tokens/s from its active-byte lower bound. The result
will decide whether full native composition is numerically and operationally
sound, and identify the measured cost that a later compact core must remove.

## Bound inputs and checks

- Donor: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, source SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
  Export its stored BF16 weights widened to FP32 with `qwen_export.py
  --quant fp32 --fold none --load-dtype bfloat16`. Record the binary hash.
- Experts: METH-126 `M126FB01` bank SHA-256
  `1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`.
  Require Qwen L24/D896, dense FFN and exact bank dimensions before native
  execution. Add the selected residual to each layer's dense FFN output from
  the same post-attention normalized activation.
- First establish dense Qwen C versus PyTorch FP32 donor logit parity on a
  bound 8-token prompt: top-1 at all positions and relative L2 error <=1e-3.
  Stop if this fails. Check combined C output is finite and differs from dense
  C on the same prompt. Run a planted invalid bank/header control that fails.
- Measure 64 single-token decode steps, one and six CPU threads, after warmup,
  with the same core and bank, recording accepted-token rate, organ profile,
  physical file bytes and peak RSS. Report dense C as matched control. If
  quality scoring is attempted, keep it explicitly diagnostic until the
  native arithmetic is matched to the BF16 quality-gated model; full quality
  cannot be inherited solely from component parity.

## Budget and stop

Local CPU/RTX 3060 only; no T4. Export and measurements may use at most
5 GB new disk, 16 GB process RSS and 45 minutes. Stop on a source hash,
format, dense parity, finite-output or negative-control failure. Do not
promote FP32 full C speed to the objective even if it runs: quality and
>=50 accepted tokens/s still have to pass on one compact native artifact.
