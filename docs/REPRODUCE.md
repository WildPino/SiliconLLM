# Reproducing the results

Three levels, from zero-dependency to full engine parity. Levels 1–2 need no trained
weights; level 3 needs the release asset described below.

## Level 1 — regenerate the charts

Every figure in the README is drawn from numbers embedded in the script (sourced from the
recorded verdicts):

```sh
python scripts/make_readme_charts.py       # -> assets/*.png
```

## Level 2 — re-run the kernel microbenchmarks (no weights needed)

The ternary-speed and cache-cliff measurements, on the Zen 2 target:

```sh
clang -O3 -mavx2 -march=znver2 benchmarks/phase57/phase57_lutbench.c   -o lutbench   -lm
clang -O3 -mavx2 -march=znver2 benchmarks/phase57/phase57_cachesweep.c -o cachesweep -lm
./lutbench      # ternary/bit-serial LUT matvec speed + bit-exactness vs scalar ref
./cachesweep    # working-set bandwidth sweep (the 16 MB L3 cliff)
```

## Level 3 — reproduce the engine parity gates end-to-end

Build a stage and run its gates against the fp32 reference:

```sh
clang -O3 -mavx2 -march=znver2 benchmarks/phase60/e1_engine.c -o bin/e1_engine -lm
bin/e1_engine --all       # G1 golden-trace · G2 top-1 · G3 BPB · G4 tokenizer · G5 generation
```

Later stages build the same way (`e2_engine.c` … `e4_engine.c`); each stage's pre-registered
gates and exact rerun commands are in [`ENGINE_PLAN.md`](ENGINE_PLAN.md). This requires the
**release asset** (below) unpacked at the repo root: the engine loads the exported model
binary, and the gates compare against the PyTorch reference over the canonical validation slice.

---

## The reproducibility asset

Large binaries are not tracked in git; they are provided as a downloadable **release asset**
— [`siliconllm-repro-v0.2.0.zip`](https://github.com/WildPino/SiliconLLM/releases/download/v0.2.0/siliconllm-repro-v0.2.0.zip)
(~322 MB, `-0` stored, with a `MANIFEST.sha256`). Contents and integrity:

| file | role | sha256 |
|---|---|---|
| `results/phase57/sp58_base.pt` | 5M Arch-A checkpoint (E1–E3 target; gated-dReLU ternary MLP) | `8f8c54424eebe64bf234224b18aa6ab9caf4dc9045508e1f611c0c17f755f191` |
| `results/phase57/moe_gran.pt` | granular-MoE checkpoint (E4 target; E32×h128 top-8) | `356478b2f63ace9d6ec429056fee5c0e14e1c0f36253edbca36300eba02d4525` |
| `results/phase55/ids.u16` | canonical validation corpus — 32,723,845 BPE-1024 tokens (uint16) | `33b8cba2a26653599f7f87a4d8e05b38be051ba850d4cbc5d09b561aae133889` |
| `results/phase60/e1_model.bin` | E1–E3.5 engine model, magic `E1M1` (from `e1_export.py`) | `c50b7a6631bb79bdf1d17d43f7867140522ad77d520e1dd7d7d35b070bc2d579` |
| `results/phase60/e4_model.bin` | E4 MoE engine model, magic `E4M1` (from `e4_export.py`) | `52086303c95ddab3592ae8285a9420d93f7424c3a17dbcf234ac3769fe0347e1` |
| `weights/bpe1024.bin` | BPE-1024 tokenizer (merges embedded) | `97de713392eabfa4dff6ebd9dd5502f0d2f3e93232db8f349860bb57f3148f9b` |

The full bundle is `siliconllm-repro-v0.2.0.zip` (~322 MB, `-0` stored) with a `MANIFEST.sha256`.

Regenerate the engine export from a checkpoint:

```sh
python benchmarks/phase60/e1_export.py     # sp58_base.pt -> E1M1 binary
python benchmarks/phase60/e4_export.py     # moe_gran.pt  -> E4M1 binary
```

## Data recipe

The corpus is **TinyStories**, tokenized with the project's **BPE-1024** codec. The exact
tokenization is defined by the code (source of truth), not re-specified here:

- tokenizer/codec: `archive/benchmarks/phase50/bpe_codec.h`
- corpus → tokens: `benchmarks/phase55/phase55_tokdump.c` → `results/phase55/ids.u16` (+ `meta.bin`)
- model architecture: `benchmarks/phase55/phase55_ssm.py` (Arch-A: D256 N96 H8 L6, SWA@5, gated-dReLU ternary MLP, vocab 1024)

The validation slice used by the engine gates is the tail of `ids.u16` (offset/length are
defined in `e1_reference.py` / `e1_engine.c`); verify it by the hash above. *(TODO: pin the
exact upstream TinyStories snapshot in this file once confirmed.)*
