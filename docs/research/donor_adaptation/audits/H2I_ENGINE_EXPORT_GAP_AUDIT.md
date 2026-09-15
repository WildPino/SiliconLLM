# H2I → `engine.c` export-gap audit

**Date:** 2026-09-15

**Scope:** read-only compatibility audit while H2I Phase B v4 and H1 S3 v2 run.

**Status:** `CONTAINER_CAN_REPRESENT_IT; EXPORTER_CANNOT_YET_EMIT_IT`.

This is not a new experiment, an export, a quality result or a rate result. It records the exact
conversion seam that becomes eligible only if H2I's already-preregistered **combined score + rank
gate** passes. If score fails, the brief forbids rank as a rescue; if rank fails, addendum B.7 says
do not export or scale. In either case this file prevents somebody from rebuilding the same map.

## 1. The object the CPU adjudicator measures

`s1/h2i_eval.py:install_trained()` installs seven arrays for each selected layer from the returned
bundle: `gate`, `up`, `down`, `router`, `rms_in`, `rms_h`, and frozen `labels`. The selected layers
are exactly `[3, 6, 9, 12, 15, 18, 21, 24]`; `h2i_applied.build_ffn_int8()` leaves the other 20 FFNs
as donor fp32 modules. `h1_qat.build_qo()` installs H0's trained ternary low-rank `q_proj` and
`o_proj` factors on all 28 layers. Direct inspection of the pinned H0 archive finds 280 keys,
layers 0–27, organs exactly `{q_proj, o_proj}`.

At adjudication the selected FFN master weights are R8-quantized in the forward, while routing is
hard top-16 with the trained **fp32** router. Non-selected FFNs, `k/v`, norms, embeddings and the
tied head retain the donor representation. Thus “trained H2I” is a mixed per-layer and per-matrix
object. It is not “the donor with every FFN carved”.

References:

- `benchmarks/donor_adaptation/s1/h2i_qat.py:save_bundle`
- `benchmarks/donor_adaptation/s1/h2i_eval.py:install_trained`
- `benchmarks/donor_adaptation/s1/h2i_applied.py:build_ffn_int8`
- `benchmarks/donor_adaptation/s1/h1_qat.py:TernaryCarvedFFN.group_mask`

## 2. The C container is already expressive enough

`donor_engine.c`'s `quant==4` is tagged-v2. Every attention/head matrix is read through the
per-matrix tagged reader, and every layer separately carries `FK_DENSE` or `FK_CARVED`. A carved
layer then carries a tagged router, tagged row-selected gate/up, and tagged transposed down.
Available kinds include fp32, packed ternary, factored, `MK_I8`, and `MK_I8_T`. Therefore one file
can, without a new container version:

1. store H0's factored q/o on every layer;
2. store untouched matrices as fp32;
3. mark only the eight H2I layers `FK_CARVED`;
4. store their trained gate/up as `MK_I8` and down as `MK_I8_T`;
5. store each trained router either fp32 (exact adjudicated object) or packed (a separate lossy
   conversion that must be measured, never assumed).

The sparse runtime then computes only the selected rows and accumulates only the selected down
columns in `ffn_carved()`. No new sparse kernel is required by the representation audit.

References:

- `benchmarks/donor_adaptation/engine/donor_engine.c:read_mat`
- `benchmarks/donor_adaptation/engine/donor_engine.c:load`
- `benchmarks/donor_adaptation/engine/donor_engine.c:ffn_carved`
- `benchmarks/donor_adaptation/engine/e1_bpb_through_engine.py:layout_bytes_v4`

## 3. The exporter cannot currently emit that object

`engine/qwen_export.py` supports useful pieces, but not their composition:

| required H2I property | current exporter behaviour | gap |
|---|---|---|
| trained R8 FFN masters | always reads donor `lay.mlp.*.weight` | no H2I-bundle ingestion |
| only 8 carved layers | a non-null `--carve-labels` makes every layer `FK_CARVED` | no per-layer FFN-kind map |
| H0 factored q/o + tagged-v2 FFN | `--factors` is refused unless `--quant tagged` | needless CLI restriction; `quant==4` reader supports factors |
| donor-fp32 matrices outside the treatment | `--fp32-organs` applies by organ name globally | no per-layer/per-matrix preservation map |
| exact trained router | `--carve-router` requires `r0`…`r27` and always calls `w_tag_packed` | no selected-layer router map and no fp32-router mode |
| faithful run witness | `CONFIG`/`BENCH` call `quant==4` “ternary” | tagged-v2 is not audibly distinguished |
| mixed-format cost | sidecar reports active **weight count** | no exact active-byte charge by matrix kind |

The router row is load-bearing. H2I's gate names the fp32-router hard arm “deployable”, but today's
exporter ternarizes the router. That may change the top-16 set and has not been adjudicated. The
container can preserve the fp32 router with `MK_F32`; doing so changes streamed bytes and must be
charged explicitly. A packed-router arm would be a new conversion treatment and needs its own
parity/quality gate after the exact arm, not a silent default.

The `quant==4` witness defect is historical: E64's result records show `quant=ternary` even for
tagged-v2 carved-int8 files. Do not patch the binary underneath the still-owed, hash-pinned E63d
measurement. A future export probe must build a separately frozen engine and require
`quant=tagged-v2` (plus the relevant matrix-kind summary) in every engine output.

References:

- `benchmarks/donor_adaptation/engine/qwen_export.py:main`
- `benchmarks/donor_adaptation/engine/qwen_export.py:emit`
- `benchmarks/donor_adaptation/engine/qwen_export.py` carved write loop
- `benchmarks/donor_adaptation/engine/donor_engine.c` `CONFIG` and `BENCH` formatters
- `benchmarks/donor_adaptation/engine/results/e64_carve_on_int8_run2.json`

## 4. Conditional continuation contract

Only after canonical H2I adjudication returns `G-H2I-score = true` **and**
`G-H2I-rank = true` may a Phase-C brief register implementation. Its minimum object is:

1. a new exporter input dedicated to the hash-pinned H2I final bundle and sidecar;
2. exact validation of model/revision, eight layer ids, labels and every trained array before
   loading/writing the donor;
3. quant==4 with H0 factored q/o, untouched matrices fp32, exactly eight `FK_CARVED` layers,
   trained R8 FFNs, and fp32 trained routers for the identity arm;
4. independent layout-byte parsing and exact consumed-byte gate;
5. Python-versus-engine logits/BPB/rank parity on the frozen H2I slice, with a planted changed
   router or restored-donor FFN that must be detected;
6. only after the exact-router arm passes, an explicitly separate packed-router conversion arm;
7. a `CONFIG` witness that distinguishes tagged-v2 and records enough matrix-kind counts to audit
   the mixed file;
8. exact active-byte accounting. No rate extrapolation from 1.5B and no 10B claim.

This audit deliberately does **not** implement those changes while H2I is unresolved. It closes
the mapping question and makes the post-gate work finite; it does not move or weaken the gate.
