# Frozen recurrent-channel/full-skip comparison

12 October2026. Frozen before observations. All48 histories, sites0/12/23,
fixed8/adaptive384 existing F32 maps and stored actual 384-channel scan latents.
No fit, source forward, new scan, generator change, optimizer or native call.
See [algebra and decisions](RECURRENT_ONLY_CHANNEL_SKIP_NEXT_20261012.md).

Controlled arms use F64 arithmetic on promoted frozen F32 maps/latents and
captured BF16/F32 operands. Projected skip: `V ym`. Full skip:
`V(ym-D W^T x)+D x`. Both use full source SiLU gate, source norm gamma and
out_proj. Reconstructed-denominator mode normalizes the COMBINED gated output.
Source-denominator is explicitly an oracle. Both modes round normalized input
to BF16, multiply actual source weights in F64, then round output to BF16.
This is a precision-controlled local operator diagnostic, not a native export.

Repeat the old F32/BF16 operator independently from stored scans and compare
every old scalar to the prior balanced-head report: absolute<=1e-4. Original
source-output reconstruction relative RMS<=1e-4. F64/BF16 versus old F32/BF16
output RMS<=1e-3; report this precision bridge separately from skip effect.
Recurrent/skip error identity F64 relative<=1e-10. Store unnormalized error
energies, inner product and source energy; retain all four outputs per layout,
all cases/sites. No cherry-picked label or early failure audit substitution.

Original local DEV budgets unchanged: equal-case output RMS mean<=.1,
worst case<=.2, worst domain mean<=.2, centered output RMS mean<=.1.
All labels/positions are retained; no model-quality admission from these gates.

Independent CPU audit uses per-head NumPy contractions, sigmoid-derived SiLU,
independent BF16 round-to-nearest-even integer conversion and blocked full F64
out projection. Compare every stored output word (relative RMS<=1e-4), all
metrics (absolute<=1e-4), all energy/cross terms (relative<=1e-8), full aggregates
and decisions, all bound hashes before/after. These are numerical checks, not
bit-exact CUDA replay. If a numerical tolerance fails, retain fault and repair
under a new frozen code/namespace; do not loosen gates after observation.

Producer900s/audit1200s, OS6GiB, CUDA allocated2GiB/reserved3GiB, outputs3GiB.
Freeze principal runtime binaries/version/GPU identity via existing qualified
runtime binding, plus actual operands/code/protocol. Windows clock and desktop
sharing make throughput admission false. Preserve engine/foreign work.
