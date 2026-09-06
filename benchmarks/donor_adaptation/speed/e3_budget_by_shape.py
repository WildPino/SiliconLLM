#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive the 50 tok/s active-weight budget PER DONOR SHAPE from E3's measured f.

Brief BRIEF_E3_ENGINE_AT_TARGET_SCALE.md s6: "RESERVATION-BREAKS -> the budget is re-derived from
measured `f` at each shape, and INDEX s7's twelve-donor screen is re-run and marked superseded."

This supersedes `donor_speed_budget.py` for the attention/KV term ONLY.  That script's own docstring
says "engine.c has never executed a transformer ... every attention row below is priced by ANALOGY
to the proj-GEMV path, and that analogy is the single largest source of error here."  E3 measured it,
so the analogy is retired; nothing else in that script is touched here.

WHAT IS MEASURED AND WHAT IS FITTED.  Six shapes x two context lengths were measured end to end
(probes/E3_ENGINE_AT_TARGET_SCALE.md s3).  A donor not among those six needs f at ITS shape, so f is
FITTED on those twelve points and the fit's residuals are printed first.  A fit whose residuals are
not small has no business being extrapolated, and the residual table is the licence to use it.

    f(shape, ctx) = A * L*NH*HD*pos  +  B * L*D          [+ rope, which is <0.1% and folded into B]
                    ^ attention: E3 s4.6 measured one FMA per ~4 cycles per thread, invariant to KV
                      size, GQA and vocabulary, so the attention term is a COUNT OF FMAs, not bytes.
                    ^ norm+glue: proportional to L*D.
    pos = (ctx+1)/2, the mean position over `--bench ctx`.

    budget(shape, ctx) = r_w * (20 ms - f)      r_w = weights delivered per ms in the weight organs.

r_w is taken from E3's measured trend: it RISES with matrix size (29.4 -> 33.8 G-w/s) as per-call
overhead amortises, and is flat at 33.8 for every shape at or above 7B.  Donors below that get the
measured value of the nearest measured shape, named in the output.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(HERE, "..", "results", "e3", "arms.json")
CONFIGS = os.path.join(HERE, "..", "configs")

# E3 s3, measured.  (D, F, L, NH, NKV, HD, V)
MEASURED_SHAPES = {
    "S05": (896, 4864, 24, 14, 2, 64, 151936),
    "S15": (1536, 8960, 28, 12, 2, 128, 151936),
    "S3":  (2048, 11008, 36, 16, 2, 128, 151936),
    "M7":  (4096, 14336, 32, 32, 8, 128, 32768),
    "Q8":  (4096, 12288, 36, 32, 8, 128, 151936),
    "T10": (4096, 14336, 48, 32, 8, 128, 32768),
}
# E3 s4.2, measured r_w in G-weights/s per shape, keyed by active weights/token (B).
R_W_BY_SIZE = [(0.494, 29.4), (1.544, 31.2), (3.086, 31.8),
               (7.114, 33.8), (7.568, 33.9), (10.603, 33.8)]


def active_weights(D, F, L, NH, NKV, HD, V):
    per = NH * HD * D + 2 * (NKV * HD * D) + NH * HD * D + 3 * D * F
    return per * L + V * D


def r_w_for(act_B):
    """Nearest measured shape by active weights, and its name -- never interpolated past 10.6 B."""
    best = min(R_W_BY_SIZE, key=lambda t: abs(t[0] - act_B))
    return best[1], best[0]


def fit(arms_path):
    """Least squares for A and B on the twelve measured points, no intercept."""
    recs = json.load(open(arms_path, encoding="utf-8"))
    rows = []
    for r in recs:
        arm, b = r["label"].rsplit("_b", 1)
        D, F, L, NH, NKV, HD, V = MEASURED_SHAPES[arm]
        o = r["median_rep_organs_ms"]
        pos = (int(b) + 1) / 2.0
        x1 = L * NH * HD * pos
        x2 = L * D
        y = o["rope"] + o["attention"] + o["norm+glue"]
        rows.append((arm, int(b), x1, x2, y))
    # normal equations for y = A*x1 + B*x2
    s11 = sum(x1 * x1 for _, _, x1, _, _ in rows)
    s12 = sum(x1 * x2 for _, _, x1, x2, _ in rows)
    s22 = sum(x2 * x2 for _, _, _, x2, _ in rows)
    s1y = sum(x1 * y for _, _, x1, _, y in rows)
    s2y = sum(x2 * y for _, _, _, x2, y in rows)
    det = s11 * s22 - s12 * s12
    A = (s1y * s22 - s2y * s12) / det
    B = (s11 * s2y - s12 * s1y) / det
    return A, B, rows


def f_of(A, B, D, L, NH, HD, ctx):
    pos = (ctx + 1) / 2.0
    return A * L * NH * HD * pos + B * L * D


def read_configs():
    out = []
    for p in sorted(glob.glob(os.path.join(CONFIGS, "*.json"))):
        base = os.path.basename(p)
        if base.endswith(".generation.json") or base == "_manifest.json":
            continue
        c = json.load(open(p, encoding="utf-8"))
        D = c.get("hidden_size")
        L = c.get("num_hidden_layers")
        NH = c.get("num_attention_heads")
        V = c.get("vocab_size")
        NKV = c.get("num_key_value_heads", NH)
        HD = c.get("head_dim") or (D // NH if D and NH else None)
        F = c.get("intermediate_size")
        name = base[:-5].replace("__", "/")
        # A donor whose layers are not all standard attention (SSM/hybrid) or whose FFN is a
        # mixture cannot be priced by this fit.  Flag rather than guess.
        why = []
        if not all(isinstance(v, int) for v in (D, L, NH, V, NKV, HD)):
            why.append("shape fields missing")
        for k in ("num_experts", "num_local_experts", "n_routed_experts",
                  "mamba_d_state", "linear_attention", "layers_block_type", "hybrid_override_pattern"):
            if k in c:
                why.append("non-uniform (%s)" % k)
                break
        out.append(dict(name=name, D=D, F=F, L=L, NH=NH, NKV=NKV, HD=HD, V=V, skip=why))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=300)
    a = ap.parse_args()

    A, B, rows = fit(ARMS)
    print("f(shape, ctx) = A*L*NH*HD*pos + B*L*D    A = %.6g ms/FMA-unit   B = %.6g ms\n" % (A, B))
    print("| arm | bench | measured f | fitted f | residual | %% |")
    print("|---|---|---|---|---|---|")
    worst = 0.0
    for arm, b, x1, x2, y in rows:
        yh = A * x1 + B * x2
        pct = 100.0 * (yh - y) / y
        worst = max(worst, abs(pct))
        print("| `%s` | %d | %.3f | %.3f | %+.3f | %+.1f%% |" % (arm, b, y, yh, yh - y, pct))
    print("\nworst residual: %.1f%%  -- the fit's licence to be extrapolated\n" % worst)

    print("### Budget at ctx = %d, re-derived per shape (INDEX s7 used one number, 522 M)\n" % a.ctx)
    print("| donor | D | V | L | NH/NKV | head weights | fitted f ms | r_w | **budget** | "
          "**head as %% of budget** |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    skipped = []
    for d in read_configs():
        if d["skip"]:
            skipped.append((d["name"], "; ".join(d["skip"])))
            continue
        f = f_of(A, B, d["D"], d["L"], d["NH"], d["HD"], a.ctx)
        act = active_weights(d["D"], d["F"], d["L"], d["NH"], d["NKV"], d["HD"], d["V"])
        rw, near = r_w_for(act / 1e9)
        budget = rw * 1e9 * max(0.0, 20.0 - f) * 1e-3
        head = d["D"] * d["V"]
        pct = (100.0 * head / budget) if budget > 0 else float("inf")
        print("| %s | %d | %d | %d | %d/%d | %.0f M | %.3f | %.1f | %s | %s |"
              % (d["name"], d["D"], d["V"], d["L"], d["NH"], d["NKV"], head / 1e6, f, rw,
                 ("**%.0f M**" % (budget / 1e6)) if budget > 0 else "**0**",
                 ("**%.0f%%**" % pct) if budget > 0 else "**no budget: f > 20 ms**"))
    if skipped:
        print("\nNot priced by this fit (the fit is for uniform dense attention only):")
        for n, w in skipped:
            print("  - %s -- %s" % (n, w))


if __name__ == "__main__":
    main()
