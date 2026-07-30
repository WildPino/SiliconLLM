#!/usr/bin/env python3
"""
Generate README benchmark charts from the project's measured findings.

Every number here comes from a real experiment run on the dev box
(Ryzen 5 3600X, Zen 2, AVX2, no AVX-512/VNNI) or a matched GPU training run.
Sources are noted per-chart. Run:  python scripts/make_readme_charts.py
Outputs PNGs into assets/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ----------------------------------------------------------------------------- style
BG      = "#0e1117"   # card background (theme-independent on GitHub)
PANEL   = "#161b22"
INK     = "#e6edf3"   # primary text
MUTED   = "#8b949e"   # secondary text
GRID    = "#30363d"
TEAL    = "#2dd4bf"   # accent 1 (the "new" / winning path)
AMBER   = "#fbbf24"   # accent 2 (baseline / reference)
RED     = "#f87171"   # warning / floor
BLUE    = "#60a5fa"   # secondary series

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 130,
    "font.family": "DejaVu Sans",
})

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)


def _finish(fig, name, sub=None):
    if sub:
        fig.text(0.5, 0.005, sub, ha="center", va="bottom", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03 if sub else 0, 1, 1))
    path = os.path.join(ASSETS, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------- 1. fewer bits = faster
# Source: benchmarks/phase57/phase57_lutbench.c on Ryzen 5 3600X (Zen 2, AVX2).
# pshufb byte-LUT matvec vs fp32; all kernels bit-exact vs scalar integer ref.
def chart_bits_speed():
    # NOTE ON PROVENANCE: ternary/4-bit/1-bit = measured on the 3600X (phase57_lutbench.c,
    # range mid-points). int8 = a LITERATURE figure (~1.2x, VNNI-less dequant): probe-1 did
    # not build an int8-dequant arm, so it is shown for context only, hatched + "(lit.)".
    # (Future: read these from a committed results JSON instead of hardcoding.)
    labels = ["fp32\n(32-bit)", "int8\n(lit.)", "4-bit", "ternary\n(1.58-bit)", "1-bit"]
    speed  = [1.0, 1.2, 1.9, 4.6, 9.0]           # x over fp32 matvec
    colors = [AMBER, RED, BLUE, TEAL, TEAL]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(labels, speed, color=colors, width=0.62, zorder=3)
    bars[1].set_hatch("//")          # int8 = literature, visually distinguished
    bars[1].set_alpha(0.55)
    for b, s in zip(bars, speed):
        ax.text(b.get_x()+b.get_width()/2, s+0.15, f"{s:.1f}×", ha="center",
                va="bottom", color=INK, fontweight="bold")
    ax.axhline(1.0, color=AMBER, ls="--", lw=1, alpha=0.6, zorder=2)
    ax.set_ylabel("speedup vs fp32 matvec  (batch-1, single core)")
    ax.set_title("Fewer bits → faster on Zen 2  (pshufb byte-LUT, no VNNI)")
    ax.set_ylim(0, 10.2)
    ax.annotate("int8 = literature reference\n(no int8 arm was built here)",
                xy=(1, 1.2), xytext=(1.35, 4.4), color=MUTED, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    _finish(fig, "bench_ternary_speed.png",
            "measured (ternary/4-bit/1-bit): phase57_lutbench.c on 3600X, bit-exact · int8 bar = literature")


# ---------------------------------------------------- 2. ternary quality cost (tiny)
# Source: phase57_ternary.py, 5M Arch-A, TinyStories, matched recipe.
def chart_ternary_quality():
    labels = ["fp32 MLP", "ternary MLP\n(1.58-bit)"]
    bpb    = [0.8104, 0.8382]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    bars = ax.bar(labels, bpb, color=[AMBER, TEAL], width=0.5, zorder=3)
    for b, v in zip(bars, bpb):
        ax.text(b.get_x()+b.get_width()/2, v+0.006, f"{v:.4f}", ha="center",
                va="bottom", color=INK, fontweight="bold")
    ax.set_ylabel("validation BPB  (lower is better)")
    ax.set_ylim(0.75, 0.90)
    ax.set_title("Ternary cost at 5M params: +0.028 BPB")
    ax.annotate("", xy=(1, 0.8382), xytext=(1, 0.8104),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(1.08, 0.824, "+0.028\n(upper bound —\nstill descending)", color=RED,
            fontsize=9.5, va="center")
    _finish(fig, "bench_ternary_quality.png",
            "phase57_ternary.py · 5M Arch-A · TinyStories · MLP-only, mixed precision")


# ------------------------------------------------------------ 3. cache-residency cliff
# Source: benchmarks/phase57/phase57_cachesweep.c on Ryzen 5 3600X.
def chart_cache_cliff():
    # working-set (MB) vs sustained bandwidth (GB/s), schematic of the measured steps
    x = np.array([0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 24, 32, 64])
    y = np.array([112, 110, 108, 106, 104, 100, 96, 90, 78, 60, 34, 30, 28.0])
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(x, y, color=TEAL, lw=2.4, marker="o", ms=4, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.032, 0.5, 4, 16, 64])
    ax.set_xticklabels(["32 KB", "512 KB", "4 MB", "16 MB", "64 MB"])
    ax.axvspan(16, 70, color=RED, alpha=0.08, zorder=0)
    ax.axvline(16, color=RED, ls="--", lw=1.4, zorder=2)
    ax.text(16*1.06, 100, "16 MB = L3 per CCX\n→ the active-slice budget", color=RED,
            fontsize=9.5, va="top")
    ax.text(0.05, 118, "compute-bound plateau ~100 GB/s", color=TEAL, fontsize=9.5)
    ax.text(24, 40, "DRAM floor\n~28 GB/s", color=MUTED, fontsize=9.5)
    ax.set_ylabel("sustained bandwidth  (GB/s)")
    ax.set_xlabel("active working-set  (log scale)")
    ax.set_ylim(0, 125)
    ax.set_title("The keystone constraint: stay under the 16 MB L3 cliff")
    _finish(fig, "bench_cache_cliff.png",
            "benchmarks/phase57/phase57_cachesweep.c · Ryzen 5 3600X · Zen 2, single CCX")


# ------------------------------------- 4. predictability + block-structured sparsity
# Source: phase58_predict.py / phase58_reg.py on converged base vs +regularizer.
def chart_predict_sparsity():
    bs   = ["BS 4", "BS 8", "BS 16", "BS 32"]
    xpos = np.arange(len(bs))
    skip_base = [40, 18, 5, 0.7]     # % blocks skippable, dReLU baseline
    skip_reg  = [68, 50, 31, 17]     # % blocks skippable, + coherence regularizer
    recall    = [86.4, 92.3, 95.0, 97.0]  # in-place predictor recall (base), %

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    w = 0.38
    ax1.bar(xpos-w/2, skip_base, w, label="dReLU baseline", color=AMBER, zorder=3)
    ax1.bar(xpos+w/2, skip_reg,  w, label="+ coherence reg.", color=TEAL, zorder=3)
    for i, (a, b) in enumerate(zip(skip_base, skip_reg)):
        ax1.text(i-w/2, a+1, f"{a:g}", ha="center", va="bottom", color=MUTED, fontsize=8.5)
        ax1.text(i+w/2, b+1, f"{b:g}", ha="center", va="bottom", color=INK, fontsize=8.5, fontweight="bold")
    ax1.set_xticks(xpos); ax1.set_xticklabels(bs)
    ax1.set_ylabel("% MLP blocks skippable  (contiguous, ρ-honest)")
    ax1.set_title("Coherence makes sparsity block-structured")
    ax1.set_ylim(0, 78)
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9.5, loc="upper right")

    ax2.plot(xpos, recall, color=TEAL, lw=2.4, marker="o", ms=6, zorder=3)
    for i, r in enumerate(recall):
        ax2.text(i, r+0.4, f"{r:.1f}%", ha="center", va="bottom", color=INK, fontweight="bold", fontsize=9.5)
    ax2.set_xticks(xpos); ax2.set_xticklabels(bs)
    ax2.set_ylabel("active-set recall  (in-place, predictor-free)")
    ax2.set_title("The active set is intrinsically predictable")
    ax2.set_ylim(84, 99)
    _finish(fig, "bench_predict_sparsity.png",
            "phase58_predict.py / phase58_reg.py · converged base · held-out ridge probe")


# ------------------------------------------------- 5. compound bytes/weight reduction
def chart_compound():
    # The bar label names the packing that produces the factor, not the information-theoretic ideal:
    # the 8× comes from the engine's real 4-bit codes. Labelling it "1.58-bit" next to an 8× bar invites
    # the reader to compute 32/1.58 = 20 and fail to reconstruct it -- which is how the 21× happened.
    stages = ["fp32\ndense", "ternary\npacked 4-bit", "+ activation\nsparsity"]
    # Cumulative × fewer MLP bytes/token vs fp32-dense, each step reconstructible from a declared constant:
    #   fp32 4 B/weight ÷ 0.5 B/weight (the engine's REAL 4-bit base-3 g=2 packing) = 8.0×
    #   × 2.12× predictor-free activation sparsity (probe-2)                        = 16.96× → 17×
    # The previous [1.0, 10.1, 21.4] was a mixed-basis error, declared rather than quietly deflated:
    # 10.1 = 2 B/weight (fp16!) ÷ 0.198 B/weight (the IDEAL 1.585-bit packing), so it took its numerator
    # from a baseline this chart does not plot and its denominator from a packing the engine does not use.
    # The two mistakes pulled opposite ways (÷2 and ×2.52) and left a net 1.26× overstatement — which is
    # exactly the unexplained 1.26 the audit isolated. Corrected 2026-07-30.
    factor = [1.0, 8.0, 17.0]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    bars = ax.bar(stages, factor, color=[AMBER, BLUE, TEAL], width=0.55, zorder=3)
    for b, f in zip(bars, factor):
        ax.text(b.get_x()+b.get_width()/2, f+0.4, f"{f:.0f}×" if f > 1 else "1×",
                ha="center", va="bottom", color=INK, fontweight="bold")
    ax.set_ylabel("cumulative × fewer MLP bytes / token")
    ax.set_ylim(0, 19)
    # Second defect in the same chart, same family: the title carried "+0.03 BPB", which README §2 itself
    # retires as "only a naive sum of two deltas measured separately at different step counts". The R1
    # matched-convergence calibration prices the combined ternary+dReLU cost at +0.013 ± 0.005.
    ax.set_title("Composing the bandwidth levers  (+0.013 ± 0.005 BPB total)")
    _finish(fig, "bench_compound.png",
            "probe-1 (ternary) × probe-2 (sparsity) · independent axes · predictor-free")


def chart_architecture():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, lines, edge, tcol=INK, fc="#0e1117"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                     fc=fc, ec=edge, lw=1.4, mutation_aspect=0.5))
        ax.text(x+w/2, y+h-2.3, title, ha="center", va="top", color=tcol,
                fontsize=11.5, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(x+w/2, y+h-5.0-i*2.5, ln, ha="center", va="top", color=MUTED, fontsize=9)

    ax.text(2, 44, "A CPU-native LLM, co-designed for the memory wall",
            color=INK, fontsize=16, fontweight="bold")
    ax.text(2, 40.5, "Small always-on “thinking” core + large sparse “knowing” tier — every byte sized to the L3 cache.",
            color=MUTED, fontsize=10.5)

    # tiers
    ax.add_patch(FancyBboxPatch((2, 12), 45, 25, boxstyle="round,pad=0.3,rounding_size=0.8",
                 fc=PANEL, ec=TEAL, lw=1.6, mutation_aspect=0.5))
    ax.text(4.5, 34.5, "THINKING · always-on, cache-resident", color=TEAL, fontsize=12, fontweight="bold")
    ax.text(4.5, 31.6, "O(1) state · ≤ 16 MB L3 · ~24–32M ternary params/token", color=MUTED, fontsize=9)
    box(4.5, 20.5, 19, 9.0, "SSM backbone", ["selective scan", "no KV cache"], GRID)
    box(25.5, 20.5, 19, 9.0, "gated MLP", ["ternary 1.58-bit", "dReLU sparse"], GRID)
    box(4.5, 12.5, 40, 7.0, "predictability", ["active set forecastable in-place (86–92%) → skip, don’t gather"], GRID, tcol=INK)

    ax.add_patch(FancyBboxPatch((53, 12), 45, 25, boxstyle="round,pad=0.3,rounding_size=0.8",
                 fc=PANEL, ec=AMBER, lw=1.6, mutation_aspect=0.5))
    ax.text(55.5, 34.5, "KNOWING · large, sparse, on-demand", color=AMBER, fontsize=12, fontweight="bold")
    ax.text(55.5, 31.6, "scales total params while active/token stays in budget", color=MUTED, fontsize=9)
    box(55.5, 12.5, 19, 17.0, "recall tier",
        ["IVF-PQ two-stage", "128K context", "~18 µs/query", "InfoNCE repr.", "drift-free"], GRID)
    box(76.5, 12.5, 20, 17.0, "granular MoE",
        ["many small experts", "top-k routed", "ternary experts", "hot pool → L3", "router = predictor"], GRID)

    ax.annotate("", xy=(52.5, 24.5), xytext=(47.5, 24.5),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))

    # chassis
    ax.add_patch(FancyBboxPatch((2, 3), 96, 6.5, boxstyle="round,pad=0.3,rounding_size=0.6",
                 fc="#12161d", ec=GRID, lw=1.3, mutation_aspect=0.5))
    ax.text(4.5, 7.5, "BLOCK-DECODE CHASSIS", color=BLUE, fontsize=11.5, fontweight="bold")
    ax.text(4.5, 4.6, "layer-major · N positions per weight-load → random routing becomes bulk-sequential (bandwidth-bound, not latency-bound)",
            color=MUTED, fontsize=9)
    _finish(fig, "architecture.png")


# ------------------------------------------------- 6. granular MoE capacity tier
# Source: probe-4 (phase59_moe.py), 4k-step matched, equal total params (~22.5M), Zen/3060.
def chart_moe():
    labels = ["A · dense\n(1024 active)", "B · dense-big\n(4096 active)",
              "C · MoE granular\nE32×h128 top-8", "D · MoE coarse\nE8×h512 top-2"]
    bpb    = [0.8799, 0.8674, 0.8589, 0.8637]
    active = ["1024", "4096", "1024", "1024"]
    colors = [AMBER, BLUE, TEAL, "#5eead4"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar(labels, bpb, color=colors, width=0.6, zorder=3)
    for b, v, a in zip(bars, bpb, active):
        ax.text(b.get_x()+b.get_width()/2, v+0.0007, f"{v:.4f}", ha="center",
                va="bottom", color=INK, fontweight="bold", fontsize=10)
        ax.text(b.get_x()+b.get_width()/2, 0.851, f"active {a}", ha="center",
                va="bottom", color=MUTED, fontsize=8.5)
    ax.axhline(bpb[0], color=AMBER, ls="--", lw=1, alpha=0.5, zorder=2)
    ax.set_ylabel("validation BPB  (lower is better)")
    ax.set_ylim(0.850, 0.885)
    ax.set_title("Granular MoE meets the quality gate at matched active cost  (probe-4)")
    ax.annotate("C passes the gate (≤ A) · C<B is single-seed, unregistered",
                xy=(2, 0.859), xytext=(0.02, 0.8832), color=TEAL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
    _finish(fig, "bench_moe.png",
            "probe-4 · phase59_moe.py · 4k-step matched, equal total params · single seed · TinyStories")


# ------------------------------------------------- 7. C engine speedup ladder
# Source: Phase 60 / ENGINE_PLAN.md, single-thread on Ryzen 5 3600X, every step parity-gated.
def chart_engine():
    labels = ["E1\nfp32 core", "E2\n+LUT MLP", "E3\n+skip",
              "E3.5\n+fast-exp scan", "E4\n+MoE tier"]
    toks   = [176.1, 179.5, 192.3, 848.0, 701.7]
    colors = [AMBER, AMBER, AMBER, TEAL, BLUE]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    bars = ax.bar(labels, toks, color=colors, width=0.62, zorder=3)
    for b, t in zip(bars, toks):
        ax.text(b.get_x()+b.get_width()/2, t+12, f"{t:.0f}", ha="center",
                va="bottom", color=INK, fontweight="bold")
    ax.set_ylabel("tokens / second  (single core, 3600X)")
    ax.set_ylim(0, 950)
    ax.set_title("The engine, end-to-end: 176 → 848 tok/s  (every step parity-gated)")
    ax.annotate("kernel gains stay hidden —\nthe exact-exp scan was the bottleneck",
                xy=(1.5, 192), xytext=(0.0, 430), color=MUTED, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("fast-exp scan (25.9×)\nunblocks the engine → 4.8×",
                xy=(2.72, 840), xytext=(1.25, 690), color=TEAL, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1))
    ax.text(4, 600, "MoE model\n(−0.021 BPB)", ha="center", va="top", color=BLUE, fontsize=9)
    _finish(fig, "bench_engine.png",
            "Phase 60 · benchmarks/phase60/ · single-thread Ryzen 5 3600X · +0.00004 BPB total")


if __name__ == "__main__":
    chart_architecture()
    chart_moe()
    chart_engine()
    chart_bits_speed()
    chart_ternary_quality()
    chart_cache_cliff()
    chart_predict_sparsity()
    chart_compound()
    print("done ->", ASSETS)
