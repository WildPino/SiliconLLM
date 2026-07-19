#!/usr/bin/env python3
# Inventor / S1e - the Architect-report chart. Style = scripts/make_readme_charts.py (project convention:
# every published result ships with a chart, measured numbers only). Sources: s1_spectral_probe / s1b
# (post-hoc truncation, real val BPB), s1c runs seeds 0-2 (from-scratch A/B, sealed brief INVENTORE_02).
# Output: docs/in_research/assets/s1_xproj_chart.png
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, INK, MUTED, GRID = "#0e1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
TEAL, AMBER, RED, BLUE = "#2dd4bf", "#fbbf24", "#f87171", "#60a5fa"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 130,
    "font.family": "DejaVu Sans",
})
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "in_research", "assets")
os.makedirs(OUT, exist_ok=True)

# measured data
posthoc_bytes = [70.3, 52.7, 35.2, 17.6, 10.8]           # % of dense x_proj bytes (r=104,78,52,26,16)
posthoc_dbpb  = [0.0001, 0.0008, 0.0078, 0.0534, 0.1419]
seeds = [0, 1, 2]
ctl  = [0.8757, 0.8811, 0.8784]
r52  = [0.8792, 0.8721, 0.8729]
r26  = [0.8683, 0.8688, 0.8696]
d52  = [a - b for a, b in zip(r52, ctl)]
d26  = [a - b for a, b in zip(r26, ctl)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# -- left: post-hoc curve vs from-scratch points --
ax1.axhline(0, color=MUTED, lw=1, ls="--", alpha=0.8)
ax1.plot(posthoc_bytes, posthoc_dbpb, "o-", color=AMBER, lw=2, ms=6, label="post-hoc SVD truncation")
ax1.annotate("r=16: +0.142", xy=(10.8, 0.055), color=AMBER, fontsize=10,
             ha="left", va="top")
ax1.errorbar([35.2], [np.mean(d52)], yerr=[[np.std(d52, ddof=1)]], fmt="s", color=BLUE, ms=9,
             capsize=5, lw=2, label="from-scratch r=52 (3 seeds)")
ax1.errorbar([17.6], [np.mean(d26)], yerr=[[np.std(d26, ddof=1)]], fmt="D", color=TEAL, ms=10,
             capsize=5, lw=2, label="from-scratch r=26 (3 seeds)")
ax1.annotate("BETTER than dense\nat 17.6% of the bytes", xy=(17.6, np.mean(d26)),
             xytext=(30, -0.017), color=TEAL, fontsize=10,
             arrowprops=dict(arrowstyle="->", color=TEAL, alpha=0.8))
ax1.set_xlabel("x_proj bytes (% of dense)")
ax1.set_ylabel("BPB delta vs dense control")
ax1.set_title("x_proj low-rank: post-hoc collapses, from-scratch WINS")
ax1.set_xlim(105, 5); ax1.set_ylim(-0.022, 0.06)
ax1.legend(loc="upper left", framealpha=0.15, fontsize=10)

# -- right: 3-seed A/B --
xs = {"dense ctl": (0, AMBER, ctl), "r=52": (1, BLUE, r52), "r=26": (2, TEAL, r26)}
for name, (i, c, vals) in xs.items():
    ax2.scatter([i] * 3, vals, color=c, s=70, zorder=3, alpha=0.9)
    m = np.mean(vals)
    ax2.hlines(m, i - 0.22, i + 0.22, color=c, lw=3)
    ax2.annotate(f"{m:.4f}", xy=(i + 0.26, m), color=c, fontsize=11, va="center")
ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["dense ctl\n(208x512)", "low-rank r=52\n(35% bytes)", "low-rank r=26\n(17.6% bytes)"])
ax2.set_ylabel("val BPB (200K tokens)")
ax2.set_title("3-seed A/B: monotone - fewer ranks, better BPB")
ax2.set_xlim(-0.5, 2.75)
ax2.annotate("C1 ADOPTED: paired delta < 0 in 3/3 seeds,\nmean -0.0095 (rule: <= -0.005)",
             xy=(0.02, 0.03), xycoords="axes fraction", color=TEAL, fontsize=10)

fig.text(0.5, 0.005,
         "8.3M Arch-A sandbox, TinyStories, phase57 recipe, sealed prereg INVENTORE_02 (gates frozen before numbers) - "
         "runs 2026-07-17..19, RTX 3060",
         ha="center", va="bottom", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
p = os.path.join(OUT, "s1_xproj_chart.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.25)
print("wrote", p)
