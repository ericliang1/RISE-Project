"""figR_results.pdf: the three final tables as one dot-range figure.

Panel A = Table 1 (recipe x encoders), B = Table 2 (vs flow-NPE),
C = Table 3 (measured-wind robustness).  Horizontal dot+range marks, one
linear radius axis per panel starting at 0; reference lines Bayes 7 m,
teacher 11.7 m, EPA 50 m.  Identity colors fixed across panels:
ours/recipe = magenta, flow = blue, no-physics stages = gray, references =
green/ink.  Direct value labels; recessive grid; ranges = 3-seed min-max.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

C_GRAY = "#8a8987"     # labels-only / no-physics stage
C_BLUE = "#2a78d6"     # flow-NPE family
C_MAG = "#e87ba4"      # ours / recipe family
C_MAGD = "#c04d7c"     # ours, emphasized (full/ens)
C_EXACT = "#008300"
C_INK = "#0b0b0b"
C_MUT = "#52514e"
C_GRID = "#e8e7e4"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})


def refs(ax, xmax, epa_label=True):
    for x, c, ls, lab in [(7.0, C_EXACT, "-", "Bayes limit 7 m"),
                          (11.7, C_EXACT, "--", "teacher ceiling 11.7 m"),
                          (50.0, C_INK, ":", "EPA 50 m")]:
        if x < xmax:
            ax.axvline(x, color=c, ls=ls, lw=1.0, zorder=1, alpha=0.85)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def row(ax, y, lo, hi, color, label_val=None, marker="o"):
    mid = (lo + hi) / 2
    if hi > lo:
        ax.plot([lo, hi], [y, y], color=color, lw=2.5,
                solid_capstyle="round", zorder=3)
    ax.plot([mid], [y], marker, color=color, ms=6.5,
            mec="white", mew=0.8, zorder=4)
    if label_val:
        txt = label_val
    elif hi <= lo:
        txt = f"{mid:.0f}"
    elif hi - lo < 2:
        txt = f"{lo:.1f}–{hi:.1f}"
    else:
        txt = f"{lo:.0f}–{hi:.0f}"
    ax.annotate(txt, (hi, y), textcoords="offset points", xytext=(6, -0.5),
                va="center", fontsize=7.2, color=C_MUT)


def main():
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4),
                             gridspec_kw=dict(height_ratios=[3.2, 1.3, 1.3],
                                              hspace=0.52))

    # ---------------- Panel A: Table 1 ---------------------------------
    ax = axes[0]
    encs = ["DeepSets", "GNN", "Set Transformer"]
    data = {  # (labels lo-hi), (distill), (recipe)
        "DeepSets": [(94, 98), (73, 74), (48.1, 49.3)],
        "GNN": [(55, 58), (49, 52), (43.6, 44.3)],
        "Set Transformer": [(69, 75), (56, 60), (44.7, 48.9)],
    }
    stage_c = [C_GRAY, C_BLUE, C_MAGD]
    ylabels, y = [], 0
    for enc in encs:
        for s, (lo, hi) in enumerate(data[enc]):
            row(ax, y, lo, hi, stage_c[s])
            ylabels.append(enc if s == 1 else "")
            y += 1
        y += 0.7
    ax.set_yticks(np.arange(len(ylabels))
                  + np.repeat([0, 3.7, 7.4], 3)[:len(ylabels)] * 0
                  if False else
                  [i + 0.7 * (i // 3) for i in range(9)])
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    refs(ax, 110)
    ax.set_xlim(0, 112)
    ax.set_title("A    One recipe, three architectures "
                 "(median 90% region radius, 3 seeds)", loc="left")
    leg = [Line2D([], [], color=c, lw=2.5, label=l) for c, l in
           [(C_GRAY, "labels only"), (C_BLUE, "+ posterior distillation"),
            (C_MAGD, "+ physics maps (full recipe)")]]
    leg += [Line2D([], [], color=C_EXACT, lw=1, label="Bayes 7 / teacher 11.7 m"),
            Line2D([], [], color=C_INK, lw=1, ls=":", label="EPA 50 m")]
    ax.legend(handles=leg, frameon=False, fontsize=6.8,
              loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=3,
              handlelength=1.5, columnspacing=1.0, borderaxespad=0)

    # ---------------- Panel B: Table 2 ---------------------------------
    ax = axes[1]
    rows_b = [("Flow-NPE (no physics)", (62.8, 71.2), C_BLUE),
              ("Flow-NPE + maps + distillation", (55.5, 56.4), C_BLUE),
              ("Ours (grid head, full recipe)", (48.1, 49.3), C_MAGD)]
    for i, (lab, (lo, hi), c) in enumerate(rows_b):
        row(ax, i, lo, hi, c)
    ax.set_yticks(range(len(rows_b)))
    ax.set_yticklabels([r[0] for r in rows_b])
    ax.invert_yaxis()
    refs(ax, 110)
    ax.set_xlim(0, 112)
    ax.set_title("B    vs. the modern head: flow-based NPE, same encoder "
                 "(3 seeds)", loc="left")

    # ---------------- Panel C: Table 3 ---------------------------------
    ax = axes[2]
    rows_c = [("Exact-physics oracle", 232, C_MUT, " "),
              ("Ours, deterministic maps", 94.7, C_MAG, "95"),
              ("Ours, ensemble maps", 82.3, C_MAGD, "82")]
    for i, (lab, v, c, txt) in enumerate(rows_c):
        row(ax, i, v, v, c, label_val=txt)
    ax.annotate("232 — coverage 0.70 ✗", (232, 0), textcoords="offset points",
                xytext=(-8, -0.5), ha="right", va="center", fontsize=7.2,
                color=C_MUT)
    ax.set_yticks(range(len(rows_c)))
    ax.set_yticklabels([r[0] for r in rows_c])
    ax.invert_yaxis()
    refs(ax, 250)
    ax.set_xlim(0, 262)
    ax.set_title("C    Measured wind (10°/10% error): physics breaks, "
                 "the student doesn't (seed 1)", loc="left")
    ax.set_xlabel("median 90% conformal region radius (m)")

    fig.savefig("figures/figR_results.pdf")
    fig.savefig("figures/figR_results_prev.png", dpi=160)
    print("figR done")


if __name__ == "__main__":
    main()
