"""Poster figures: six forms, one visual system.

Entity colors are fixed across every figure (color follows the entity):
  blue  = our method (physics input images, any budget)
  orange = network baseline without physics
  aqua  = direct physics inversion
References (7 m floor, EPA 50 m) are dashed ink lines, never series colors.
Palette: dataviz reference instance (validated slots 1-3, light surface).

Usage: python scripts/poster_figs.py   ->  figures/poster/*.{pdf,png}
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from common import load_config, resolve

# ---- palette (dataviz reference instance, light mode) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L, BLUE_D = "#86b6ef", "#104281"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fcfcfb", "#fbe0d5", "#f6b899", "#f09062", "#eb6834",
              "#c94e1d", "#a03c14", "#75290b"]
CMAP_B = LinearSegmentedColormap.from_list("seqb", SEQ_BLUE)
CMAP_O = LinearSegmentedColormap.from_list("seqo", SEQ_ORANGE)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 12,
})

OUT = pathlib.Path("figures/poster")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def despine(ax, keep=("left", "bottom")):
    for s in ax.spines:
        ax.spines[s].set_visible(s in keep)


# ------------------------------------------------- Fig A: regions to scale
def fig_scale():
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect("equal")
    ax.set_xticks([0, 250, 500]); ax.set_yticks([0, 250, 500])
    ax.set_xlabel("site coordinate (m)")
    despine(ax)
    cx, cy = 250, 250
    rows = [  # radius, color, fill, lw, label, label_dy
        (94, ORANGE, "none", 2.4, "baseline network  94 m", None),
        (71, BLUE_L, "none", 2.4, "ours, measured wind  70–73 m", None),
        (47, BLUE, BLUE, 2.4, "ours, exact wind  46–48 m", None),
    ]
    for r, ec, fc, lw, lab, _ in rows:
        ax.add_patch(Circle((cx, cy), r, ec=ec, lw=lw,
                            fc=(fc if fc == "none" else fc),
                            alpha=(0.18 if fc != "none" else 1.0),
                            zorder=3))
        if fc != "none":
            ax.add_patch(Circle((cx, cy), r, ec=ec, lw=lw, fc="none",
                                zorder=4))
    ax.add_patch(Circle((cx, cy), 50, ec=MUTED, lw=1.6, ls=(0, (4, 3)),
                        fc="none", zorder=5))
    ax.add_patch(Circle((cx, cy), 7, ec=INK, lw=2.0, fc=INK, zorder=6))
    ax.plot(cx, cy, marker="+", color=SURF, ms=6, mew=1.4, zorder=7)
    lab = [(94, ORANGE, "baseline network   94–98 m", 196),
           (71, BLUE_L, "ours, measured wind   70–73 m", 118),
           (47, BLUE, "ours, exact wind   46–48 m", 40),
           (50, MUTED, "EPA facility association   50 m", -40),
           (7, INK, "smallest supported region   6–7 m", -118)]
    for r, c, txt, dy in lab:
        y0 = cy + (r if dy > 0 else -r)
        ax.annotate(txt, xy=(cx, y0), xytext=(cx + 155, cy + dy),
                    ha="left", va="center", fontsize=11.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.4,
                                    shrinkA=2, shrinkB=1,
                                    connectionstyle="arc3,rad=-0.18"))
    ax.set_title("Median 90% search regions, drawn to scale on the "
                 "500 m site", fontsize=13.5, color=INK, pad=12,
                 fontweight="bold", loc="left")
    save(fig, "poster_scale")


# ------------------------------------------------ Fig B: measured-wind ladder
def fig_ladder():
    rows = [  # label, lo, hi, color, frac<50
        ("direct physics inversion", 282, 282, AQUA, "0%"),
        ("network, no physics images", 107.9, 110.5, ORANGE, "0%"),
        ("ours: single-wind images", 82.2, 84.2, "#6da7ec", "15–17%"),
        ("ours: + average & variability", 74.7, 76.3, "#3987e5", "20–21%"),
        ("ours: + fit quality", 70.1, 72.9, BLUE_D, "24–27%"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, lo, hi, c, frac) in zip(y, rows):
        mid = (lo + hi) / 2
        ax.plot([lo, hi], [yi, yi], color=c, lw=5, solid_capstyle="round",
                zorder=3)
        ax.plot(mid, yi, "o", color=c, ms=11, mec=SURF, mew=2, zorder=4)
        txt = f"{lo:.0f}–{hi:.0f} m" if hi > lo else "full site (282 m)"
        ax.annotate(f"{txt}    <50 m: {frac}", xy=(min(hi + 7, 292), yi),
                    va="center", ha="left", fontsize=11, color=INK2)
    ax.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate("EPA 50 m", xy=(56, len(rows) - 0.38), ha="left",
                fontsize=10.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, 405); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("median 90% region radius (m), 3-seed range")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Measured wind (10°/10% error): each added physics "
                 "image tightens the guaranteed region",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)
    save(fig, "poster_ladder")


# ------------------------------------------------ Fig C: dumbbell, exact wind
def fig_dumbbell():
    rows = [  # estimator, (lo,hi) without, (lo,hi) with
        ("DeepSets", (94, 98), (47.1, 47.7)),
        ("GNN", (55, 58), (46.2, 47.5)),
        ("Set Transformer", (69, 75), (46.2, 47.3)),
        ("flow-based estimator", (62.8, 71.2), (57.6, 61.2)),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(46.2, 47.7, color="#cde2fb", alpha=0.55, zorder=1)
    ax.annotate("1.5 m band, all 9 runs", xy=(46.9, len(rows) - 0.05),
                ha="center", va="bottom", fontsize=10.5, color="#1c5cab")
    for yi, (lab, wo, wi) in zip(y, rows):
        mo, mi = np.mean(wo), np.mean(wi)
        ax.annotate("", xy=(mi + 1.2, yi), xytext=(mo - 1.2, yi),
                    arrowprops=dict(arrowstyle="-|>", color=BASE, lw=1.8,
                                    shrinkA=8, shrinkB=8))
        ax.plot([wo[0], wo[1]], [yi, yi], color=ORANGE, lw=5,
                solid_capstyle="round", zorder=3)
        ax.plot(mo, yi, "o", color=ORANGE, ms=10, mec=SURF, mew=2, zorder=4)
        ax.plot([wi[0], wi[1]], [yi, yi], color=BLUE, lw=5,
                solid_capstyle="round", zorder=3)
        ax.plot(mi, yi, "o", color=BLUE, ms=10, mec=SURF, mew=2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(35, 105); ax.set_ylim(-0.6, len(rows) + 0.35)
    ax.set_xlabel("median 90% region radius (m), exact wind, 3-seed range")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    h = [plt.Line2D([], [], color=ORANGE, marker="o", lw=5, ms=9,
                    mec=SURF, mew=2, label="without physics images"),
         plt.Line2D([], [], color=BLUE, marker="o", lw=5, ms=9,
                    mec=SURF, mew=2, label="with physics images")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=11)
    ax.set_title("The physics images move every estimator into the same "
                 "narrow band", fontsize=13.5, fontweight="bold",
                 loc="left", pad=12)
    save(fig, "poster_dumbbell")


# ----------------------------------------- Fig D: the four physics images
def fig_maps():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    ens = np.load(dd / "ch4tu_test_maps_ens.npz")["maps"]
    rmg = np.load(dd / "pw_test_resid_marg.npz")["noisy"].astype(np.float32)
    q = d["q"]; tc = d["true_cell"]; ns = d["n_sensors"]
    xs = d["xs"]
    ok = np.where((q > 60) & (q < 200) & (ns >= 8)
                  & (xs[:, 0] > 0.25) & (xs[:, 0] < 0.75)
                  & (xs[:, 1] > 0.25) & (xs[:, 1] < 0.75))[0]
    i = int(ok[2]) if len(ok) > 2 else int(ok[0])
    sx = d["sensors"][i, :int(ns[i])] * 500
    tx = ((tc[i] % 64 + 0.5) / 64 * 500, (tc[i] // 64 + 0.5) / 64 * 500)
    panels = [
        ("average evidence  $\\bar z_c$", ens[i, 0].reshape(64, 64),
         CMAP_B, "dark = readings point here"),
        ("evidence variability  $v_c$", ens[i, 1].reshape(64, 64),
         CMAP_O, "dark = fragile to wind error"),
        ("sensitivity  $\\overline{\\log b_c}$", ens[i, 2].reshape(64, 64),
         CMAP_B, "dark = visible to this layout"),
        ("fit quality  (from $\\bar R_c$)", (10 - rmg[i].reshape(64, 64)),
         CMAP_B, "dark = readings well explained"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.55))
    for ax, (title, img, cm, sub) in zip(axes, panels):
        lo, hi = np.percentile(img, [1, 99.7])
        ax.imshow(np.clip(img, lo, hi), origin="lower", cmap=cm,
                  extent=[0, 500, 0, 500], interpolation="nearest")
        ax.scatter(sx[:, 0], sx[:, 1], s=34, c=INK, edgecolors=SURF,
                   linewidths=1.4, zorder=5, label="sensor masts")
        ax.add_patch(Circle(tx, 16, ec=INK, fc="none", lw=2.0, zorder=6))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASE)
        ax.set_title(title, fontsize=12.5, color=INK, pad=8)
        ax.text(0.5, -0.075, sub, transform=ax.transAxes, ha="center",
                fontsize=10.5, color=INK2)
    axes[0].annotate("true source", xy=tx, xytext=(370, 55),
                     fontsize=10.5, color=INK, ha="center",
                     bbox=dict(boxstyle="round,pad=0.25", fc=SURF,
                               ec="none", alpha=0.85),
                     arrowprops=dict(arrowstyle="-", color=INK, lw=1.2,
                                     shrinkB=14))
    fig.suptitle("What the input processor computes for one scenario "
                 "(closed form, milliseconds); ○ marks the true source, "
                 "dots the masts", fontsize=13.5, fontweight="bold",
                 x=0.5, y=1.04)
    save(fig, "poster_maps")


# --------------------------------------------------- Fig E: pipeline diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.6, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")

    def box(x, w, title, lines, accent=None, y=3, h=19):
        fc = "#f9f9f7" if accent is None else "#eef4fd"
        ec = BASE if accent is None else BLUE
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.6,rounding_size=1.2",
                     fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", fontsize=11.6,
                color=INK, fontweight="bold")
        for j, t in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.2 - 3.3 * j, t, ha="center",
                    fontsize=10.1, color=INK2)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 12.5), (x1, 12.5),
                     arrowstyle="-|>", mutation_scale=16, color=INK2,
                     lw=1.8))

    box(1, 15.5, "sensor data",
        ["readings $y$", "measured wind $\\hat w$",
         "error budget $\\varepsilon$"])
    arrow(17.3, 20.8)
    box(21.2, 24, "physics input processor",
        ["4 images per cell (Eqs. 6–9):",
         "evidence · variability ·",
         "sensitivity · fit quality",
         "closed form, milliseconds"], accent=True)
    arrow(46, 49.5)
    box(49.9, 20, "network + tiny head",
        ["any set network $f_\\theta$",
         "zero-init conv head $h_\\phi$",
         "0.2% of parameters,",
         "one forward pass"])
    arrow(70.7, 74.2)
    box(74.6, 24.4, "guaranteed region",
        ["split conformal, Eq. 14", "90% coverage, calibrated",
         "under deployment wind"])
    ax.text(33.2, 0.4, "at $\\varepsilon = 0$: reduces exactly to the "
            "classical matched filter (Eq. 10)", ha="center",
            fontsize=10, color="#1c5cab")
    ax.set_title("One budget-aware pipeline: physics supplies the "
                 "evidence, learning supplies calibration and "
                 "generalization", fontsize=13.5, fontweight="bold",
                 loc="left", pad=10)
    save(fig, "poster_pipeline")


# ------------------------------------------------------- Fig F: audit bars
def fig_audit():
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    rows = [("baseline network", 96, ORANGE, "94–98 m"),
            ("ours (exact wind)", 47.4, BLUE, "47.1–47.7 m")]
    y = np.arange(len(rows))[::-1]
    bb = dict(boxstyle="round,pad=0.22", fc=SURF, ec="none", alpha=0.92)
    for yi, (lab, v, c, txt) in zip(y, rows):
        ax.barh(yi, v, height=0.52, color=c, zorder=3)
        ax.annotate(txt, xy=(v + 2.5, yi), va="center", fontsize=11.5,
                    color=INK2, bbox=bb, zorder=6)
    ax.axvline(6.6, color=INK, lw=1.8, ls=(0, (4, 3)), zorder=4)
    ax.annotate("smallest supported region  6.2–7.0 m", xy=(9, 1.62),
                fontsize=10.5, color=INK, ha="left", bbox=bb, zorder=6)
    ax.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate("EPA 50 m", xy=(52, -0.55), ha="left", fontsize=10.5,
                color=MUTED, bbox=bb, zorder=6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, 112); ax.set_ylim(-0.75, 1.85)
    ax.set_xlabel("median 90% region radius (m), exact wind")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("The audit: both models hold the 90% guarantee; only the "
                 "measuring stick shows what it is worth",
                 fontsize=13.5, fontweight="bold", loc="left", pad=12)
    save(fig, "poster_audit")


if __name__ == "__main__":
    fig_scale()
    fig_ladder()
    fig_dumbbell()
    fig_maps()
    fig_pipeline()
    fig_audit()
    print("poster figures done")
