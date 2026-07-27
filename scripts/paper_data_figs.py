"""Data figures for the paper: per-scenario results shown four standard ways.

  ecdf     distribution of region radii (all 2,000 test scenarios per curve)
  scatter  paired per-scenario comparison, baseline vs ours, measured wind
  coverage the guarantee check: empirical coverage vs the nominal band
  rates    performance by leak-rate band (median radius and share below 50 m)

Entity colors match the poster set: blue = ours, orange = baseline network,
aqua = direct inversion; references are dashed ink.  Condition encoding:
solid = measured wind (deployment), dashed = exact wind (laboratory).

Usage: python scripts/paper_data_figs.py  ->  figures/paper/*.{pdf,png}
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import load_config, resolve

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_D = "#104281"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 12,
})
OUT = pathlib.Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)
N_CELLS = 64 * 64


def rad(sizes):
    return np.sqrt(sizes / N_CELLS / np.pi) * 500


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


cfg = load_config()
dd = resolve(cfg, "data_dir")


def sizes(f, key="sizes"):
    return rad(np.load(dd / f)[key].astype(np.float64))


# ------------------------------- Fig 0: representative example (masks)
def pick_scenario(n_masts=8, cond="clean"):
    """Stated rule: among scenarios with exactly n_masts sensors, the one
    jointly closest (log scale) to that subgroup's median baseline and
    median method radii -- typical, not a best case."""
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    if cond == "noisy":
        zb = np.load(dd / "pw_audit_pw_noisy_nomaps_seed1_noisy.npz")
        zo = np.load(dd / "pw_audit_pw_noisy_ensr_seed1_noisy.npz")
    else:
        zb = np.load(dd / "pw_audit_pw_clean_nomaps_seed1_clean.npz")
        zo = np.load(dd / "pw_audit_pw_clean_seed1_clean.npz")
    rb, ro = rad(zb["sizes"].astype(float)), rad(zo["sizes"].astype(float))
    grp = d["n_sensors"].astype(int) == n_masts
    dist = (np.log(rb / np.median(rb[grp])) ** 2
            + np.log(ro / np.median(ro[grp])) ** 2)
    dist[~grp] = np.inf
    i = int(np.argmin(dist))
    return i, d, zb, zo, rb, ro


def fig_example():
    i, d, zb, zo, rb, ro = pick_scenario(8, cond="noisy")
    mb = np.unpackbits(zb["masks"][i])[:N_CELLS].reshape(64, 64)
    mo = np.unpackbits(zo["masks"][i])[:N_CELLS].reshape(64, 64)
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns] * 500
    tc = int(d["true_cell"][i])
    tx = ((tc % 64 + 0.5) / 64 * 500, (tc // 64 + 0.5) / 64 * 500)

    fig, ax = plt.subplots(figsize=(6.6, 6.9))
    ext = [0, 500, 0, 500]
    ax.imshow(np.where(mb, 1.0, np.nan), origin="lower", extent=ext,
              cmap=LinearSegmentedColormap.from_list("ob", ["#fbe0d5",
                                                            "#fbe0d5"]),
              interpolation="nearest", zorder=1)
    ax.imshow(np.where(mo, 1.0, np.nan), origin="lower", extent=ext,
              cmap=LinearSegmentedColormap.from_list("bb", ["#9ec5f4",
                                                            "#9ec5f4"]),
              interpolation="nearest", alpha=0.9, zorder=2)
    ax.contour(mb.astype(float), levels=[0.5], colors=[ORANGE],
               linewidths=2.2, extent=ext, zorder=3)
    ax.contour(mo.astype(float), levels=[0.5], colors=[BLUE],
               linewidths=2.2, extent=ext, zorder=4)
    from matplotlib.patches import Circle
    ax.add_patch(Circle(tx, 50, ec=MUTED, fc="none", lw=1.6,
                        ls=(0, (4, 3)), zorder=5))
    ax.scatter(sx[:, 0], sx[:, 1], s=52, c=INK, edgecolors=SURF,
               linewidths=1.6, zorder=7)
    ax.plot(*tx, marker="+", color=INK, ms=9, mew=2.0, zorder=8)
    h = [plt.Line2D([], [], color=ORANGE, lw=2.2,
                    label=f"no images ({rb[i]:.0f} m)"),
         plt.Line2D([], [], color=BLUE, lw=2.2,
                    label=f"ours ({ro[i]:.0f} m)"),
         plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 3)),
                    label="50 m scale"),
         plt.Line2D([], [], color=INK, marker="+", lw=0, ms=9, mew=2,
                    label="true source"),
         plt.Line2D([], [], color=INK, marker="o", lw=0, ms=7,
                    mec=SURF, label="masts")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, frameon=False, fontsize=10.5,
              handletextpad=0.5, columnspacing=1.1)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect("equal")
    ax.set_xticks([0, 250, 500]); ax.set_yticks([0, 250, 500])
    ax.set_xlabel("site coordinate (m)")
    despine(ax)
    save(fig, "fig_data_example")


# ------------------------------ Fig 0b: full-width pipeline (paper size)
def fig_pipeline():
    """Visual pipeline: the SAME scenario flows left to right -- raw site
    data, the four computed physics images, the network join, and the
    resulting guaranteed region.  Minimal text."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    i, d, zb, zo, rb, ro = pick_scenario(8)
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns]
    tc = int(d["true_cell"][i])
    txn = ((tc % 64 + 0.5) / 64, (tc // 64 + 0.5) / 64)
    um = d["u_mean"][i] / (np.linalg.norm(d["u_mean"][i]) + 1e-9)
    ens = np.load(dd / "ch4tu_test_maps_ens.npz")["maps"][i]
    rm = np.load(dd / "pw_test_resid_marg.npz")["noisy"][i].astype(float)
    zon = np.load(dd / "pw_audit_pw_noisy_ensr_seed1_noisy.npz")
    mo = np.unpackbits(zon["masks"][i])[:N_CELLS].reshape(64, 64)
    CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb",
        "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
    CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5",
        "#f6b899", "#eb6834", "#a03c14"])

    fig = plt.figure(figsize=(12.4, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100)
    ax.set_ylim(0, 24); ax.axis("off")

    def frame(x, w, title, accent=False):
        ax.add_patch(FancyBboxPatch((x, 1.6), w, 20.2,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            fc="#eef4fd" if accent else "#f9f9f7",
            ec=BLUE if accent else BASE, lw=1.4))
        ax.text(x + w / 2, 20.0, title, ha="center", fontsize=10.4,
                color=INK, fontweight="bold")

    def arrow(x0, x1, lab=None):
        ax.add_patch(FancyArrowPatch((x0, 11.5), (x1, 11.5),
                     arrowstyle="-|>", mutation_scale=13, color=INK2,
                     lw=1.6))
        if lab:
            ax.text((x0 + x1) / 2, 14.0, lab, ha="center", fontsize=7.8,
                    color=MUTED)

    def inset(x, y, w, h):
        # convert data coords (100 x 24) to figure fraction
        return fig.add_axes([x / 100, y / 24, w / 100, h / 24])

    # --- box 1: the site ---
    frame(1.0, 14.5, "sensor data")
    a1 = inset(3.2, 4.3, 10.0, 12.6)
    a1.scatter(sx[:, 0], sx[:, 1], s=16, c=INK, edgecolors=SURF,
               linewidths=0.8, zorder=4)
    a1.annotate("", xy=(0.72 + um[0] * 0.22, 0.9 + um[1] * 0.04),
                xytext=(0.72 - um[0] * 0.22, 0.9 - um[1] * 0.04),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_xticks([]); a1.set_yticks([])
    for s in a1.spines.values():
        s.set_color(BASE)
    ax.text(8.25, 2.9, "readings $y$ \u00b7 wind $\\hat w$ \u00b7 "
            "budget $\\varepsilon$", ha="center", fontsize=8.0,
            color=INK2)

    # --- box 2: the four physics images ---
    frame(18.6, 40.0, "physics input processor", accent=True)
    panels = [("evidence", ens[0].reshape(64, 64), CMB),
              ("fragility", ens[1].reshape(64, 64), CMO),
              ("sensitivity", ens[2].reshape(64, 64), CMB),
              ("fit quality", (10 - rm).reshape(64, 64), CMB)]
    for k, (lab, img, cm) in enumerate(panels):
        akk = inset(20.4 + 9.35 * k, 5.6, 8.1, 12.0)
        lo, hi = np.percentile(img, [2, 99])
        akk.imshow(np.clip(img, lo, hi), origin="lower", cmap=cm,
                   interpolation="nearest")
        akk.set_xticks([]); akk.set_yticks([])
        for s in akk.spines.values():
            s.set_color(BASE)
        ax.text(20.4 + 9.35 * k + 4.05, 3.6, lab, ha="center",
                fontsize=8.2, color=INK2)
    ax.text(38.6, 1.9, "closed form, one image set per scenario",
            ha="center", fontsize=7.6, color=MUTED)

    # --- box 3: network join ---
    frame(62.7, 15.5, "network + head")
    for k in range(3):
        ax.add_patch(FancyBboxPatch((65.5 + k * 1.1, 8.2 - k * 1.1),
            6.5, 6.5, boxstyle="round,pad=0.3,rounding_size=0.6",
            fc="#ffffff", ec=BASE, lw=1.1, zorder=3 + k))
    ax.text(69.7, 10.0, "$f_\\theta$", ha="center", fontsize=11,
            color=INK, zorder=8)
    ax.text(70.5, 5.0, "$+\\;h_\\phi$ (0.2%)", ha="center",
            fontsize=8.4, color=INK2)
    ax.text(70.45, 2.6, "label training only", ha="center", fontsize=7.6,
            color=MUTED)

    # --- box 4: guaranteed region ---
    frame(81.4, 17.6, "90% region")
    a4 = inset(84.4, 4.3, 10.4, 12.6)
    a4.imshow(np.where(mo, 1.0, np.nan), origin="lower",
              cmap=LinearSegmentedColormap.from_list("bb", ["#9ec5f4",
                                                            "#9ec5f4"]),
              interpolation="nearest", extent=[0, 1, 0, 1])
    a4.contour(mo.astype(float), levels=[0.5], colors=[BLUE],
               linewidths=1.6, extent=[0, 1, 0, 1])
    a4.scatter(sx[:, 0], sx[:, 1], s=12, c=INK, edgecolors=SURF,
               linewidths=0.7, zorder=4)
    a4.add_patch(Circle(txn, 0.028, fc="none", ec=INK, lw=1.3, zorder=5))
    a4.set_xlim(0, 1); a4.set_ylim(0, 1)
    a4.set_xticks([]); a4.set_yticks([])
    for s in a4.spines.values():
        s.set_color(BASE)
    ax.text(90.2, 2.9, "guaranteed 90% coverage", ha="center",
            fontsize=8.0, color=INK2)

    arrow(15.9, 18.2)
    arrow(59.1, 62.3)
    arrow(78.6, 81.0, )
    save(fig, "fig_pipeline")


# ------------------------- Fig 1b: grouped method comparison (main text)
def fig_compare():
    rows = [  # label, lo, hi, color, note, group
        ("exact-model reference", 6.2, 7.0, INK, "oracle", 0),
        ("baseline network", 94, 98, ORANGE, "0% $<$ 50 m", 0),
        ("ours", 46.2, 47.7, BLUE, "52–54% $<$ 50 m,\n9 runs, 3 architectures", 0),
        ("direct physics inversion", 282, 282, AQUA, "", 1),
        ("baseline network", 107.9, 110.5, ORANGE, "0% $<$ 50 m", 1),
        ("ours (full images)", 70.1, 72.9, BLUE_D, "24–27% $<$ 50 m", 1),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ys = []
    y = 0
    for k, (lab, lo, hi, c, note, grp) in enumerate(rows):
        if k in (0, 3):
            y -= 0.75
            ax.text(4.3, y + 0.62,
                    ["EXACT WIND", "MEASURED WIND (10\u00b0/10% error)"][grp],
                    fontsize=10, color=INK, fontweight="bold", va="center")
        ys.append(y)
        mid = np.sqrt(lo * hi)
        ax.plot([lo, hi], [y, y], color=c, lw=6, solid_capstyle="round",
                zorder=3)
        ax.plot(mid, y, "o", color=c, ms=10, mec=SURF, mew=2, zorder=4)
        txt = f"{lo:.0f}–{hi:.0f} m" if hi > lo + 0.5 else \
            (f"{lo:.0f} m (full site)" if lo > 200 else f"{lo:.1f}–{hi:.1f} m")
        ax.annotate(f"{txt}   {note}".replace("\\n", "\n"),
                    xy=(hi * 1.09, y), va="center", fontsize=9.6,
                    color=INK2, linespacing=1.15)
        ax.annotate(lab, xy=(lo / 1.09, y), va="center", ha="right",
                    fontsize=10.6, color=INK)
        y -= 1
    ax.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate("50 m facility scale", xy=(50, y - 0.05), ha="center",
                fontsize=9.4, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    ax.set_xscale("log")
    ax.set_xlim(1.55, 1000)
    ax.set_ylim(y - 0.45, 0.75)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 400])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "400"])
    ax.set_yticks([])
    ax.set_xlabel("median 90% region radius (m), log scale; "
                  "bars span 3 training seeds")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_data_compare")


# ---------------------- Fig 1c: component build-up with map thumbnails
def fig_components():
    """Measured wind, DeepSets base: input-only network, then each physics
    image added cumulatively; the added map(s) are shown beside each row."""
    i, d, zb, zo, rb, ro = pick_scenario(8)
    det = np.load(dd / "ch4tu_test_maps_det.npz")["maps"][i]
    ens = np.load(dd / "ch4tu_test_maps_ens.npz")["maps"][i]
    rm = np.load(dd / "pw_test_resid_marg.npz")["noisy"][i].astype(float)
    CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb",
        "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
    CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5",
        "#f6b899", "#eb6834", "#a03c14"])
    rows = [  # label, lo, hi, color, frac, thumbs [(img, cmap), ...]
        ("input-only network", 107.9, 110.5, ORANGE, "0%", []),
        ("+ evidence pair", 82.2, 84.2, "#6da7ec", "15–17%",
         [(det[0].reshape(64, 64), CMB), (det[1].reshape(64, 64), CMB)]),
        ("+ wind-fragility image", 74.7, 76.3, "#3987e5", "20–21%",
         [(ens[1].reshape(64, 64), CMO)]),
        ("+ fit-quality image", 70.1, 72.9, BLUE_D, "24–27%",
         [((10 - rm).reshape(64, 64), CMB)]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.96, bottom=0.15)
    n = len(rows)
    for k, (lab, lo, hi, c, frac, thumbs) in enumerate(rows):
        y = n - 1 - k
        mid = (lo + hi) / 2
        ax.plot([lo, hi], [y, y], color=c, lw=6, solid_capstyle="round",
                zorder=3)
        ax.plot(mid, y, "o", color=c, ms=10, mec=SURF, mew=2, zorder=4)
        ax.annotate(f"{lo:.0f}–{hi:.0f} m    <50 m: {frac}",
                    xy=(hi + 2.5, y), va="center", fontsize=10,
                    color=INK2)
        if k > 0:
            pm = (rows[k - 1][1] + rows[k - 1][2]) / 2
            ax.annotate("", xy=(mid, y + 0.18), xytext=(pm, y + 0.82),
                        arrowprops=dict(arrowstyle="-|>", color=BASE,
                                        lw=1.6, shrinkA=2, shrinkB=2))
            ax.annotate(f"$-${pm - mid:.0f} m",
                        xy=((mid + pm) / 2 + 2.5, y + 0.5), fontsize=9.5,
                        color=INK, va="center")
        # thumbnails beside the row label (figure coordinates)
        for j, (img, cm) in enumerate(thumbs):
            axk = fig.add_axes([0.015 + 0.055 * j,
                                0.15 + 0.81 * (y + 0.18) / n, 0.05,
                                0.81 * 0.64 / n])
            lo2, hi2 = np.percentile(img, [2, 99])
            axk.imshow(np.clip(img, lo2, hi2), origin="lower", cmap=cm,
                       interpolation="nearest")
            axk.set_xticks([]); axk.set_yticks([])
            for s in axk.spines.values():
                s.set_color(BASE)
    ax.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate("50 m facility scale", xy=(50, len(rows) - 0.52),
                ha="center", fontsize=9.4, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11)
    ax.set_xlim(40, 135); ax.set_ylim(-0.55, n - 0.35)
    ax.set_xlabel("median 90% region radius (m), measured wind, "
                  "3-seed range")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    save(fig, "fig_data_components")


# --------------------------------------------- Fig 1: radius distributions
def fig_ecdf():
    curves = [  # label, radii, color, linestyle
        ("exact-model reference (exact wind)",
         sizes("ch4t_audit_M0.npz", "exact_sizes"), INK, (0, (4, 3))),
        ("ours, exact wind",
         sizes("ch4t_audit_lever_suffstats_labels.npz"), BLUE_D, (0, (4, 3))),
        ("baseline network, exact wind",
         sizes("ch4t_audit_M0.npz"), ORANGE, (0, (4, 3))),
        ("ours, measured wind",
         sizes("pw_audit_pw_noisy_ensr_lam0.0_seed1_noisy.npz"), BLUE, "-"),
        ("baseline network, measured wind",
         sizes("pw_audit_pw_noisy_nomaps_lam0.0_seed1_noisy.npz"),
         ORANGE, "-"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for lab, r, c, ls in curves:
        x = np.sort(r)
        yv = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, yv, color=c, ls=ls, lw=2.2, label=lab,
                solid_capstyle="round")
    ax.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.annotate("EPA 50 m", xy=(52, 0.03), ha="left", fontsize=10.5,
                color=MUTED,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none",
                          alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlim(3, 400); ax.set_ylim(0, 1.0)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 400])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "400"])
    ax.set_xlabel("90% region radius (m), log scale")
    ax.set_ylabel("fraction of test scenarios at or below")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_title("Region-size distributions over 2,000 test scenarios "
                 "(dashed = exact wind, solid = measured wind)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    save(fig, "fig_data_ecdf")


# ------------------------------------- Fig 2: paired per-scenario scatter
def fig_scatter():
    base = sizes("pw_audit_pw_noisy_nomaps_lam0.0_seed1_noisy.npz")
    ours = sizes("pw_audit_pw_noisy_ensr_lam0.0_seed1_noisy.npz")
    frac = float((ours < base).mean()) * 100
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = (4, 420)
    ax.plot(lim, lim, color=BASE, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.scatter(base, ours, s=13, color=BLUE, alpha=0.28, lw=0, zorder=3)
    ax.plot(np.median(base), np.median(ours), "o", color=BLUE_D, ms=11,
            mec=SURF, mew=2, zorder=5)
    ax.annotate(f"medians:\n{np.median(base):.0f} m → "
                f"{np.median(ours):.0f} m",
                xy=(np.median(base), np.median(ours)),
                xytext=(200, 40), fontsize=11, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2,
                                shrinkB=8))
    ax.text(0.04, 0.96, f"{frac:.0f}% of scenarios improve\n"
            "(points below the diagonal)",
            transform=ax.transAxes, va="top", fontsize=11.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    tk = [5, 10, 20, 50, 100, 200, 400]
    ax.set_xticks(tk); ax.set_xticklabels(map(str, tk))
    ax.set_yticks(tk); ax.set_yticklabels(map(str, tk))
    ax.set_xlabel("baseline network radius (m), measured wind")
    ax.set_ylabel("our radius (m), measured wind")
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Scenario-by-scenario, same test set",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    save(fig, "fig_data_scatter")


# ------------------------------------------- Fig 3: the guarantee check
def fig_coverage():
    res = json.load(open("results/paired_wind.json"))

    def rng(pre, cond):
        vs = [r[cond]["coverage"] for t, r in res.items()
              if t.startswith(pre) and cond in r]
        return min(vs), max(vs)

    rows = [  # label, (lo,hi), color
        ("baseline network, exact wind", (0.890, 0.890), ORANGE),
        ("ours, exact wind", (0.888, 0.895), BLUE_D),
        ("direct inversion, measured wind", (1.0, 1.0), AQUA),
        ("baseline network, measured wind",
         rng("pw_noisy_nomaps_lam0.0", "noisy"), ORANGE),
        ("ours single-wind, measured", rng("pw_noisy_lam0.0", "noisy"),
         "#6da7ec"),
        ("ours + moments, measured", rng("pw_noisy_ens_lam0.0", "noisy"),
         "#3987e5"),
        ("ours full, measured wind", rng("pw_noisy_ensr_lam0.0", "noisy"),
         BLUE_D),
    ]
    n = 2000
    half = 1.96 * np.sqrt(0.9 * 0.1 / n)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0.9 - half, 0.9 + half, color="#eef0ee", zorder=1)
    ax.axvline(0.9, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.annotate("nominal 90%", xy=(0.9, len(rows) - 0.28), ha="center",
                fontsize=10.5, color=INK)
    ax.annotate("shaded: sampling band (n = 2,000)",
                xy=(0.928, 3.5), fontsize=9.8, color=MUTED,
                va="center", ha="left")
    for yi, (lab, (lo, hi), c) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=c, lw=5, solid_capstyle="round",
                zorder=3)
        ax.plot((lo + hi) / 2, yi, "o", color=c, ms=10, mec=SURF, mew=2,
                zorder=4)
        txt = f"{lo:.2f}" if hi - lo < 5e-3 else f"{lo:.2f}–{hi:.2f}"
        ax.annotate(txt, xy=(max(hi, lo) + 0.006, yi), va="center",
                    fontsize=10.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlim(0.865, 1.03); ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlabel("empirical coverage of the 90% region, test set")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("The guarantee holds everywhere; direct inversion holds "
                 "it only at coverage 1.0 (the full site)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    save(fig, "fig_data_coverage")


# --------------------------------------- Fig 4: results by leak-rate band
def fig_rates():
    strata = json.load(open("results/revision_analyses.json")
                       )["recipe_rate_strata"]
    order = ["10-30kgh", "30-100kgh", "100-250kgh", "250-500kgh"]
    labs = ["10–30", "30–100", "100–250", "250–500"]
    med = [strata[k]["median_radius_m"] for k in order]
    f50 = [strata[k]["frac_below_50m"] * 100 for k in order]
    ns = [strata[k]["n"] for k in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    yv = np.arange(len(order))[::-1]
    a1.axvline(50, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    a1.annotate("50 m", xy=(50, 3.62), ha="center", fontsize=10,
                color=MUTED, annotation_clip=False)
    bb = dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.9)
    for yi, m, nsc in zip(yv, med, ns):
        a1.plot([0, m], [yi, yi], color=GRID, lw=1.2, zorder=2)
        a1.plot(m, yi, "o", color=BLUE, ms=11, mec=SURF, mew=2, zorder=4)
        a1.annotate(f"{m:.1f} m", xy=(m + 2.5, yi), va="center",
                    fontsize=10.5, color=INK2, bbox=bb, zorder=6)
    a1.set_yticks(yv); a1.set_yticklabels(labs, fontsize=11.5)
    a1.set_ylabel("leak rate (kg h$^{-1}$)")
    a1.set_xlim(0, 78); a1.set_ylim(-0.55, 3.8)
    a1.set_xlabel("median region radius (m)")
    for yi, f, nsc in zip(yv, f50, ns):
        a2.barh(yi, f, height=0.5, color=BLUE, zorder=3)
        a2.annotate(f"{f:.0f}%   (n={nsc})", xy=(f + 1.5, yi), va="center",
                    fontsize=10.5, color=INK2)
    a2.set_xlim(0, 78); a2.set_xlabel("share of regions below 50 m")
    for ax in (a1, a2):
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        despine(ax)
    fig.suptitle("Performance by leak rate (ours, exact wind): the "
                 "largest emitters are the hardest cases",
                 fontsize=13, fontweight="bold", x=0.125, ha="left",
                 y=1.02)
    save(fig, "fig_data_rates")


if __name__ == "__main__":
    fig_example()
    fig_pipeline()
    fig_ecdf()
    fig_scatter()
    fig_coverage()
    fig_rates()
    print("paper data figures done")
