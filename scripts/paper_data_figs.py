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


# --------------------------------------------- Fig 1: radius distributions
def fig_ecdf():
    curves = [  # label, radii, color, linestyle
        ("smallest supported (exact wind)",
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
    fig_ecdf()
    fig_scatter()
    fig_coverage()
    fig_rates()
    print("paper data figures done")
