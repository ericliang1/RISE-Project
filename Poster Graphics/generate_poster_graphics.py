"""Poster figures for the physics-guided methane localization result.

Four graphics, following `methane_poster_graph_suggestions.md`:

  01_region_radius_dumbbell     before/after 90% region radius, per network
  02_scenarios_within_50m       share of scenarios reaching the 50 m scale
  03_component_ablation_waterfall   the four maps added one at a time (DeepSets)
  04_localization_example       one held-out scenario, baseline vs physics-guided

Styling matches `scripts/paper_data_figs.py` (the figures/ house style) with
poster-scale type: orange = without physics maps (the baseline network),
blue = with physics maps (ours), dashed ink = the 50 m reference.

Every number is read from the frozen measured-wind conformal audits, so the
figures agree with Tables I-II and Figs. 2-3 of the paper by construction.

Usage: python "Poster Graphics/generate_poster_graphics.py"
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------- house style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_D = "#104281"
BLUE_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", BLUE_D]
BLUE_FILL, ORANGE_FILL = "#9ec5f4", "#fbe0d5"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 15,
})

DATA = pathlib.Path("/projectnb/rise-tower/eric1/csr-data")
OUT = pathlib.Path(__file__).resolve().parent
SITE = 500.0          # site is 500 x 500 m
N_CELLS = 64 * 64     # candidate source grid
SEEDS = (1, 2, 3)


def rad(sizes):
    """Region cell count -> equivalent-area radius in metres."""
    return np.sqrt(sizes / N_CELLS / np.pi) * SITE


def audit(tag, seed):
    """Load one measured-wind conformal audit, tolerating the two naming
    conventions used across the runs (`_lam0.0_seed2` vs `_seed1`)."""
    for name in (f"pw_audit_pw_noisy_{tag}_seed{seed}_noisy.npz",
                 f"pw_audit_pw_noisy_{tag}_lam0.0_seed{seed}_noisy.npz"):
        if (DATA / name).exists():
            return np.load(DATA / name)
    raise FileNotFoundError(f"no audit for {tag} seed {seed}")


def stats(tag):
    """(median radius, share below 50 m) per seed, as arrays over SEEDS."""
    med, f50 = [], []
    for s in SEEDS:
        r = rad(audit(tag, s)["sizes"].astype(float))
        med.append(np.median(r))
        f50.append((r < 50).mean() * 100)
    return np.array(med), np.array(f50)


def mid(v):
    """Representative value for a metric reported as a three-seed range: the
    midpoint of the range, as the paper's tables and figures quote it."""
    return float((np.min(v) + np.max(v)) / 2)


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png,svg}}")


def despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


# Networks in the fixed order used everywhere: DeepSets, GNN, Set Transformer.
NETS = [("DeepSets", ""), ("GNN", "_gnn"), ("Set Transformer", "_st")]
BOX = dict(boxstyle="round,pad=0.22", fc=SURF, ec="none", alpha=0.95)


def measure():
    """Read the per-network with/without numbers once, for figures 1 and 2."""
    return {lab: {"without": stats(f"nomaps{sfx}"), "with": stats(f"ensr{sfx}")}
            for lab, sfx in NETS}


# --------------------------------- Fig 1: before/after radius, per network
def fig_dumbbell(m):
    """Priority 1. Vertical dumbbells: x = network, y = 90% region radius.
    Dot = median across seeds, whisker = range over the three seeds."""
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    xs = np.arange(len(NETS))
    for x, (lab, _) in zip(xs, NETS):
        wo, wi = m[lab]["without"][0], m[lab]["with"][0]
        a, b = mid(wo), mid(wi)
        # the arrow does the talking: baseline down to physics-guided
        ax.annotate("", xy=(x, b + 1.4), xytext=(x, a - 1.4),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=22,
                                    color=BASE, lw=5, shrinkA=0, shrinkB=0),
                    zorder=3)
        # the GNN pair nearly coincides, so the two value labels are pushed
        # clear of the dots (above / below) rather than set beside them
        for y, rng, c, dy in ((a, wo, ORANGE, 21), (b, wi, BLUE, -25)):
            ax.plot([x, x], [rng.min(), rng.max()], color=c, lw=9,
                    solid_capstyle="round", zorder=4)
            ax.plot(x, y, "o", color=c, ms=13, mec=SURF, mew=2.2, zorder=5)
            ax.annotate(f"{y:.0f} m", xy=(x, y), xytext=(0, dy),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=16, color=INK, fontweight="bold", bbox=BOX,
                        zorder=6)
        drop = a - b
        txt = f"$-${drop:.0f} m" if drop >= 1 else "no change"
        ax.annotate(txt, xy=(x, (a + b) / 2), xytext=(26, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=15, color=INK2, bbox=BOX, zorder=6)

    ax.axhline(50, color=MUTED, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.annotate("50 m facility-scale target", xy=(len(NETS) - 0.55, 51.0),
                va="bottom", ha="right", fontsize=14, color=MUTED, zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in NETS], fontsize=17)
    ax.set_xlim(-0.5, len(NETS) - 0.22)
    ax.set_ylim(45, 121)
    ax.set_xlabel("model", fontsize=16, labelpad=10)
    ax.set_ylabel("90% guaranteed-region radius (m)\nsmaller is better",
                  fontsize=16, linespacing=1.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    h = [plt.Line2D([], [], color=ORANGE, marker="o", lw=0, ms=13, mec=SURF,
                    mew=2, label="without physics maps"),
         plt.Line2D([], [], color=BLUE, marker="o", lw=0, ms=13, mec=SURF,
                    mew=2, label="with physics maps")]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=15,
              handletextpad=0.4)
    ax.set_title("Physics-guided features reduce localization region size",
                 fontsize=20, fontweight="bold", loc="left", pad=16)
    ax.annotate("coverage held at 0.89–0.92 against the nominal 90%; "
                "bars span three training seeds",
                xy=(0, -0.155), xycoords="axes fraction", fontsize=13,
                color=MUTED)
    save(fig, "01_region_radius_dumbbell")


# ------------------------- Fig 2: share of scenarios inside the 50 m scale
def fig_within50(m):
    """Priority 2. Grouped bars, same network order and colour code."""
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    xs = np.arange(len(NETS))
    w = 0.28
    for off, key, c, lab in ((-w / 2 - 0.015, "without", ORANGE,
                              "without physics maps"),
                             (w / 2 + 0.015, "with", BLUE,
                              "with physics maps")):
        f50 = [m[l][key][1] for l, _ in NETS]
        vals = np.array([mid(v) for v in f50])
        lo = np.array([v.min() for v in f50])
        hi = np.array([v.max() for v in f50])
        ax.bar(xs + off, vals, width=w, color=c, zorder=3, label=lab)
        # three-seed range as a thin ink whisker above the bar, not a slot in it
        ax.vlines(xs + off, lo, hi, color=INK2, lw=1.8, zorder=5)
        ax.hlines(np.concatenate([lo, hi]), np.tile(xs + off, 2) - 0.045,
                  np.tile(xs + off, 2) + 0.045, color=INK2, lw=1.8, zorder=5)
        for x, v, h in zip(xs + off, vals, hi):
            ax.annotate(f"{v:.0f}%", xy=(x, max(v, h) + 1.0), ha="center",
                        va="bottom", fontsize=17, color=INK,
                        fontweight="bold", zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in NETS], fontsize=17)
    ax.set_xlim(-0.55, len(NETS) - 0.45)
    ax.set_ylim(0, 36)
    ax.set_xlabel("model", fontsize=16, labelpad=10)
    ax.set_ylabel("test scenarios within 50 m (%)\nlarger is better",
                  fontsize=16, linespacing=1.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=15)
    ax.set_title("More scenarios reach the 50 m facility scale",
                 fontsize=20, fontweight="bold", loc="left", pad=16)
    ax.annotate("2,000 held-out scenarios per run; whiskers span three "
                "training seeds",
                xy=(0, -0.155), xycoords="axes fraction", fontsize=13,
                color=MUTED)
    save(fig, "02_scenarios_within_50m")


# ------------------------- Fig 3: component ablation, DeepSets, waterfall
LADDER = [  # x label, audit tag, colour
    ("input-only\nnetwork", "nomaps", ORANGE),
    ("+ source\nevidence", "zdet", BLUE),
    ("+ sensitivity", "lam0.0", BLUE_STEPS[0]),
    ("+ wind\nfragility", "ens_lam0.0", BLUE_STEPS[0]),
    ("+ fit quality", "ensr", BLUE_STEPS[0]),
]


def fig_waterfall():
    """Priority 3. Cumulative DeepSets ladder: a full bar at the input-only
    radius, one floating bar per added map, and a full bar at the total."""
    levels, fracs, ranges = [], [], []
    for _, tag, _ in LADDER:
        med, f50 = stats(tag)
        levels.append(mid(med))
        fracs.append(f50)
        ranges.append(med)

    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    w = 0.6
    n = len(LADDER)
    # slot 0 = input-only total, slots 1..4 = decrements, slot n = full method
    ax.bar(0, levels[0], width=w, color=ORANGE, zorder=3)
    for k in range(1, n):
        top, bot, c = levels[k - 1], levels[k], LADDER[k][2]
        ax.bar(k, top - bot, bottom=bot, width=w, color=c, zorder=3)
        ax.annotate(f"$-${top - bot:.0f} m", xy=(k, top + 1.8), ha="center",
                    va="bottom", fontsize=16, color=INK,
                    fontweight="bold" if k == 1 else "normal", zorder=5)
        ax.plot([k - 1 - w / 2, k + w / 2], [bot, bot], color=BASE, lw=1.4,
                ls=(0, (3, 2)), zorder=2)
    ax.bar(n, levels[-1], width=w, color=BLUE_D, zorder=3)
    ax.plot([n - 1 - w / 2, n - w / 2], [levels[-1], levels[-1]], color=BASE,
            lw=1.4, ls=(0, (3, 2)), zorder=2)

    # the two totals carry the headline numbers; the rungs just carry a level
    for k, v, f in ((0, levels[0], fracs[0]), (n, levels[-1], fracs[-1])):
        ax.annotate(f"{v:.0f} m", xy=(k, v + 1.8), ha="center", va="bottom",
                    fontsize=18, color=INK, fontweight="bold", zorder=5)
        lo, hi = f.min(), f.max()
        note = (f"{lo:.0f}% within 50 m" if hi - lo < 1
                else f"{lo:.0f}–{hi:.0f}% within 50 m")
        ax.annotate(note, xy=(k, levels[0] * 0.30), ha="center", va="center",
                    rotation=90, fontsize=14, color=SURF, zorder=5)
    for k in range(1, n):
        ax.annotate(f"{levels[k]:.0f} m", xy=(k + w / 2 + 0.04, levels[k]),
                    ha="left", va="center", fontsize=13.5, color=INK2,
                    bbox=BOX, zorder=5)

    ax.axhline(50, color=MUTED, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.annotate("50 m facility-scale target", xy=(n - 0.45, 51.5), va="bottom",
                ha="right", fontsize=14, color=MUTED, zorder=6)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels([l for l, _, _ in LADDER] + ["full\nmethod"],
                       fontsize=14.5, linespacing=1.35)
    ax.set_xlim(-0.6, n + 0.6)
    ax.set_ylim(0, 126)
    ax.set_xlabel("added physics component", fontsize=16, labelpad=10)
    ax.set_ylabel("90% guaranteed-region radius (m)\nsmaller is better",
                  fontsize=16, linespacing=1.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    ax.set_title("Each physics feature contributes to smaller regions",
                 fontsize=20, fontweight="bold", loc="left", pad=16)
    ax.annotate("DeepSets base, measured wind, three-seed range midpoints — a "
                "representative model, not a cross-model comparison",
                xy=(0, -0.215), xycoords="axes fraction", fontsize=13,
                color=MUTED)
    save(fig, "03_component_ablation_waterfall")


# ------------------------- Fig 4: one held-out scenario, baseline vs ours
def pick_scenario(n_masts=8):
    """Stated rule: among held-out scenarios with exactly `n_masts` masts
    whose regions both cover the true source, take the one jointly closest
    (log scale) to that subgroup's median baseline and median ours radii.
    Typical of the subgroup, not a best case."""
    d = dict(np.load(DATA / "ch4t_test.npz", allow_pickle=True))
    zb, zo = audit("nomaps", 1), audit("ensr", 1)
    rb, ro = (rad(z["sizes"].astype(float)) for z in (zb, zo))
    grp = d["n_sensors"].astype(int) == n_masts
    ok = grp & zb["covered"] & zo["covered"]
    dist = (np.log(rb / np.median(rb[grp])) ** 2
            + np.log(ro / np.median(ro[grp])) ** 2)
    return int(np.argmin(np.where(ok, dist, np.inf))), d, zb, zo, rb, ro


def fig_example(n_masts=8):
    """Priority 4. Three panels on identical axes: the site, the baseline
    region, the physics-guided region.

    The saved audits keep the conformal region masks but not the network's
    per-cell probability map, so the panels shade the 90% guaranteed region
    itself rather than a probability heatmap."""
    i, d, zb, zo, rb, ro = pick_scenario(n_masts)
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % 64 + 0.5) / 64 * SITE, (tc // 64 + 0.5) / 64 * SITE)
    um = d["u_mean"][i] / (np.linalg.norm(d["u_mean"][i]) + 1e-9)
    masks = [np.unpackbits(z["masks"][i])[:N_CELLS].reshape(64, 64)
             for z in (zb, zo)]
    ext = [0, SITE, 0, SITE]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6))
    fig.patch.set_facecolor("white")
    panels = [("A. Sensors and true source", None, None, None),
              ("B. Baseline model", masks[0], ORANGE, ORANGE_FILL),
              ("C. Physics-guided model", masks[1], BLUE, BLUE_FILL)]
    for k, (ax, (title, mask, edge, fill)) in enumerate(zip(axes, panels)):
        ax.set_facecolor("white")
        if mask is not None:
            ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", extent=ext,
                      cmap=LinearSegmentedColormap.from_list("f", [fill, fill]),
                      interpolation="nearest", zorder=2)
            ax.contour(mask.astype(float), levels=[0.5], colors=[edge],
                       linewidths=2.6, extent=ext, zorder=3)
        ax.add_patch(Rectangle((0, 0), SITE, SITE, fc="none", ec=BASE, lw=1.6,
                               zorder=1))
        ax.add_patch(Circle(tx, 50, fc="none", ec=MUTED, lw=1.8,
                            ls=(0, (4, 3)), zorder=5))
        ax.scatter(sx[:, 0], sx[:, 1], s=95, c=INK, marker="o",
                   edgecolors="white", linewidths=1.8, zorder=7)
        ax.plot(*tx, marker="*", color="#f2c230", ms=26, mec=INK, mew=1.4,
                zorder=8)
        # wind blows toward +u; drawn in the same corner of every panel
        ax.add_patch(FancyArrowPatch((60 - um[0] * 42, 462 - um[1] * 42),
                                     (60 + um[0] * 42, 462 + um[1] * 42),
                                     arrowstyle="-|>", mutation_scale=20,
                                     color=INK2, lw=2.4, zorder=6))
        ax.annotate("wind", xy=(60, 418), ha="center", fontsize=14,
                    color=INK2, zorder=6)
        if mask is not None:
            r = (rb if k == 1 else ro)[i]
            ax.annotate(f"90% region\nradius {r:.0f} m", xy=(492, 486),
                        ha="right", va="top", fontsize=15.5, color=edge,
                        fontweight="bold", linespacing=1.35, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="none", alpha=0.92))
        ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=12)
        ax.set_xlim(0, SITE)
        ax.set_ylim(0, SITE)
        ax.set_aspect("equal")
        ax.set_xticks([0, 250, 500])
        ax.set_yticks([0, 250, 500])
        ax.set_xlabel("site coordinate (m)", fontsize=14)
        if k:
            ax.set_yticklabels([])
        despine(ax)

    h = [plt.Line2D([], [], color=INK, marker="o", lw=0, ms=10, mec="white",
                    mew=1.6, label=f"sensor masts ({ns})"),
         plt.Line2D([], [], color="#f2c230", marker="*", lw=0, ms=17, mec=INK,
                    mew=1.2, label="true methane source"),
         plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 3)),
                    label="50 m facility scale"),
         plt.Line2D([], [], color=ORANGE, lw=2.6, label="baseline 90% region"),
         plt.Line2D([], [], color=BLUE, lw=2.6,
                    label="physics-guided 90% region")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False,
               fontsize=14.5, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Example: physics produces a more focused search region",
                 fontsize=21, fontweight="bold", x=0.055, ha="left", y=1.125)
    fig.text(0.055, 1.035, f"one held-out scenario ({ns} masts, "
             f"{d['q'][i]:.0f} kg h$^{{-1}}$), typical of its subgroup; "
             "identical axes and 90% coverage in both panels",
             fontsize=13.5, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.17, top=0.92, bottom=0.10)
    with matplotlib.rc_context({"savefig.facecolor": "white"}):
        save(fig, "04_localization_example")
    return i


if __name__ == "__main__":
    m = measure()
    for lab, _ in NETS:
        (wo, wo50), (wi, wi50) = m[lab]["without"], m[lab]["with"]
        print(f"{lab:16s} {wo.min():.0f}-{wo.max():.0f} m "
              f"({wo50.min():.0f}-{wo50.max():.0f}% <50 m)  ->  "
              f"{wi.min():.0f}-{wi.max():.0f} m "
              f"({wi50.min():.0f}-{wi50.max():.0f}% <50 m)")
    fig_dumbbell(m)
    fig_within50(m)
    fig_waterfall()
    idx = fig_example()
    print(f"poster graphics done (example scenario index {idx})")
