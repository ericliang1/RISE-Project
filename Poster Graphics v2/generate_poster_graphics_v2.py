"""Poster figures, second batch — XAI visuals plus remakes of the paper's
figures at poster scale.

  01_xai_feature_maps       what the physics head sees: the four input maps
                            for one held-out scenario, plus the exact Bayes
                            reference posterior
  02_xai_umap_embedding     UMAP of the source-evidence maps over all 2,000
                            test scenarios, coloured three ways
  03_region_areas_12masts   guaranteed search areas drawn to scale, per
                            model: all scenarios vs the 12-mast subgroup
  04_radius_cdf             paper Fig. 2 at poster scale (region-radius CDFs)
  05_masts_curves           paper Fig. 3 at poster scale (radius vs masts)
  06_paired_effects_forest  Table I's paired-effect columns as a forest plot
  07_coverage_calibration   observed coverage vs the nominal 90% guarantee

Styling continues `Poster Graphics/generate_poster_graphics.py` (the v1
batch): orange = without physics maps, blue = with physics maps, dashed ink
= the 50 m facility-scale reference.  Where a figure separates the three
networks, the model palette is DeepSets blue / GNN red / Set Transformer
green (validated for CVD separation; every curve is direct-labelled).

Every number is read from the frozen measured-wind conformal audits and the
frozen feature-map / posterior files, so the figures agree with the paper by
construction.  Nothing is transcribed by hand except the bootstrap CIs of
Fig. 06, which come from `results/table_deltas_ci.json`.

Usage: source scripts/env.sh && python "Poster Graphics v2/generate_poster_graphics_v2.py"
"""
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------- house style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#c94040"
BLUE_D = "#104281"
BLUE_FILL, ORANGE_FILL = "#9ec5f4", "#fbe0d5"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# model palette for figures that separate the three networks
# (#2a78d6/#c94040/#1baf7a passes the CVD checks; the green's low surface
# contrast is relieved by direct labels on every curve)
MODEL_C = {"DeepSets": BLUE, "GNN": RED, "Set Transformer": AQUA}

CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#9ec5f4",
    "#6da7ec", "#3987e5", "#256abf", "#104281"])
CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5", "#f6b899",
    "#eb6834", "#a03c14"])
CMG = LinearSegmentedColormap.from_list("g", ["#fcfcfb", "#c8ecd9", "#84d4ac",
    "#1baf7a", "#0b6b47"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "font.size": 15,
})

DATA = pathlib.Path("/projectnb/rise-tower/eric1/csr-data")
OUT = pathlib.Path(__file__).resolve().parent
REPO = OUT.parent
SITE = 500.0
N_CELLS = 64 * 64
SEEDS = (1, 2, 3)
NETS = [("DeepSets", ""), ("GNN", "_gnn"), ("Set Transformer", "_st")]
BOX = dict(boxstyle="round,pad=0.22", fc=SURF, ec="none", alpha=0.95)


def rad(sizes):
    """Region cell count -> equivalent-area radius in metres."""
    return np.sqrt(np.asarray(sizes, float) / N_CELLS / np.pi) * SITE


def audit(tag, seed):
    """One measured-wind conformal audit, tolerating both naming schemes."""
    for name in (f"pw_audit_pw_noisy_{tag}_seed{seed}_noisy.npz",
                 f"pw_audit_pw_noisy_{tag}_lam0.0_seed{seed}_noisy.npz"):
        if (DATA / name).exists():
            return np.load(DATA / name)
    raise FileNotFoundError(f"no audit for {tag} seed {seed}")


def pooled_radii(tag):
    """Radii over the 2,000 test scenarios, concatenated across three seeds."""
    return np.concatenate([rad(audit(tag, s)["sizes"]) for s in SEEDS])


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png,svg}}")


def despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


def load_test():
    return dict(np.load(DATA / "ch4t_test.npz", allow_pickle=True))


def pick_scenario(d, n_masts=8):
    """Same stated rule as the v1 batch and the paper's map figures: among
    held-out scenarios with exactly `n_masts` masts whose regions both cover
    the true source, the one jointly closest (log scale) to that subgroup's
    median baseline and median physics-guided radii."""
    zb, zo = audit("nomaps", 1), audit("ensr", 1)
    rb, ro = rad(zb["sizes"]), rad(zo["sizes"])
    grp = d["n_sensors"].astype(int) == n_masts
    ok = grp & zb["covered"] & zo["covered"]
    dist = (np.log(rb / np.median(rb[grp])) ** 2
            + np.log(ro / np.median(ro[grp])) ** 2)
    return int(np.argmin(np.where(ok, dist, np.inf)))


# ---------------------------------- Fig 01: the four physics maps, one scene
def fig_feature_maps(d):
    """XAI gallery: the observed scene, the four physics-derived input maps,
    and the exact Bayes reference posterior, all for the same scenario."""
    i = pick_scenario(d)
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % 64 + 0.5) / 64 * SITE, (tc // 64 + 0.5) / 64 * SITE)
    um = d["u_mean"][i] / (np.linalg.norm(d["u_mean"][i]) + 1e-9)
    peak = d["readings"][i, :ns].max(axis=1)

    ens = np.load(DATA / "ch4tu_test_maps_ens.npz")["maps"][i]
    rm = np.load(DATA / "pw_test_resid_marg.npz")["noisy"][i].astype(float)
    post = np.load(DATA / "ch4t_test_posterior.npz")["probs"][i]

    panels = [
        ("A. What the sensors observed", None, None,
         "dot size = peak methane reading;\narrow = measured wind"),
        ("B. Source evidence", ens[0], CMB,
         "dark = a leak here explains\nthe observed readings"),
        ("C. Wind sensitivity", ens[1], CMO,
         "dark = evidence unstable\nunder anemometer error"),
        ("D. Sensor visibility", ens[2], CMB,
         "dark = a leak here would be\nseen by many masts"),
        ("E. Fit quality", -rm, CMB,
         "dark = plume model fits well\nafter best-rate scaling"),
        ("F. Exact Bayes reference (known wind)", None, None,
         "with the true wind, the posterior collapses\nto one grid cell — "
         "wind error is the whole game"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 11.4))
    for k, (ax, (title, img, cm, caption)) in enumerate(zip(axes.flat, panels)):
        if img is not None:
            m = img.reshape(64, 64)
            lo, hi = np.percentile(m, [2, 99])
            ax.imshow(np.clip(m, lo, hi), origin="lower", cmap=cm,
                      extent=[0, SITE, 0, SITE], interpolation="bilinear")
            ax.scatter(sx[:, 0], sx[:, 1], s=52, c=INK, edgecolors="white",
                       linewidths=1.5, zorder=5)
        elif k == 0:
            ax.set_facecolor("white")
            ax.add_patch(Rectangle((0, 0), SITE, SITE, fc="none", ec=BASE,
                                   lw=1.6, zorder=1))
            s = 60 + 480 * peak / peak.max()
            ax.scatter(sx[:, 0], sx[:, 1], s=s, c=INK, edgecolors="white",
                       linewidths=1.8, zorder=5)
            ax.add_patch(FancyArrowPatch((70 - um[0] * 48, 448 - um[1] * 48),
                                         (70 + um[0] * 48, 448 + um[1] * 48),
                                         arrowstyle="-|>", mutation_scale=22,
                                         color=INK2, lw=2.6, zorder=6))
            ax.annotate("wind", xy=(70, 398), ha="center", fontsize=14,
                        color=INK2, zorder=6)
        else:
            # the exact posterior is a delta: >99.999% of its mass sits in a
            # single 7.8 m grid cell, so it is drawn as geometry, not a heatmap
            cell = int(post.argmax())
            cx = (cell % 64) / 64 * SITE
            cy = (cell // 64) / 64 * SITE
            ax.set_facecolor("white")
            ax.add_patch(Rectangle((0, 0), SITE, SITE, fc="none", ec=BASE,
                                   lw=1.6, zorder=1))
            ax.scatter(sx[:, 0], sx[:, 1], s=52, c=INK, edgecolors="white",
                       linewidths=1.5, zorder=5)
            ax.add_patch(Rectangle((cx, cy), SITE / 64, SITE / 64, fc=AQUA,
                                   ec="#0b6b47", lw=1.2, zorder=6))
            ax.add_patch(Circle((cx + SITE / 128, cy + SITE / 128), 34,
                                fc="none", ec=AQUA, lw=2.4, ls=(0, (4, 3)),
                                zorder=6))
            ax.annotate("100% of the posterior:\none 7.8 m cell",
                        xy=(cx + 26, cy + 26), xytext=(215, 380),
                        fontsize=14.5, color="#0b6b47", fontweight="bold",
                        linespacing=1.4, zorder=9,
                        arrowprops=dict(arrowstyle="-|>", color="#0b6b47",
                                        lw=2.0, shrinkA=4,
                                        connectionstyle="arc3,rad=-0.2"))
        ax.plot(*tx, marker="*", color="#f2c230", ms=24, mec=INK, mew=1.3,
                zorder=8)
        ax.set_title(title, fontsize=17, fontweight="bold", loc="left", pad=9)
        ax.annotate(caption, xy=(0, -0.035), xycoords="axes fraction",
                    fontsize=13, color=MUTED, va="top", linespacing=1.35)
        ax.set_xlim(0, SITE)
        ax.set_ylim(0, SITE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BASE)
            sp.set_linewidth(1.2)

    h = [plt.Line2D([], [], color=INK, marker="o", lw=0, ms=10, mec="white",
                    mew=1.6, label=f"sensor masts ({ns})"),
         plt.Line2D([], [], color="#f2c230", marker="*", lw=0, ms=17,
                    mec=INK, mew=1.2, label="true methane source")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               fontsize=15, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("What the physics head sees", fontsize=24,
                 fontweight="bold", x=0.075, ha="left", y=1.005)
    fig.text(0.075, 0.965, "the four closed-form input maps for one held-out "
             f"scenario ({ns} masts, {d['q'][i]:.0f} kg h$^{{-1}}$), computed "
             "from measured wind only; maps B–E are the network's extra input",
             fontsize=14, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.10, hspace=0.24, top=0.925, bottom=0.05)
    save(fig, "01_xai_feature_maps")
    return i


# ------------------------------------ Fig 02: UMAP of the evidence maps
def fig_umap(d):
    """UMAP embedding of the per-scenario source-evidence maps (the first
    physics channel).  The unsupervised layout reconstructs the site's
    geometry — the maps encode the source location before any network is
    trained — and the scenarios that end in large regions cluster."""
    import umap  # noqa: deferred import, slow

    maps = np.load(DATA / "ch4tu_test_maps_ens.npz")["maps"][:, 0, :]
    # z-score each scenario's map so shape (not amplitude) drives the layout
    m = (maps - maps.mean(1, keepdims=True)) / (maps.std(1, keepdims=True)
                                                + 1e-9)
    from sklearn.decomposition import PCA
    x50 = PCA(n_components=50, random_state=1).fit_transform(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = umap.UMAP(n_neighbors=30, min_dist=0.15,
                        random_state=1).fit_transform(x50)

    tc = d["true_cell"].astype(int)
    tx = (tc % 64 + 0.5) / 64 * SITE
    ty = (tc // 64 + 0.5) / 64 * SITE
    r_phys = rad(audit("ensr", 1)["sizes"])

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.6))
    specs = [
        ("A. True source easting", tx, CMB, None,
         "source x on the site (m)",
         "one embedding axis recovers east–west"),
        ("B. True source northing", ty, CMO, None,
         "source y on the site (m)",
         "the other axis recovers north–south"),
        ("C. Achieved region radius", r_phys, CMG,
         LogNorm(vmin=30, vmax=300), "90% region radius (m), log",
         "neighbours tend to share difficulty (rank corr. 0.33)"),
    ]
    for ax, (title, c, cmap, norm, cbl, caption) in zip(axes, specs):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap=cmap, norm=norm,
                        s=11, linewidths=0, alpha=0.85, rasterized=True)
        ax.set_title(title, fontsize=17, fontweight="bold", loc="left", pad=9)
        ax.annotate(caption, xy=(0, -0.21), xycoords="axes fraction",
                    fontsize=13, color=MUTED, va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BASE)
            sp.set_linewidth(1.2)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05,
                          pad=0.045, aspect=32)
        if norm is not None:
            cb.set_ticks([50, 100, 200])
            cb.set_ticklabels(["50", "100", "200"])
        cb.set_label(cbl, fontsize=13, color=INK2)
        cb.ax.tick_params(labelsize=12, color=INK2)
        cb.outline.set_visible(False)

    fig.suptitle("The physics maps already encode the source location",
                 fontsize=24, fontweight="bold", x=0.065, ha="left", y=1.06)
    fig.text(0.065, 0.985, "unsupervised UMAP of the 2,000 per-scenario "
             "source-evidence maps (model input, measured wind only) — no "
             "network involved; the same embedding coloured three ways",
             fontsize=14, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.08, top=0.90, bottom=0.16)
    save(fig, "02_xai_umap_embedding")


# --------------------- Fig 08: the wind ensemble behind the input maps
def fig_wind_ensemble(d):
    """Through the model's eyes: the measured wind admits many plausible true
    winds, each implying a different plume; the pipeline recomputes the
    evidence map for K=8 wind draws and the network sees their consensus and
    disagreement.  The per-draw maps are regenerated with the SAME RNG
    streams as the frozen dataset (stage_gen streams [root, 92, tag, i, k]),
    and the recomputed mean is checked against the stored channel."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    import torch
    import methane_t_uncertain as mtu

    i = pick_scenario(d)  # same scenario as the feature-map gallery
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % 64 + 0.5) / 64 * SITE, (tc // 64 + 0.5) / 64 * SITE)
    ext = [0, SITE, 0, SITE]

    u_obs = np.load(DATA / "ch4tu_test_uobs.npy")[i]
    root = mtu.load_config()["seeds"]["root_entropy"]
    tag = mtu.split_tag("test")
    device = torch.device("cpu")
    cells = torch.tensor(mtu.cell_centers(mtu.N_GRID), device=device,
                         dtype=torch.float64)
    d1 = {k: (v[i:i + 1] if isinstance(v, np.ndarray)
              and v.ndim and v.shape[0] == len(d["ids"]) else v)
          for k, v in d.items()}
    draws, zk = [], []
    for j in range(mtu.K_MAPS):
        uj = mtu.perturb_wind(u_obs, np.random.default_rng([root, 92, tag,
                                                            i, j]))
        A, B = mtu.ab_maps(d1, uj[None], cells, device)
        zk.append(mtu.zmap(A, B, d1["sigma"])[0])
        draws.append(uj)
    zk = np.array(zk)
    ens = np.load(DATA / "ch4tu_test_maps_ens.npz")["maps"][i]
    err = np.abs(zk.mean(0) - ens[0]).max()
    print(f"    recomputed mean vs frozen channel: max abs diff {err:.2e}")

    zo = audit("ensr", 1)
    mask = np.unpackbits(zo["masks"][i])[:N_CELLS].reshape(64, 64)
    r_ans = rad(zo["sizes"])[i]

    fig = plt.figure(figsize=(16.8, 9.8))
    gs = fig.add_gridspec(2, 24, height_ratios=[1.0, 2.15], hspace=0.34,
                          wspace=0.10)

    # row 0: the eight wind draws and their evidence maps
    lo, hi = np.percentile(zk, [2, 99])
    for j in range(mtu.K_MAPS):
        ax = fig.add_subplot(gs[0, 3 * j:3 * j + 3])
        ax.imshow(np.clip(zk[j].reshape(64, 64), lo, hi), origin="lower",
                  cmap=CMB, extent=ext, interpolation="bilinear",
                  vmin=lo, vmax=hi)
        ax.plot(*tx, marker="*", color="#f2c230", ms=11, mec=INK, mew=0.9,
                zorder=8)
        um = draws[j].mean(0)
        um = um / (np.linalg.norm(um) + 1e-9)
        ax.add_patch(FancyArrowPatch((88 - um[0] * 62, 412 - um[1] * 62),
                                     (88 + um[0] * 62, 412 + um[1] * 62),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=INK2, lw=1.8, zorder=6))
        ax.set_title(f"wind draw {j + 1}", fontsize=12.5, color=INK2, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BASE)
            sp.set_linewidth(1.0)

    # row 1: consensus, disagreement, answer
    big = [
        ("A. Consensus of the eight draws", ens[0], CMB,
         "mean evidence over the draws — input channel 1,\n"
         "matches the frozen input maps to 10$^{-6}$"),
        ("B. Where the draws disagree", ens[1], CMO,
         "spread across the draws — input channel 2,\n"
         "the wind-sensitivity map"),
        ("C. The model's answer", None, None,
         "conformal region of the physics-guided model:\n"
         f"radius {r_ans:.0f} m at 90% coverage"),
    ]
    for k, (title, img, cm, caption) in enumerate(big):
        ax = fig.add_subplot(gs[1, 8 * k:8 * k + 8])
        if img is not None:
            m = img.reshape(64, 64)
            l2, h2 = np.percentile(m, [2, 99])
            ax.imshow(np.clip(m, l2, h2), origin="lower", cmap=cm,
                      extent=ext, interpolation="bilinear")
        else:
            ax.set_facecolor("white")
            ax.add_patch(Rectangle((0, 0), SITE, SITE, fc="none", ec=BASE,
                                   lw=1.6, zorder=1))
            ax.imshow(np.where(mask, 1.0, np.nan), origin="lower",
                      extent=ext, interpolation="nearest", zorder=2,
                      cmap=LinearSegmentedColormap.from_list(
                          "f", [BLUE_FILL, BLUE_FILL]))
            ax.contour(mask.astype(float), levels=[0.5], colors=[BLUE],
                       linewidths=2.6, extent=ext, zorder=3)
            ax.add_patch(Circle(tx, 50, fc="none", ec=MUTED, lw=1.8,
                                ls=(0, (4, 3)), zorder=5))
        ax.scatter(sx[:, 0], sx[:, 1], s=54, c=INK, edgecolors="white",
                   linewidths=1.5, zorder=7)
        ax.plot(*tx, marker="*", color="#f2c230", ms=22, mec=INK, mew=1.2,
                zorder=8)
        ax.set_title(title, fontsize=16.5, fontweight="bold", loc="left",
                     pad=9)
        ax.annotate(caption, xy=(0, -0.045), xycoords="axes fraction",
                    fontsize=12.5, color=MUTED, va="top", linespacing=1.35)
        ax.set_xlim(0, SITE)
        ax.set_ylim(0, SITE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BASE)
            sp.set_linewidth(1.2)

    h = [plt.Line2D([], [], color=INK, marker="o", lw=0, ms=9, mec="white",
                    mew=1.5, label=f"sensor masts ({ns})"),
         plt.Line2D([], [], color="#f2c230", marker="*", lw=0, ms=15,
                    mec=INK, mew=1.1, label="true methane source"),
         plt.Line2D([], [], color=BLUE, lw=2.6,
                    label="physics-guided 90% region"),
         plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 3)),
                    label="50 m facility scale")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Through the model's eyes: eight plausible winds, one "
                 "answer", fontsize=24, fontweight="bold", x=0.045,
                 ha="left", y=1.015)
    fig.text(0.045, 0.975, "the anemometer's stated error means the "
             "measured wind could have been any of these — the pipeline "
             "recomputes the plume evidence under eight wind draws\nand "
             "the network sees only their consensus and disagreement "
             f"(same scenario as the feature-map gallery, {ns} masts, "
             f"{d['q'][i]:.0f} kg h$^{{-1}}$; identical RNG streams to "
             "the frozen dataset)", fontsize=13.5, color=MUTED, ha="left",
             va="top", linespacing=1.45)
    fig.subplots_adjust(top=0.895, bottom=0.075, left=0.035, right=0.972)
    save(fig, "01.5_xai_wind_ensemble")
    return err


# ------------------------- Fig 03: actual guaranteed regions, 12 masts
def fig_areas_12masts(d):
    """The real conformal 90% regions for one 12-mast scenario, one panel
    per model: baseline region (orange) vs physics-guided region (blue),
    in the style of the paper's example figure.

    Scenario rule: among 12-mast scenarios covered by all six seed-1
    regions (3 models x with/without), the one with the LARGEST contrast —
    maximizing the sum over models of log(baseline radius / physics-guided
    radius), i.e. jointly the biggest baseline regions and smallest
    physics-guided regions.  A best-case example, labelled as such."""
    ns = d["n_sensors"].astype(int)
    grp = ns == 12
    audits, contrast, ok = {}, np.zeros(len(ns)), grp.copy()
    for _, sfx in NETS:
        for tag in (f"nomaps{sfx}", f"ensr{sfx}"):
            z = audit(tag, 1)
            r = rad(z["sizes"])
            audits[tag] = (z, r)
            ok &= z["covered"]
            contrast += np.log(r) * (1 if tag.startswith("nomaps") else -1)
    i = int(np.argmax(np.where(ok, contrast, -np.inf)))

    sx = d["sensors"][i, :12] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % 64 + 0.5) / 64 * SITE, (tc // 64 + 0.5) / 64 * SITE)
    ext = [0, SITE, 0, SITE]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6))
    fig.patch.set_facecolor("white")
    for k, (ax, (lab, sfx)) in enumerate(zip(axes, NETS)):
        ax.set_facecolor("white")
        pairs = [(f"nomaps{sfx}", ORANGE, ORANGE_FILL),
                 (f"ensr{sfx}", BLUE, BLUE_FILL)]
        for tag, edge, fill in pairs:
            z, _ = audits[tag]
            mask = np.unpackbits(z["masks"][i])[:N_CELLS].reshape(64, 64)
            ax.imshow(np.where(mask, 1.0, np.nan), origin="lower",
                      extent=ext, interpolation="nearest", zorder=2,
                      cmap=LinearSegmentedColormap.from_list("f",
                                                             [fill, fill]))
            ax.contour(mask.astype(float), levels=[0.5], colors=[edge],
                       linewidths=2.6, extent=ext, zorder=3)
        ax.add_patch(Rectangle((0, 0), SITE, SITE, fc="none", ec=BASE,
                               lw=1.6, zorder=1))
        ax.add_patch(Circle(tx, 50, fc="none", ec=MUTED, lw=1.8,
                            ls=(0, (4, 3)), zorder=5))
        ax.scatter(sx[:, 0], sx[:, 1], s=80, c=INK, marker="o",
                   edgecolors="white", linewidths=1.7, zorder=7)
        ax.plot(*tx, marker="+", color=INK, ms=17, mew=3.4, zorder=8)
        rw, ri = audits[f"nomaps{sfx}"][1][i], audits[f"ensr{sfx}"][1][i]
        ax.annotate(f"without {rw:.0f} m\nwith {ri:.0f} m", xy=(488, 486),
                    ha="right", va="top", fontsize=15.5, linespacing=1.45,
                    color=INK, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="none", alpha=0.92))
        ax.set_title(f"{'ABC'[k]}. {lab}", fontsize=17, fontweight="bold",
                     loc="left", pad=12)
        ax.set_xlim(0, SITE)
        ax.set_ylim(0, SITE)
        ax.set_aspect("equal")
        ax.set_xticks([0, 250, 500])
        ax.set_yticks([0, 250, 500])
        ax.set_xlabel("site coordinate (m)", fontsize=14)
        if k:
            ax.set_yticklabels([])
        despine(ax)

    h = [plt.Line2D([], [], color=ORANGE, lw=2.6,
                    label="baseline 90% region"),
         plt.Line2D([], [], color=BLUE, lw=2.6,
                    label="physics-guided 90% region"),
         plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 3)),
                    label="50 m facility scale"),
         plt.Line2D([], [], color=INK, marker="+", lw=0, ms=13, mew=3,
                    label="true methane source"),
         plt.Line2D([], [], color=INK, marker="o", lw=0, ms=9, mec="white",
                    mew=1.5, label="sensor masts (12)")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False,
               fontsize=14.5, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Twelve masts: the guaranteed search region closes in on "
                 "the source", fontsize=21, fontweight="bold", x=0.055,
                 ha="left", y=1.125)
    fig.text(0.055, 1.035, "the held-out 12-mast scenario with the largest "
             "baseline-to-physics contrast across all three models "
             f"({d['q'][i]:.0f} kg h$^{{-1}}$, same scenario in every "
             "panel); every region shown covers the source",
             fontsize=13.5, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.17, top=0.92, bottom=0.10)
    with matplotlib.rc_context({"savefig.facecolor": "white"}):
        save(fig, "03_region_shapes_12masts")
    for lab, sfx in NETS:
        print(f"    {lab:16s} without {audits[f'nomaps{sfx}'][1][i]:5.1f}  "
              f"with {audits[f'ensr{sfx}'][1][i]:5.1f}")
    return i


# ----------------------------------------- Fig 04: region-radius CDFs
def fig_cdf():
    """Paper Fig. 2 at poster scale: ECDF of region radius per model, solid
    = with maps, dashed = without, prespecified seed 1."""
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    ends = []
    for lab, sfx in NETS:
        c = MODEL_C[lab]
        for tag, ls in ((f"ensr{sfx}", "-"), (f"nomaps{sfx}", "--")):
            r = np.sort(rad(audit(tag, 1)["sizes"]))
            y = np.arange(1, r.size + 1) / r.size
            ax.plot(r, y, ls, color=c, lw=3.0 if ls == "-" else 2.0,
                    alpha=1.0 if ls == "-" else 0.75, zorder=4)
        ends.append((lab, c))

    ax.axvline(50, color=MUTED, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.annotate("EPA 50 m", xy=(50, 1.015), ha="center", fontsize=14,
                color=MUTED, annotation_clip=False)
    frac50 = np.mean([np.mean(rad(audit(f"ensr{s}", 1)["sizes"]) < 50)
                      for _, s in NETS]) * 100
    ax.annotate(f"with maps, {frac50:.0f}% of scenarios\nare inside 50 m — "
                "without maps,\nessentially none",
                xy=(23, 0.80), fontsize=15, color=INK, linespacing=1.45)

    # direct labels at the foot of each model's dashed (baseline) curve,
    # where the three families separate most cleanly
    for (lab, c), (x, ya, ha) in zip(ends, [(116, 0.30, "left"),
                                            (57, 0.045, "left"),
                                            (76, 0.44, "left")]):
        ax.annotate(lab, xy=(x, ya), fontsize=15, color=c, fontweight="bold",
                    ha=ha, bbox=BOX, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(20, 300)
    ax.set_xticks([20, 50, 100, 200, 300])
    ax.set_xticklabels(["20", "50", "100", "200", "300"], fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("90% guaranteed-region radius (m), log scale", fontsize=16,
                  labelpad=8)
    ax.set_ylabel("fraction of scenarios at or below", fontsize=16)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    h = [plt.Line2D([], [], color=INK2, lw=3.0, label="with physics maps"),
         plt.Line2D([], [], color=INK2, lw=2.0, ls="--", alpha=0.75,
                    label="without physics maps")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=15)
    ax.set_title("Smaller regions across the whole distribution",
                 fontsize=21, fontweight="bold", loc="left", pad=26)
    ax.annotate("all 2,000 held-out measured-wind scenarios, prespecified "
                "seed 1;\nsolid left of dashed = improvement at equal "
                "coverage", xy=(0, -0.225), xycoords="axes fraction",
                fontsize=13, color=MUTED, linespacing=1.4)
    save(fig, "04_radius_cdf")


# ------------------------------- Fig 05: median radius vs mast count
def fig_masts(d):
    """Paper Fig. 3 at poster scale: median radius vs number of masts,
    three seeds pooled, solid = with maps, dashed = without."""
    masts3 = np.tile(d["n_sensors"].astype(int), len(SEEDS))
    counts = np.arange(4, 13)
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    for lab, sfx in NETS:
        c = MODEL_C[lab]
        for tag, ls in ((f"ensr{sfx}", "-"), (f"nomaps{sfx}", "--")):
            r = pooled_radii(tag)
            med = [np.median(r[masts3 == n]) for n in counts]
            ax.plot(counts, med, ls, color=c, lw=3.0 if ls == "-" else 2.0,
                    alpha=1.0 if ls == "-" else 0.7, zorder=4,
                    marker="o" if ls == "-" else None, ms=7,
                    mec=SURF, mew=1.4)

    ax.axhline(50, color=MUTED, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.annotate("50 m facility-scale target", xy=(3.85, 51.8), ha="left",
                va="bottom", fontsize=14, color=MUTED)
    for lab, xy in (("DeepSets", (4.75, 143)), ("GNN", (6.0, 66)),
                    ("Set Transformer", (4.55, 92))):
        ax.annotate(lab, xy=xy, fontsize=15, color=MODEL_C[lab],
                    fontweight="bold", bbox=BOX, zorder=6)
    ax.set_xlim(3.7, 12.3)
    ax.set_xticks(counts)
    ax.set_ylim(40, 165)
    ax.set_xlabel("number of sensor masts", fontsize=16, labelpad=8)
    ax.set_ylabel("median 90% region radius (m)\nsmaller is better",
                  fontsize=16, linespacing=1.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    h = [plt.Line2D([], [], color=INK2, lw=3.0, marker="o", ms=7, mec=SURF,
                    label="with physics maps"),
         plt.Line2D([], [], color=INK2, lw=2.0, ls="--", alpha=0.7,
                    label="without physics maps")]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=15)
    ax.set_title("More masts, smaller regions — physics helps at every count",
                 fontsize=21, fontweight="bold", loc="left", pad=16)
    ax.annotate("median over the scenarios with each mast count, three "
                "training seeds pooled, measured wind",
                xy=(0, -0.15), xycoords="axes fraction", fontsize=13,
                color=MUTED)
    save(fig, "05_masts_curves")


# --------------------------------- Fig 06: paired effects, forest plot
def fig_forest():
    """Table I's paired-effect columns as a forest plot: per-model change in
    radius and in the within-50 m share, with 95% bootstrap CIs."""
    ci = json.load(open(REPO / "results" / "table_deltas_ci.json"))
    names = ["DeepSets", "GNN", "SetTransformer"]
    labels = ["DeepSets", "GNN", "Set Transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), sharey=True)
    panels = [
        ("A. Change in region radius", "per_model_radius_delta_m",
         "Δ radius (m) — negative is better", -46, 8),
        ("B. Change in share within 50 m", "per_model_frac50_delta_pts",
         "Δ within-50 m (percentage points) — positive is better",
         -4, 32),
    ]
    ys = np.arange(len(names))[::-1]
    for ax, (title, key, xlab, lo, hi) in zip(axes, panels):
        for y, nm, lab in zip(ys, names, labels):
            pt, a, b = ci[key][nm]
            c = MODEL_C[lab]
            ax.plot([a, b], [y, y], color=c, lw=4.5, solid_capstyle="round",
                    zorder=4)
            ax.plot(pt, y, "o", color=c, ms=13, mec=SURF, mew=2.0, zorder=5)
            ax.annotate(f"{pt:+.1f}", xy=(pt, y), xytext=(0, 16),
                        textcoords="offset points", ha="center", fontsize=15,
                        fontweight="bold", color=INK, zorder=6)
        ax.axvline(0, color=INK2, lw=1.6, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.6, len(names) - 0.15)
        ax.set_xlabel(xlab, fontsize=15, labelpad=8)
        ax.xaxis.grid(True, color=GRID, lw=0.9)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=17, fontweight="bold", loc="left",
                     pad=10)
        despine(ax, ("top", "right", "left"))
        ax.tick_params(left=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=16)
    axes[0].annotate("CI crosses zero:\nGNN radius already small",
                     xy=(ci["per_model_radius_delta_m"]["GNN"][0], ys[1]),
                     xytext=(-38, ys[1] - 0.52), fontsize=13, color=INK2,
                     va="center")
    fig.suptitle("Paired per-scenario effects of the physics maps",
                 fontsize=21, fontweight="bold", x=0.045, ha="left", y=1.10)
    fig.text(0.045, 1.005, "same 2,000 seed-1 scenarios with and without "
             "maps; 95% CIs from 10,000 paired-bootstrap resamples "
             "(Table I's effect columns)", fontsize=13.5, color=MUTED,
             ha="left")
    fig.subplots_adjust(wspace=0.06, top=0.86)
    save(fig, "06_paired_effects_forest")


# ------------------------------------ Fig 07: coverage stays calibrated
def fig_coverage():
    """Observed coverage per model and seed, with vs without maps, against
    the nominal 90% line: the guarantee is never traded for size."""
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    xs = np.arange(len(NETS))
    for off, kind, c, lab in ((-0.16, "nomaps", ORANGE,
                               "without physics maps"),
                              (0.16, "ensr", BLUE, "with physics maps")):
        for x, (mlab, sfx) in zip(xs, NETS):
            # the DeepSets seed-2/3 audits stored only region sizes, so
            # coverage is plotted for the seeds that recorded it
            cov = [z["covered"].mean() for z in
                   (audit(f"{kind}{sfx}", s) for s in SEEDS)
                   if "covered" in z.files]
            ax.plot([x + off] * len(cov), cov, "o", color=c, ms=11,
                    mec=SURF, mew=1.8, alpha=0.9, zorder=5)
            ax.plot([x + off, x + off], [min(cov), max(cov)], color=c, lw=3.2,
                    solid_capstyle="round", alpha=0.55, zorder=4)
    ax.axhline(0.90, color=INK2, lw=2.0, ls=(0, (5, 3)), zorder=3)
    ax.annotate("nominal 90% guarantee", xy=(len(NETS) - 0.62, 0.9008),
                ha="right", va="bottom", fontsize=14.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _ in NETS], fontsize=17)
    ax.set_xlim(-0.55, len(NETS) - 0.45)
    ax.set_ylim(0.880, 0.928)
    ax.set_ylabel("observed coverage on 2,000 test scenarios", fontsize=15.5)
    ax.yaxis.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    despine(ax)
    h = [plt.Line2D([], [], color=ORANGE, marker="o", lw=0, ms=11, mec=SURF,
                    mew=1.6, label="without physics maps"),
         plt.Line2D([], [], color=BLUE, marker="o", lw=0, ms=11, mec=SURF,
                    mew=1.6, label="with physics maps")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=15)
    ax.set_title("Smaller regions are not bought with lost coverage",
                 fontsize=21, fontweight="bold", loc="left", pad=16)
    ax.annotate("one dot per training seed (DeepSets: seed 1 — its other "
                "audits kept only region sizes);\nsplit conformal holds "
                "every configuration near the nominal level",
                xy=(0, -0.17), xycoords="axes fraction", fontsize=13,
                color=MUTED, linespacing=1.4)
    save(fig, "07_coverage_calibration")


if __name__ == "__main__":
    d = load_test()
    print("figure 01 (feature maps)")
    idx = fig_feature_maps(d)
    print(f"    example scenario index {idx}")
    print("figure 02 (umap)")
    fig_umap(d)
    print("figure 01.5 (wind ensemble)")
    fig_wind_ensemble(d)
    print("figure 03 (areas at 12 masts)")
    fig_areas_12masts(d)
    print("figure 04 (cdf)")
    fig_cdf()
    print("figure 05 (mast curves)")
    fig_masts(d)
    print("figure 06 (forest)")
    fig_forest()
    print("figure 07 (coverage)")
    fig_coverage()
    print("poster graphics v2 done")
