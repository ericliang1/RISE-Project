"""Stage 7: figures F1-F5 with the exact filenames the LaTeX expects.

  fig1_pipeline.pdf          pipeline on a real median-difficulty scenario
  fig2_easy_vs_ambiguous.pdf 2x4: layout / exact / learned / both regions
  fig3_coverage.pdf          empirical vs nominal, raw HPD vs conformal
  fig4_audit.pdf             H2 area-vs-area scatter + D1 area-vs-error scatter
  fig5_doseresponse.pdf      snapshots N=4..12 + area-vs-N curves

Design: validated categorical palette in fixed order -- baseline learned =
blue #2a78d6, exact oracle = green #008300, physics-distilled variant = magenta
#e87ba4; neutral gray reference lines; heatmaps use single-hue colormaps with
a power-law normalization (gamma 0.35) so sharp posteriors stay visible;
conformal regions are drawn as translucent fills with a thin outline.
The distilled-model series appear only if its artifacts exist.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from common import get_device, load_config, resolve
from conformal import region_mask, tail_threshold
from exact_posterior import log_posterior_scenario
from inference import ckpt_stem, heatmaps, load_model, load_split
from dose_response import subset_dict

C_BASE = "#2a78d6"      # baseline learned
C_EXACT = "#008300"     # exact oracle
C_IMPR = "#e87ba4"      # improved learned
C_GRID = "#e3e2df"
C_TEXT = "#0b0b0b"
C_MUT = "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def heat(ax, P, n, cmap):
    """Heatmap with power-law normalization so sharp maps remain visible."""
    img = P.reshape(n, n)
    vmax = float(img.max())
    if vmax <= 0:
        vmax = 1.0
    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
              norm=PowerNorm(0.35, vmin=0.0, vmax=vmax),
              interpolation="nearest")


def region_overlay(ax, P, t_hat, n, color, ls="-"):
    """Conformal region as translucent fill + thin outline."""
    mask = region_mask(P.astype(np.float64), t_hat).reshape(n, n).astype(float)
    cc = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    ax.contourf(cc, cc, mask, levels=[0.5, 1.5], colors=[color], alpha=0.18)
    ax.contour(cc, cc, mask, levels=[0.5], colors=[color], linewidths=1.0,
               linestyles=ls)


def marks(ax, d, i):
    ns = int(d["n_sensors"][i])
    ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1],
               marker="^", s=16, c=C_TEXT, edgecolors="white",
               linewidths=0.4, zorder=5)
    ax.scatter(*d["xs"][i], marker="*", s=90, c="white", edgecolors=C_TEXT,
               linewidths=0.9, zorder=6)


def panel(ax, title=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)


def have_model2(data_dir):
    return (data_dir / "audit_scatter_model2_seed1.npz").exists()


# ---------------------------------------------------------------- fig 3
def fig3_coverage(cfg, fig_dir, results_dir):
    with open(results_dir / "conformal_model_seed1.json") as f:
        sweep = json.load(f)["sweep"]
    nom = np.array([r["nominal"] for r in sweep])
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    ax.plot([0.45, 1], [0.45, 1], "--", color=C_MUT, lw=0.8, zorder=1)
    ax.plot(nom, [r["raw_hpd"] for r in sweep], color=C_EXACT, lw=1.8,
            label="Raw HPD (uncalibrated)")
    ax.plot(nom, [r["conformal"] for r in sweep], color=C_BASE, lw=1.8,
            label="Conformal (calibrated)")
    i90 = int(np.argmin(np.abs(nom - 0.90)))
    ax.scatter([0.90], [sweep[i90]["conformal"]], s=22, c=C_BASE, zorder=5)
    ax.annotate(f'{sweep[i90]["conformal"]:.3f} at nominal 0.90',
                (0.90, sweep[i90]["conformal"]), textcoords="offset points",
                xytext=(-8, -14), ha="right", fontsize=6.5, color=C_MUT)
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage (test)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=C_GRID, lw=0.5)
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.45, 1.02)
    fig.savefig(fig_dir / "fig3_coverage.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig4_audit(cfg, fig_dir, data_dir):
    zb = np.load(data_dir / "audit_scatter_model_seed1.npz")
    series = [("Baseline (preregistered)", C_BASE, zb)]
    if have_model2(data_dir):
        series.append(("Physics-distilled (same model)", C_IMPR,
                       np.load(data_dir / "audit_scatter_model2_seed1.npz")))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axes[0]
    all_a = np.concatenate([z["area_learned"] for _, _, z in series]
                           + [zb["area_exact"]])
    lim = [max(all_a[all_a > 0].min() * 0.7, 2e-4), 1.2]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8, zorder=1)
    for name, col, z in series:
        ax.scatter(z["area_exact"], z["area_learned"], s=4, alpha=0.22,
                   c=col, edgecolors="none", label=name, rasterized=True)
        med = np.median(z["area_learned"] / z["area_exact"])
        ax.plot(lim, [min(v * med, 2) for v in lim], color=col, lw=0.9,
                ls=":", zorder=2)
        ax.annotate(f"median ratio {med:.1f}$\\times$",
                    (lim[0] * 1.3, lim[0] * med * 1.45), fontsize=6.5,
                    color=col, rotation=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Exact-posterior region area (fraction)")
    ax.set_ylabel("Learned region area (fraction)")
    ax.set_title("H2: sharpness at matched 90% coverage")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.1,
              markerscale=2.5)
    ax.grid(color=C_GRID, lw=0.5)

    ax = axes[1]
    for name, col, z in series:
        ax.scatter(z["area_learned"], z["map_err_learned"], s=4, alpha=0.22,
                   c=col, edgecolors="none", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Learned region area (fraction)")
    ax.set_ylabel("MAP localization error")
    ax.set_title("D1: area vs. localization error")
    ax.grid(color=C_GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_audit.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig5_dose(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    alpha = cfg["conformal"]["alpha"]
    use2 = have_model2(data_dir)
    snap_tag = "model2" if use2 else "model"
    zb = np.load(data_dir / "dose_curves_model_seed1.npz")
    Ns, base_l, curves_e = zb["Ns"], zb["curves_l"], zb["curves_e"]
    impr_l = (np.load(data_dir / "dose_curves_model2_seed1.npz")["curves_l"]
              if use2 else None)
    snap_curves = impr_l if use2 else base_l

    # flagship: strong, clean exact contraction
    slopes = curves_e[:, -1] - curves_e[:, 0]
    flag = int(np.argsort(slopes)[len(slopes) // 10])
    d1 = load_split(data_dir, "dose1_master")
    t_hat = tail_threshold(
        np.load(data_dir / f"scores_{snap_tag}_seed1.npz")["tails"], alpha)
    model, _ = load_model(
        cfg, resolve(cfg, "checkpoints_dir") / f"{ckpt_stem(snap_tag, 1)}.pt",
        device)

    snapNs = [4, 6, 8, 10, 12]
    fig = plt.figure(figsize=(6.8, 4.1))
    gs = fig.add_gridspec(2, len(snapNs), height_ratios=[1.3, 1], hspace=0.28)
    order = d1["order"][flag]
    snap_col = C_IMPR if use2 else C_BASE
    for j, N in enumerate(snapNs):
        ax = fig.add_subplot(gs[0, j])
        sub = subset_dict(d1, flag, order[:N])
        P = heatmaps(model, sub, device)[0]
        heat(ax, P, n, "Blues")
        region_overlay(ax, P, t_hat, n, snap_col)
        marks(ax, sub, 0)
        panel(ax, f"N = {N}")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(Ns, curves_e.mean(0), color=C_EXACT, lw=2, label="Exact oracle")
    ax.plot(Ns, base_l.mean(0), color=C_BASE, lw=2, label="Baseline")
    if use2:
        ax.plot(Ns, impr_l.mean(0), color=C_IMPR, lw=2, label="Distilled")
    ax.plot(Ns, curves_e[flag], color=C_EXACT, lw=1.1, ls="--", alpha=0.8)
    ax.plot(Ns, snap_curves[flag], color=snap_col, lw=1.1, ls="--", alpha=0.8,
            label="Shown scenario (dashed)")
    ax.set_xlabel("Number of sensors N")
    ax.set_ylabel("90% region area\n(mean over 50 scenarios)")
    ax.set_yscale("log")
    ax.grid(color=C_GRID, lw=0.5)
    ax.legend(frameon=False, ncols=4, loc="lower left")
    fig.savefig(fig_dir / "fig5_doseresponse.pdf")
    plt.close(fig)
    return flag


# ---------------------------------------------------------------- fig 2
def fig2_scenarios(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    alpha = cfg["conformal"]["alpha"]
    use2 = have_model2(data_dir)
    tag = "model2" if use2 else "model"
    lcol = C_IMPR if use2 else C_BASE
    t_l = tail_threshold(
        np.load(data_dir / f"scores_{tag}_seed1.npz")["tails"], alpha)
    t_e = tail_threshold(np.load(data_dir / "scores_exact.npz")["tails"], alpha)
    model, _ = load_model(
        cfg, resolve(cfg, "checkpoints_dir") / f"{ckpt_stem(tag, 1)}.pt", device)
    d_s = load_split(data_dir, "dose2_spread")
    d_c = load_split(data_dir, "dose2_confined")
    z = np.load(data_dir / "dose_curves_model_seed1.npz")
    contrast = z["geo_conf_e"] / np.maximum(z["geo_spread_e"], 1e-9)
    b = int(np.nanargmax(contrast))

    col_titles = ["Sensor layout", "Exact posterior", "Learned heatmap",
                  "Both 90% regions"]
    row_labels = ["Easy\n(well-spread)", "Ambiguous\n(confined)"]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.5))
    for row, dd in enumerate([d_s, d_c]):
        sub = subset_dict(dd, b, np.arange(6))
        P_l = heatmaps(model, sub, device)[0]
        P_e = torch.exp(log_posterior_scenario(sub, 0, cfg, device)).cpu().numpy()
        ax = axes[row, 0]
        marks(ax, sub, 0); panel(ax)
        ax = axes[row, 1]
        heat(ax, P_e, n, "Greens"); marks(ax, sub, 0); panel(ax)
        ax = axes[row, 2]
        heat(ax, P_l, n, "Blues"); marks(ax, sub, 0); panel(ax)
        ax = axes[row, 3]
        region_overlay(ax, P_l, t_l, n, lcol)
        region_overlay(ax, P_e, t_e, n, C_EXACT, ls="--")
        marks(ax, sub, 0); panel(ax)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=7.5)
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=8)
    leg = [Line2D([], [], color=lcol, lw=1.2, label="Learned region"),
           Line2D([], [], color=C_EXACT, lw=1.2, ls="--", label="Exact region"),
           Line2D([], [], color=C_TEXT, marker="^", lw=0, ms=4, label="Sensor"),
           Line2D([], [], color=C_TEXT, marker="*", lw=0, ms=8,
                  markerfacecolor="white", label="True source")]
    fig.legend(handles=leg, frameon=False, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_easy_vs_ambiguous.pdf")
    plt.close(fig)
    return b


# ---------------------------------------------------------------- fig 1
def fig1_pipeline(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    d = load_split(data_dir, "test")
    P_l = np.load(data_dir / "heatmaps_model_seed1_test.npz")["probs"]
    P_e = np.load(data_dir / "test_posterior.npz")["probs"]
    t_l = float(np.load(data_dir / "regions_model_seed1_test.npz")["tail_qhat"])
    sizes = np.load(data_dir / "regions_model_seed1_test.npz")["sizes"]
    i = int(np.argsort(np.abs(sizes - np.median(sizes)))[0])

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.85))
    marks(axes[0], d, i); panel(axes[0], "Sparse sensors\n+ meteorology")
    heat(axes[1], P_l[i], n, "Blues"); marks(axes[1], d, i)
    panel(axes[1], "DeepSets localizer\n(posterior heatmap)")
    heat(axes[2], P_l[i], n, "Blues")
    region_overlay(axes[2], P_l[i], t_l, n, C_BASE)
    marks(axes[2], d, i); panel(axes[2], "90% conformal region")
    heat(axes[3], P_e[i], n, "Greens"); marks(axes[3], d, i)
    panel(axes[3], "Exact grid posterior\n(auditor)")
    for k in range(3):
        fig.add_artist(FancyArrowPatch(
            (0.247 + 0.242 * k, 0.5), (0.268 + 0.242 * k, 0.5),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
            color=C_MUT))
    fig.savefig(fig_dir / "fig1_pipeline.pdf")
    plt.close(fig)


def main():
    cfg = load_config()
    device = get_device()
    fig_dir = resolve(cfg, "figures_dir")
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig_dir.mkdir(exist_ok=True)
    fig3_coverage(cfg, fig_dir, results_dir)
    print("fig3 done", flush=True)
    fig4_audit(cfg, fig_dir, data_dir)
    print("fig4 done", flush=True)
    flag = fig5_dose(cfg, fig_dir, data_dir, device)
    print(f"fig5 done (flagship scenario {flag})", flush=True)
    b = fig2_scenarios(cfg, fig_dir, data_dir, device)
    print(f"fig2 done (geometry scenario {b})", flush=True)
    fig1_pipeline(cfg, fig_dir, data_dir, device)
    print("fig1 done", flush=True)


if __name__ == "__main__":
    main()
