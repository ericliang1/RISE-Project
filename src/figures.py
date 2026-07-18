"""Stage 7: figures F1-F5 with the exact filenames the LaTeX expects.

  fig1_pipeline.pdf          schematic + real example panels
  fig2_easy_vs_ambiguous.pdf 2x4: layout / exact / learned / both regions
  fig3_coverage.pdf          empirical vs nominal, raw HPD vs conformal
  fig4_audit.pdf             H2 area-vs-area scatter + D1 area-vs-error scatter
  fig5_doseresponse.pdf      snapshots N=4..12 + area-vs-N curves

Colors (validated categorical palette, fixed order): learned/conformal = blue
#2a78d6, exact/raw-HPD = green #008300; reference lines neutral gray; single-
hue sequential colormaps for heatmaps; truth marker black star.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import cell_center_of, get_device, load_config, resolve
from conformal import region_mask, tail_threshold
from exact_posterior import log_posterior_scenario
from inference import heatmaps, load_model, load_split
from dose_response import subset_dict

C_LEARN = "#2a78d6"
C_EXACT = "#008300"
C_GRID = "#d5d4d0"
C_TEXT = "#0b0b0b"
C_MUT = "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def grid_img(P, n):
    """(n*n,) cell-ordered probs -> (n, n) image, origin lower (iy row)."""
    return P.reshape(n, n)


def draw_scenario(ax, d, i, P, qhat, n, cmap, rcolor, show_sensors=True,
                  title=None, region=True):
    img = grid_img(P, n)
    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
              interpolation="nearest")
    if region and qhat is not None:
        mask = region_mask(P.astype(np.float64), qhat).reshape(n, n)
        ax.contour(np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                   np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                   mask.astype(float), levels=[0.5], colors=[rcolor],
                   linewidths=1.2)
    if show_sensors:
        ns = int(d["n_sensors"][i])
        ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1],
                   marker="^", s=14, c=C_TEXT, edgecolors="white",
                   linewidths=0.4, zorder=5)
    ax.scatter(*d["xs"][i], marker="*", s=60, c=C_TEXT, edgecolors="white",
               linewidths=0.5, zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.5)


def fig3_coverage(cfg, fig_dir, results_dir):
    with open(results_dir / "conformal_model_seed1.json") as f:
        sweep = json.load(f)["sweep"]
    nom = [r["nominal"] for r in sweep]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0.45, 1], [0.45, 1], "--", color=C_MUT, lw=0.8, zorder=1)
    ax.plot(nom, [r["raw_hpd"] for r in sweep], color=C_EXACT, lw=2,
            label="Raw HPD (uncalibrated)")
    ax.plot(nom, [r["conformal"] for r in sweep], color=C_LEARN, lw=2,
            label="Conformal (calibrated)")
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage (test)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=C_GRID, lw=0.4)
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.45, 1.02)
    fig.savefig(fig_dir / "fig3_coverage.pdf")
    plt.close(fig)


def fig4_audit(cfg, fig_dir, data_dir):
    z = np.load(data_dir / "audit_scatter_model_seed1.npz")
    al, ae, err = z["area_learned"], z["area_exact"], z["map_err_learned"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    ax = axes[0]
    lim = [max(min(ae.min(), al.min()) * 0.7, 2e-4), min(1.0, max(ae.max(), al.max()) * 1.5)]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8)
    ax.scatter(ae, al, s=4, alpha=0.25, c=C_LEARN, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Exact-posterior region area (fraction)")
    ax.set_ylabel("Learned region area (fraction)")
    ax.set_title("H2: sharpness at matched 90% coverage")
    ax.grid(color=C_GRID, lw=0.4, which="both")
    ax = axes[1]
    ax.scatter(al, err, s=4, alpha=0.25, c=C_LEARN, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Learned region area (fraction)")
    ax.set_ylabel("MAP localization error")
    ax.set_title("D1: area vs. localization error")
    ax.grid(color=C_GRID, lw=0.4, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_audit.pdf")
    plt.close(fig)


def fig5_dose(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    n_cells = n * n
    alpha = cfg["conformal"]["alpha"]
    z = np.load(data_dir / "dose_curves_model_seed1.npz")
    Ns, curves_l, curves_e = z["Ns"], z["curves_l"], z["curves_e"]
    # flagship: strongest exact-area contraction with clean monotone trend
    slopes = (curves_e[:, -1] - curves_e[:, 0])
    flag = int(np.argsort(slopes)[len(slopes) // 10])   # strongly contracting
    d1 = load_split(data_dir, "dose1_master")
    qhat_l = tail_threshold(
        np.load(data_dir / "scores_model_seed1.npz")["tails"], alpha)
    model, _ = load_model(cfg, resolve(cfg, "checkpoints_dir") / "seed1.pt", device)

    snapNs = [4, 6, 8, 10, 12]
    fig = plt.figure(figsize=(6.8, 4.2))
    gs = fig.add_gridspec(2, len(snapNs), height_ratios=[1.35, 1], hspace=0.25)
    order = d1["order"][flag]
    for j, N in enumerate(snapNs):
        ax = fig.add_subplot(gs[0, j])
        sub = subset_dict(d1, flag, order[:N])
        P = heatmaps(model, sub, device)[0]
        draw_scenario(ax, sub, 0, P, qhat_l, n, "Blues", C_LEARN,
                      title=f"N = {N}")
    ax = fig.add_subplot(gs[1, :])
    for b in range(curves_l.shape[0]):
        ax.plot(Ns, curves_l[b], color=C_LEARN, alpha=0.08, lw=0.6)
        ax.plot(Ns, curves_e[b], color=C_EXACT, alpha=0.08, lw=0.6)
    ax.plot(Ns, curves_l.mean(0), color=C_LEARN, lw=2, label="Learned (mean)")
    ax.plot(Ns, curves_e.mean(0), color=C_EXACT, lw=2, label="Exact (mean)")
    ax.plot(Ns, curves_l[flag], color=C_LEARN, lw=1.2, ls="--",
            label="Learned (shown scenario)")
    ax.plot(Ns, curves_e[flag], color=C_EXACT, lw=1.2, ls="--",
            label="Exact (shown scenario)")
    ax.set_xlabel("Number of sensors N")
    ax.set_ylabel("90% region area")
    ax.set_yscale("log")
    ax.grid(color=C_GRID, lw=0.4)
    ax.legend(frameon=False, ncols=2)
    fig.savefig(fig_dir / "fig5_doseresponse.pdf")
    plt.close(fig)
    return flag


def fig2_scenarios(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    alpha = cfg["conformal"]["alpha"]
    qhat_l = tail_threshold(
        np.load(data_dir / "scores_model_seed1.npz")["tails"], alpha)
    qhat_e = tail_threshold(
        np.load(data_dir / "scores_exact.npz")["tails"], alpha)
    model, _ = load_model(cfg, resolve(cfg, "checkpoints_dir") / "seed1.pt", device)
    d_s = load_split(data_dir, "dose2_spread")
    d_c = load_split(data_dir, "dose2_confined")
    # pick the base scenario with the largest exact-area contrast
    z = np.load(data_dir / "dose_curves_model_seed1.npz")
    contrast = z["geo_conf_e"] / np.maximum(z["geo_spread_e"], 1e-9)
    b = int(np.nanargmax(contrast))

    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.6))
    for row, (dd, tag) in enumerate([(d_s, "Easy (well-spread)"),
                                     (d_c, "Ambiguous (confined)")]):
        sub = subset_dict(dd, b, np.arange(6))
        P_l = heatmaps(model, sub, device)[0]
        P_e = torch.exp(log_posterior_scenario(sub, 0, cfg, device)).cpu().numpy()
        ax = axes[row, 0]
        draw_scenario(ax, sub, 0, np.zeros_like(P_l), None, n, "Greys", None,
                      title=f"{tag}: layout" if row == 0 else "layout",
                      region=False)
        draw_scenario(axes[row, 1], sub, 0, P_e, None, n, "Greens", None,
                      title="exact posterior" if row == 0 else None, region=False)
        draw_scenario(axes[row, 2], sub, 0, P_l, None, n, "Blues", None,
                      title="learned heatmap" if row == 0 else None, region=False)
        ax = axes[row, 3]
        draw_scenario(ax, sub, 0, P_l * 0, None, n, "Greys", None, region=False,
                      title="both 90% regions" if row == 0 else None)
        for P, qh, col, ls in [(P_l, qhat_l, C_LEARN, "-"),
                               (P_e, qhat_e, C_EXACT, "--")]:
            mask = region_mask(P.astype(np.float64), qh).reshape(n, n)
            ax.contour(np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                       np.linspace(0, 1, n, endpoint=False) + 0.5 / n,
                       mask.astype(float), levels=[0.5], colors=[col],
                       linewidths=1.2, linestyles=ls)
        axes[row, 0].set_ylabel(tag.split(" ")[0], fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_easy_vs_ambiguous.pdf")
    plt.close(fig)
    return b


def fig1_pipeline(cfg, fig_dir, data_dir, device):
    n = cfg["grid"]["n"]
    alpha = cfg["conformal"]["alpha"]
    d = load_split(data_dir, "test")
    P_l = np.load(data_dir / "heatmaps_model_seed1_test.npz")["probs"]
    P_e = np.load(data_dir / "test_posterior.npz")["probs"]
    qhat_l = float(np.load(data_dir / "regions_model_seed1_test.npz")["tail_qhat"])
    # pick a mid-difficulty scenario: median learned area
    sizes = np.load(data_dir / "regions_model_seed1_test.npz")["sizes"]
    i = int(np.argsort(np.abs(sizes - np.median(sizes)))[0])

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9))
    titles = ["Sparse sensors\n+ meteorology", "DeepSets localizer\n(posterior heatmap)",
              "90% conformal region", "Exact grid posterior\n(auditor)"]
    draw_scenario(axes[0], d, i, np.zeros(n * n), None, n, "Greys", None,
                  region=False)
    draw_scenario(axes[1], d, i, P_l[i], None, n, "Blues", None, region=False)
    draw_scenario(axes[2], d, i, P_l[i], qhat_l, n, "Blues", C_LEARN)
    draw_scenario(axes[3], d, i, P_e[i], None, n, "Greens", None, region=False)
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=7.5)
    for k in range(3):
        fig.add_artist(FancyArrowPatch(
            (0.245 + 0.242 * k, 0.5), (0.27 + 0.242 * k, 0.5),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
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
