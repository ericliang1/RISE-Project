"""Extra figures: dataset gallery, supervision targets, posterior zoo, dataset
statistics, per-scenario improvement, memorization ablation, H3 knob panels.

  fig6_dataset.pdf       gallery of scenarios: plume field + sensors + wind
  fig7_targets.pdf       point-label target vs raw posterior vs tempered teacher
  fig8_zoo.pdf           exact / M0 / M1 heatmaps across difficulty quantiles
  fig9_stats.pdf         benchmark statistics panel
  fig10_improvement.pdf  inefficiency ECDFs + paired per-scenario improvement
  fig11_memorization.pdf D4-augmentation ablation training curves
  fig12_knobs.pdf        noise-sweep curves + geometry paired scatter
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm

from common import get_device, load_config, resolve
from forward_model import unit_response
from inference import load_split
from model import smoothed_targets
from train import blur_teacher

C_BASE = "#2a78d6"
C_EXACT = "#008300"
C_IMPR = "#e87ba4"
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


def panel(ax, title=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)


def heat(ax, P, n, cmap, gamma=0.35):
    v = float(np.max(P))
    ax.imshow(P.reshape(n, n), origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
              norm=PowerNorm(gamma, vmin=0.0, vmax=v if v > 0 else 1.0),
              interpolation="nearest")


def star(ax, xy):
    ax.scatter(*xy, marker="*", s=90, c="white", edgecolors=C_TEXT,
               linewidths=0.9, zorder=6)


def hpd_areas(P, mass=0.90):
    """Vectorized raw HPD-90 area fraction per row."""
    Ps = np.sort(P, axis=1)[:, ::-1]
    cum = np.cumsum(Ps, axis=1)
    return (1 + (cum < mass).sum(axis=1)) / P.shape[1]


# ---------------------------------------------------------------- fig 6
def fig6_dataset(cfg, fig_dir, data_dir, device):
    ph = cfg["physics"]
    d = load_split(data_dir, "test")
    # spread over sensor count and noise: sort by N then sigma, take 8 spread
    order = np.lexsort((d["sigma"], d["n_sensors"]))
    picks = order[np.linspace(40, len(order) - 40, 8).astype(int)]
    ng = 128
    cc = (np.arange(ng) + 0.5) / ng
    X, Y = np.meshgrid(cc, cc)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), device=device,
                       dtype=torch.float64)
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.5))
    for ax, i in zip(axes.ravel(), picks):
        t = lambda a: torch.tensor(np.asarray(a, np.float64), device=device)
        A = unit_response(pts, torch.full((ng * ng,), 1.0, device=device,
                                          dtype=torch.float64),
                          t(d["xs"][i]).expand(ng * ng, 2),
                          t(d["u"][i]).expand(ng * ng, 2),
                          t([d["D"][i]]).expand(ng * ng),
                          ph["sigma_s"], ph["quad_nodes"], ph["tau_max"])
        C = (d["q"][i] * A).cpu().numpy()
        heat(ax, C, ng, "Purples", gamma=0.4)
        ns = int(d["n_sensors"][i])
        vals = np.arcsinh((d["readings"][i, :ns] * d["keep"][i, :ns]).sum(1)
                          / np.maximum(d["keep"][i, :ns].sum(1), 1)
                          / d["sigma"][i])
        sc = ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1],
                        marker="^", s=22, c=vals, cmap="Greys", vmin=-1,
                        edgecolors=C_TEXT, linewidths=0.5, zorder=5)
        star(ax, d["xs"][i])
        u = d["u"][i]
        ax.annotate("", xy=(0.14 + 0.05 * u[0] / np.hypot(*u) * 2,
                            0.86 + 0.05 * u[1] / np.hypot(*u) * 2),
                    xytext=(0.14, 0.86),
                    arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.2))
        panel(ax, f"N={ns}, $\\sigma$={d['sigma'][i]:.3f}, "
                  f"$\\|u\\|$={np.hypot(*u):.1f}")
    fig.suptitle("Benchmark scenarios: concentration field at $t{=}1$, sensors "
                 "(shaded by reading), wind, true source", fontsize=8, y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_dataset.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 7
def fig7_targets(cfg, fig_dir, data_dir):
    n = cfg["grid"]["n"]
    d = load_split(data_dir, "test")
    P_e = np.load(data_dir / "test_posterior.npz")["probs"]
    areas = hpd_areas(P_e)
    i = int(np.argsort(areas)[int(0.75 * len(areas))])   # structured/ambiguous

    tc = torch.tensor(d["true_cell"][i:i + 1])
    m = cfg["model"]
    bump = smoothed_targets(tc, n, m["target_smooth_std_cells"],
                            m["target_trunc_sigmas"], "cpu")[0].numpy()
    raw = P_e[i]
    temp = blur_teacher(torch.tensor(raw[None].astype(np.float32)), n,
                        0.75)[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    for ax, P, ttl, cm in [
            (axes[0], bump, "M0 target: smoothed point label", "Blues"),
            (axes[1], raw, "Exact posterior (raw teacher)", "Greens"),
            (axes[2], temp, "Tempered teacher (0.75-cell blur)", "Greens")]:
        heat(ax, P, n, cm)
        star(ax, d["xs"][i])
        ns = int(d["n_sensors"][i])
        ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1],
                   marker="^", s=16, c=C_TEXT, edgecolors="white",
                   linewidths=0.4, zorder=5)
        panel(ax, ttl)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_targets.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 8
def fig8_zoo(cfg, fig_dir, data_dir):
    n = cfg["grid"]["n"]
    d = load_split(data_dir, "test")
    P_e = np.load(data_dir / "test_posterior.npz")["probs"]
    P_0 = np.load(data_dir / "heatmaps_model_seed1_test.npz")["probs"]
    P_1 = np.load(data_dir / "heatmaps_model2_seed1_test.npz")["probs"]
    areas = hpd_areas(P_e)
    qs = [0.05, 0.30, 0.55, 0.80, 0.95]
    picks = [int(np.argsort(areas)[int(q * len(areas))]) for q in qs]

    fig, axes = plt.subplots(3, len(picks), figsize=(6.8, 4.3))
    rows = [("Exact posterior", P_e, "Greens"),
            ("M0 (point labels)", P_0, "Blues"),
            ("M1 (distilled)", P_1, "RdPu")]
    for r, (name, P, cm) in enumerate(rows):
        for c, i in enumerate(picks):
            ax = axes[r, c]
            heat(ax, P[i].astype(np.float64), n, cm)
            star(ax, d["xs"][i])
            panel(ax, f"oracle area {areas[picks[c]]:.3f}" if r == 0 else None)
        axes[r, 0].set_ylabel(name, fontsize=7.5)
    fig.suptitle("Posterior shape across difficulty (columns: oracle HPD-90 "
                 "area quantiles 5–95%)", fontsize=8, y=0.99)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_zoo.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 9
def fig9_stats(cfg, fig_dir, data_dir):
    d = load_split(data_dir, "train")
    P_e = np.load(data_dir / "test_posterior.npz")["probs"]
    areas = hpd_areas(P_e)
    snr = []
    for i in range(0, len(d["ids"]), 4):
        ns = int(d["n_sensors"][i])
        r = d["readings"][i, :ns][d["keep"][i, :ns]]
        snr.append(np.abs(r).max() / d["sigma"][i])
    snr = np.array(snr)

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6))
    def hist(ax, x, bins, ttl, log=False, color=C_BASE):
        ax.hist(x, bins=bins, color=color, alpha=0.85)
        if log:
            ax.set_xscale("log")
        ax.set_title(ttl, fontsize=7.5)
        ax.grid(color=C_GRID, lw=0.4)
    hist(axes[0, 0], d["q"], np.logspace(np.log10(0.5), np.log10(5), 25),
         "release rate $q$", log=True)
    hist(axes[0, 1], d["sigma"], np.logspace(-2, -1, 25),
         "noise std $\\sigma$", log=True)
    axes[0, 2].bar(*np.unique(d["n_sensors"], return_counts=True),
                   color=C_BASE, alpha=0.85)
    axes[0, 2].set_title("sensor count $N$", fontsize=7.5)
    axes[0, 2].grid(color=C_GRID, lw=0.4)
    hist(axes[1, 0], np.clip(snr, 1e-2, None),
         np.logspace(-2, 4, 30), "peak reading SNR (max$|y|/\\sigma$)",
         log=True)
    hist(axes[1, 1], areas, np.logspace(np.log10(2e-4), 0, 30),
         "oracle HPD-90 area (test)", log=True, color=C_EXACT)
    err_e = np.linalg.norm(
        np.stack([(np.argmax(P_e, 1) % 64 + 0.5) / 64,
                  (np.argmax(P_e, 1) // 64 + 0.5) / 64], -1)
        - load_split(data_dir, "test")["xs"], axis=1)
    hist(axes[1, 2], np.clip(err_e, 1e-3, None),
         np.logspace(-3, 0, 30), "oracle MAP error (test)", log=True,
         color=C_EXACT)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_stats.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 10
def fig10_improvement(cfg, fig_dir, data_dir):
    z0 = np.load(data_dir / "audit_scatter_model_seed1.npz")
    z1 = np.load(data_dir / "audit_scatter_model2_seed1.npz")
    r0 = z0["area_learned"] / z0["area_exact"]
    r1 = z1["area_learned"] / z1["area_exact"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    for r, col, name in [(r0, C_BASE, "M0"), (r1, C_IMPR, "M1")]:
        xs = np.sort(r)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, lw=1.8,
                label=f"{name} (median {np.median(r):.1f}$\\times$)")
        ax.axvline(np.median(r), color=col, lw=0.8, ls=":")
    ax.axvline(1.0, color=C_MUT, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("inefficiency ratio (learned area / oracle area)")
    ax.set_ylabel("fraction of test scenarios")
    ax.set_title("Inefficiency distribution")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=C_GRID, lw=0.4)

    ax = axes[1]
    lim = [min(r0.min(), r1.min()) * 0.8, max(r0.max(), r1.max()) * 1.2]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8)
    ax.scatter(r0, r1, s=4, alpha=0.2, c=C_IMPR, edgecolors="none",
               rasterized=True)
    frac = float((r1 < r0).mean())
    ax.annotate(f"M1 sharper on {frac:.0%} of scenarios",
                (0.05, 0.92), xycoords="axes fraction", fontsize=7.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("M0 inefficiency ratio")
    ax.set_ylabel("M1 inefficiency ratio")
    ax.set_title("Per-scenario improvement")
    ax.grid(color=C_GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_improvement.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 11
def fig11_memorization(cfg, fig_dir, results_dir):
    rows = list(csv.DictReader(open(results_dir / "memorization_curves.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for aug, col, name in [("1", C_BASE, "with D4 augmentation"),
                           ("0", "#d03b3b", "without augmentation")]:
        e = [int(r["epoch"]) for r in rows if r["aug"] == aug]
        tl = [float(r["train_loss"]) for r in rows if r["aug"] == aug]
        vn = [float(r["val_nll"]) for r in rows if r["aug"] == aug]
        axes[0].plot(e, tl, color=col, lw=1.6, label=name)
        axes[1].plot(e, vn, color=col, lw=1.6, label=name)
    axes[1].axhline(np.log(4096), color=C_MUT, lw=0.8, ls="--")
    axes[1].annotate("uniform ($\\log 4096$)", (0.55, 0.93),
                     xycoords="axes fraction", fontsize=6.5, color=C_MUT)
    axes[0].set_ylabel("training loss"); axes[1].set_ylabel("validation NLL")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(color=C_GRID, lw=0.4)
    axes[0].legend(frameon=False)
    axes[0].set_title("Memorization without symmetry augmentation", loc="left",
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_memorization.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 12
def fig12_knobs(cfg, fig_dir, data_dir):
    z0 = np.load(data_dir / "dose_curves_model_seed1.npz")
    z1 = np.load(data_dir / "dose_curves_model2_seed1.npz")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    sig = z0["sig_vals"]
    ax.plot(sig, z0["sweep_e"].mean(0), color=C_EXACT, lw=1.8, label="Exact")
    ax.plot(sig, z0["sweep_l"].mean(0), color=C_BASE, lw=1.8, label="M0")
    ax.plot(sig, z1["sweep_l"].mean(0), color=C_IMPR, lw=1.8, label="M1")
    ax.set_xscale("log")
    ax.set_xlabel("noise std $\\sigma$")
    ax.set_ylabel("mean 90% region area")
    ax.set_title("Knob 3: noise")
    ax.legend(frameon=False)
    ax.grid(color=C_GRID, lw=0.4)

    ax = axes[1]
    lim = [2e-4, 1.3]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8)
    ax.scatter(z0["geo_spread_e"], z0["geo_conf_e"], s=14, c=C_EXACT,
               alpha=0.6, edgecolors="none", label="Exact")
    ax.scatter(z1["geo_spread_l"], z1["geo_conf_l"], s=14, c=C_IMPR,
               alpha=0.6, edgecolors="none", label="M1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("area, well-spread layout")
    ax.set_ylabel("area, confined layout")
    ax.set_title("Knob 2: geometry (points above diagonal widen)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=C_GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_knobs.pdf")
    plt.close(fig)


def main():
    cfg = load_config()
    device = get_device()
    fig_dir = resolve(cfg, "figures_dir")
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig6_dataset(cfg, fig_dir, data_dir, device); print("fig6 done", flush=True)
    fig7_targets(cfg, fig_dir, data_dir); print("fig7 done", flush=True)
    fig8_zoo(cfg, fig_dir, data_dir); print("fig8 done", flush=True)
    fig9_stats(cfg, fig_dir, data_dir); print("fig9 done", flush=True)
    fig10_improvement(cfg, fig_dir, data_dir); print("fig10 done", flush=True)
    if (results_dir / "memorization_curves.csv").exists():
        fig11_memorization(cfg, fig_dir, results_dir); print("fig11 done", flush=True)
    fig12_knobs(cfg, fig_dir, data_dir); print("fig12 done", flush=True)


if __name__ == "__main__":
    main()
