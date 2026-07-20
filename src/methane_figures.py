"""CH4-primary figure suite: regenerates fig1..fig14 (canonical names) from
the methane testbed, in physical units.  Also reruns the mode-recovery sanity
at sigma = 1e-6 ppm (asymptotic for along-wind ridge degeneracy).

Usage:  python src/methane_figures.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from common import cell_center_of, get_device, load_config, pos_to_cell, resolve, update_json
from conformal import region_mask, regions, tail_scores, tail_threshold
from inference import to_batch
from methane_model import U_SCALE, plume_ppm_per_kgh
from methane_pipeline import SPLITS, ch4_cfg, ch4_posteriors, model_probs, stab_of
from methane_extras import load_ch4
from methane_extras2 import base_scenario, one_scenario
from model import DeepSetsLocalizer, smoothed_targets
from train import blur_teacher

N = 64
C0, C1, CE = "#2a78d6", "#e87ba4", "#008300"
CG, CT, CM = "#e3e2df", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": CM, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})


def heat(ax, P, cmap, gamma=0.35, extent=(0, 500, 0, 500)):
    v = float(np.max(P))
    ax.imshow(P.reshape(N, N) if P.size == N * N else P, origin="lower",
              extent=extent, cmap=cmap,
              norm=PowerNorm(gamma, vmin=0, vmax=v if v > 0 else 1),
              interpolation="nearest")


def marks(ax, d, i, scale=500.0):
    ns = int(d["n_sensors"][i])
    ax.scatter(d["sensors"][i, :ns, 0] * scale, d["sensors"][i, :ns, 1] * scale,
               marker="^", s=16, c=CT, edgecolors="white", lw=0.4, zorder=5)
    ax.scatter(*(d["xs"][i] * scale), marker="*", s=90, c="white",
               edgecolors=CT, lw=0.9, zorder=6)


def panel(ax, title=None):
    ax.set_xlim(0, 500); ax.set_ylim(0, 500)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)


def region_overlay(ax, P, t_hat, color, ls="-"):
    m = region_mask(P.astype(np.float64), t_hat).reshape(N, N).astype(float)
    cc = (np.arange(N) + 0.5) / N * 500
    ax.contourf(cc, cc, m, levels=[0.5, 1.5], colors=[color], alpha=0.18)
    ax.contour(cc, cc, m, levels=[0.5], colors=[color], linewidths=1.0,
               linestyles=ls)


def plume_field(d, i, ng=96):
    cc = (np.arange(ng) + 0.5) / ng
    X, Y = np.meshgrid(cc, cc)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
    f = plume_ppm_per_kgh(pts, torch.tensor(d["xs"][i])[None],
                          torch.tensor(d["u"][i])[None],
                          torch.tensor([stab_of(d, i)])).numpy() * d["q"][i]
    return f.reshape(ng, ng)


def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig_dir = resolve(cfg, "figures_dir")
    alpha = cfg["conformal"]["alpha"]

    data = {n: load_ch4(data_dir, n) for n in SPLITS}
    P_ex = {n: np.load(data_dir / f"ch4_{n}_posterior.npz")["probs"]
            for n in ("calib", "test")}
    d_t = data["test"]
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, d_t["true_cell"])
    e_area = reg_e["sizes"] / 4096
    models, P_t, ths = {}, {}, {}
    ck_dir = resolve(cfg, "checkpoints_dir")
    for m in ("M0", "M1"):
        mod = DeepSetsLocalizer(cfg).to(device)
        mod.load_state_dict(torch.load(ck_dir / f"ch4_{m.lower()}_s1.pt",
                                       weights_only=False))
        mod.eval()
        models[m] = mod
        P_t[m] = model_probs(mod, d_t, device)
        P_c = model_probs(mod, data["calib"], device)
        r2 = np.random.default_rng(cfg["conformal"]["score_seed"])
        ths[m] = tail_threshold(tail_scores(P_c, data["calib"]["true_cell"],
                                            r2), alpha)
    ratios = {m: np.load(data_dir / f"ch4_audit_{m}.npz")["sizes"]
              / np.maximum(reg_e["sizes"], 1) for m in models}

    # ---- fig1 pipeline (median-identifiable scenario) ----
    ident_idx = np.where(e_area < 0.10)[0]
    i = int(ident_idx[np.argsort(e_area[ident_idx])[len(ident_idx) // 2]])
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9))
    heat(axes[0], plume_field(d_t, i), "Purples", 0.4)
    marks(axes[0], d_t, i); panel(axes[0], "CH$_4$ plume + sensors")
    heat(axes[1], P_t["M0"][i], "Blues"); marks(axes[1], d_t, i)
    panel(axes[1], "Amortized heatmap")
    heat(axes[2], P_t["M0"][i], "Blues")
    region_overlay(axes[2], P_t["M0"][i], ths["M0"], C0)
    marks(axes[2], d_t, i); panel(axes[2], "90% conformal region")
    heat(axes[3], P_ex["test"][i], "Greens"); marks(axes[3], d_t, i)
    panel(axes[3], "Exact posterior (teacher/auditor)")
    for k in range(3):
        fig.add_artist(FancyArrowPatch((0.247 + 0.242 * k, 0.5),
                                       (0.268 + 0.242 * k, 0.5),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9,
                                       color=CM))
    fig.savefig(fig_dir / "fig1_pipeline.pdf"); plt.close(fig)
    print("fig1", flush=True)

    # ---- fig2 identifiable vs weak ----
    weak_idx = np.where(e_area >= 0.5)[0]
    j = int(weak_idx[10])
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.5))
    for row, idx, lab in [(0, i, "Identifiable"), (1, j, "Weakly identifiable")]:
        heat(axes[row, 0], plume_field(d_t, idx), "Purples", 0.4)
        marks(axes[row, 0], d_t, idx); panel(axes[row, 0])
        heat(axes[row, 1], P_ex["test"][idx], "Greens")
        marks(axes[row, 1], d_t, idx); panel(axes[row, 1])
        heat(axes[row, 2], P_t["M1"][idx], "RdPu")
        marks(axes[row, 2], d_t, idx); panel(axes[row, 2])
        region_overlay(axes[row, 3], P_t["M1"][idx], ths["M1"], C1)
        region_overlay(axes[row, 3], P_ex["test"][idx], th_e, CE, ls="--")
        marks(axes[row, 3], d_t, idx); panel(axes[row, 3])
        axes[row, 0].set_ylabel(lab, fontsize=7.5)
    for c, t in enumerate(["Plume + sensors", "Exact posterior",
                           "M1 heatmap", "Both 90% regions"]):
        axes[0, c].set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_easy_vs_ambiguous.pdf"); plt.close(fig)
    print("fig2", flush=True)

    # ---- fig3 coverage sweep ----
    sweep = json.load(open(results_dir / "ch4_extras2.json"))["coverage_sweep"]
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    ax.plot([0.45, 1], [0.45, 1], "--", color=CM, lw=0.8)
    s0 = [r for r in sweep if r["model"] == "M0"]
    ax.plot([r["nominal"] for r in s0], [r["raw_hpd"] for r in s0],
            color=CE, lw=1.8, label="Raw HPD (uncalibrated)")
    ax.plot([r["nominal"] for r in s0], [r["conformal"] for r in s0],
            color=C0, lw=1.8, label="Conformal (calibrated)")
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage (test)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=CG, lw=0.5)
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.45, 1.02)
    fig.savefig(fig_dir / "fig3_coverage.pdf"); plt.close(fig)
    print("fig3", flush=True)

    # ---- fig4 audit scatter + D1 ----
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axes[0]
    lim = [2e-4, 1.3]
    ax.plot(lim, lim, "--", color=CM, lw=0.8)
    for m, col, name in [("M0", C0, "M0: point labels"),
                         ("M1", C1, "M1: physics-distilled")]:
        s = np.load(data_dir / f"ch4_audit_{m}.npz")["sizes"] / 4096
        ax.scatter(e_area, s, s=4, alpha=0.2, c=col, edgecolors="none",
                   label=name, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("exact-posterior region area (site fraction)")
    ax.set_ylabel("learned region area")
    ax.set_title("Sharpness at matched 90% coverage")
    ax.legend(frameon=False, loc="upper left", markerscale=2.5)
    ax.grid(color=CG, lw=0.5)
    ax = axes[1]
    for m, col in [("M0", C0), ("M1", C1)]:
        est = cell_center_of(P_t[m].argmax(1), N)
        err = np.linalg.norm(est - d_t["xs"], axis=1) * 500
        s = np.load(data_dir / f"ch4_audit_{m}.npz")["sizes"] / 4096
        ax.scatter(s, err, s=4, alpha=0.2, c=col, edgecolors="none",
                   rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("learned region area (site fraction)")
    ax.set_ylabel("MAP localization error (m)")
    ax.set_title("Region area predicts error")
    ax.grid(color=CG, lw=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_audit.pdf"); plt.close(fig)
    print("fig4", flush=True)

    # ---- fig5 dose knob1: snapshots + curves ----
    z = np.load(data_dir / "ch4_dose_curves.npz")
    Ns, cM0, cM1, cEx = z["Ns"], z["M0"], z["M1"], z["exact"]
    slopes = cEx[:, -1] - cEx[:, 0]
    flag = int(np.argsort(slopes)[len(slopes) // 10])
    srng, xs_b, q_b, u_b, stab_b, sig_b, master = base_scenario(flag)
    grng = np.random.default_rng(123)
    fig = plt.figure(figsize=(6.8, 4.1))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1], hspace=0.28)
    for c, Nn in enumerate([4, 6, 8, 10, 12]):
        ax = fig.add_subplot(gs[0, c])
        d1, _ = one_scenario(xs_b, q_b, u_b, stab_b, sig_b, master[:Nn], grng)
        P = model_probs(models["M1"], d1, device)[0]
        heat(ax, P, "RdPu")
        region_overlay(ax, P, ths["M1"], C1)
        marks(ax, d1, 0); panel(ax, f"{Nn} masts")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(Ns, cEx.mean(0), color=CE, lw=2, label="Exact oracle")
    ax.plot(Ns, cM0.mean(0), color=C0, lw=2, label="M0")
    ax.plot(Ns, cM1.mean(0), color=C1, lw=2, label="M1")
    ax.plot(Ns, cEx[flag], color=CE, lw=1.1, ls="--", alpha=0.8)
    ax.plot(Ns, cM1[flag], color=C1, lw=1.1, ls="--", alpha=0.8,
            label="Shown site (dashed)")
    ax.set_xlabel("number of sensor masts")
    ax.set_ylabel("90% region area\n(mean over 50 sites)")
    ax.grid(color=CG, lw=0.5)
    ax.legend(frameon=False, ncols=4, loc="upper right")
    fig.savefig(fig_dir / "fig5_doseresponse.pdf"); plt.close(fig)
    print("fig5", flush=True)

    # ---- fig6 gallery ----
    order = np.lexsort((d_t["sigma"], d_t["n_sensors"]))
    picks = order[np.linspace(40, len(order) - 40, 8).astype(int)]
    fig, axes = plt.subplots(2, 4, figsize=(6.8, 3.5))
    for ax, k in zip(axes.ravel(), picks):
        heat(ax, plume_field(d_t, k), "Purples", 0.4)
        marks(ax, d_t, k)
        u = d_t["u"][k] * U_SCALE
        panel(ax, f"{d_t['q'][k]:.0f} kg/h, {np.hypot(*u):.1f} m/s, "
                  f"{'ABCDEF'[stab_of(d_t, k)]}, N={int(d_t['n_sensors'][k])}")
    fig.suptitle("CH$_4$-500 scenarios: plume, sensors, source", fontsize=8,
                 y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_dataset.pdf"); plt.close(fig)
    print("fig6", flush=True)

    # ---- fig7 targets ----
    k = int(ident_idx[np.argsort(e_area[ident_idx])[int(0.8 * len(ident_idx))]])
    tc = torch.tensor(d_t["true_cell"][k:k + 1])
    bump = smoothed_targets(tc, N, 0.75, 3.0, "cpu")[0].numpy()
    raw = P_ex["test"][k]
    temp = blur_teacher(torch.tensor(raw[None].astype(np.float32)), N,
                        0.75)[0].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    for ax, P, ttl, cm in [(axes[0], bump, "M0 target: point label", "Blues"),
                           (axes[1], raw, "Exact posterior (raw)", "Greens"),
                           (axes[2], temp, "Tempered teacher (M1)", "Greens")]:
        heat(ax, P, cm)
        marks(ax, d_t, k); panel(ax, ttl)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_targets.pdf"); plt.close(fig)
    print("fig7", flush=True)

    # ---- fig8 zoo ----
    qs = [0.02, 0.10, 0.25, 0.55, 0.90]
    picks = [int(np.argsort(e_area)[int(q * len(e_area))]) for q in qs]
    fig, axes = plt.subplots(3, 5, figsize=(6.8, 4.3))
    rows = [("Exact posterior", P_ex["test"], "Greens"),
            ("M0 (point labels)", P_t["M0"], "Blues"),
            ("M1 (distilled)", P_t["M1"], "RdPu")]
    for r, (name, P, cm) in enumerate(rows):
        for c, k in enumerate(picks):
            heat(axes[r, c], P[k].astype(np.float64), cm)
            axes[r, c].scatter(*(d_t["xs"][k] * 500), marker="*", s=70,
                               c="white", edgecolors=CT, lw=0.8, zorder=6)
            panel(axes[r, c],
                  f"oracle {e_area[k]:.3f}" if r == 0 else None)
        axes[r, 0].set_ylabel(name, fontsize=7.5)
    fig.suptitle("Posterior shape across identifiability", fontsize=8, y=0.99)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_zoo.pdf"); plt.close(fig)
    print("fig8", flush=True)

    # ---- fig9 stats ----
    d_tr = data["train"]
    snr = []
    for k in range(0, len(d_tr["ids"]), 4):
        ns = int(d_tr["n_sensors"][k])
        r = d_tr["readings"][k, :ns][d_tr["keep"][k, :ns]]
        snr.append(np.abs(r).max() / d_tr["sigma"][k])
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6))
    def hist(ax, x, bins, ttl, log=False, color=C0):
        ax.hist(x, bins=bins, color=color, alpha=0.85)
        if log:
            ax.set_xscale("log")
        ax.set_title(ttl, fontsize=7.5)
        ax.grid(color=CG, lw=0.4)
    hist(axes[0, 0], d_tr["q"], np.logspace(1, np.log10(500), 25),
         "release rate (kg/h)", log=True)
    hist(axes[0, 1], d_tr["sigma"], np.logspace(np.log10(0.2), np.log10(3), 25),
         "sensor noise (ppm)", log=True)
    axes[0, 2].bar(*np.unique(d_tr["n_sensors"], return_counts=True),
                   color=C0, alpha=0.85)
    axes[0, 2].set_title("sensor masts $N$", fontsize=7.5)
    axes[0, 2].grid(color=CG, lw=0.4)
    hist(axes[1, 0], np.clip(snr, 1e-2, None), np.logspace(-2, 4, 30),
         "peak reading SNR", log=True)
    hist(axes[1, 1], e_area, np.logspace(np.log10(2e-4), 0, 30),
         "oracle HPD-90 area (test)", log=True, color=CE)
    err_e = np.linalg.norm(cell_center_of(P_ex["test"].argmax(1), N)
                           - d_t["xs"], axis=1) * 500
    hist(axes[1, 2], np.clip(err_e, 1, None), np.logspace(0, np.log10(500), 30),
         "oracle MAP error (m)", log=True, color=CE)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig9_stats.pdf"); plt.close(fig)
    print("fig9", flush=True)

    # ---- fig10 improvement (identifiable subset) ----
    ident = e_area < 0.10
    r0, r1 = ratios["M0"][ident], ratios["M1"][ident]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    for r, col, name in [(r0, C0, "M0"), (r1, C1, "M1")]:
        xs_ = np.sort(r)
        ax.plot(xs_, np.arange(1, len(xs_) + 1) / len(xs_), color=col, lw=1.8,
                label=f"{name} (median {np.median(r):.1f}$\\times$)")
        ax.axvline(np.median(r), color=col, lw=0.8, ls=":")
    ax.axvline(1.0, color=CM, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("inefficiency (learned / oracle area)")
    ax.set_ylabel("fraction of identifiable scenarios")
    ax.set_title("Identifiable-scenario inefficiency")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=CG, lw=0.4)
    ax = axes[1]
    lim = [min(r0.min(), r1.min()) * 0.8, max(r0.max(), r1.max()) * 1.2]
    ax.plot(lim, lim, "--", color=CM, lw=0.8)
    ax.scatter(r0, r1, s=4, alpha=0.25, c=C1, edgecolors="none",
               rasterized=True)
    ax.annotate(f"M1 sharper on {float((r1 < r0).mean()):.0%}",
                (0.05, 0.92), xycoords="axes fraction", fontsize=7.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("M0 inefficiency"); ax.set_ylabel("M1 inefficiency")
    ax.set_title("Per-scenario improvement")
    ax.grid(color=CG, lw=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_improvement.pdf"); plt.close(fig)
    print("fig10", flush=True)

    # ---- fig11 memorization ----
    rows = list(csv.DictReader(open(results_dir / "ch4_memorization.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for aug, col, name in [("1", C0, "with D4 augmentation"),
                           ("0", "#d03b3b", "without augmentation")]:
        e = [int(r["epoch"]) for r in rows if r["aug"] == aug]
        tl = [float(r["train_loss"]) for r in rows if r["aug"] == aug]
        vn = [float(r["val_nll"]) for r in rows if r["aug"] == aug]
        axes[0].plot(e, tl, color=col, lw=1.6, label=name)
        axes[1].plot(e, vn, color=col, lw=1.6)
    axes[1].axhline(np.log(4096), color=CM, lw=0.8, ls="--")
    axes[0].set_ylabel("training loss"); axes[1].set_ylabel("validation NLL")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(color=CG, lw=0.4)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_memorization.pdf"); plt.close(fig)
    print("fig11", flush=True)

    # ---- fig12 knobs 2-3 ----
    z23 = np.load(data_dir / "ch4_dose23.npz")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    sig = z23["sig_vals"]
    ax.plot(sig, z23["k3_ex"].mean(0), color=CE, lw=1.8, label="Exact")
    ax.plot(sig, z23["k3_M0"].mean(0), color=C0, lw=1.8, label="M0")
    ax.plot(sig, z23["k3_M1"].mean(0), color=C1, lw=1.8, label="M1")
    ax.set_xscale("log")
    ax.set_xlabel("sensor noise (ppm)")
    ax.set_ylabel("mean 90% region area")
    ax.set_title("Noise response")
    ax.legend(frameon=False)
    ax.grid(color=CG, lw=0.4)
    ax = axes[1]
    lim = [2e-4, 1.3]
    ax.plot(lim, lim, "--", color=CM, lw=0.8)
    ax.scatter(z23["spread_ex"], z23["conf_ex"], s=14, c=CE, alpha=0.6,
               edgecolors="none", label="Exact")
    ax.scatter(z23["spread_M1"], z23["conf_M1"], s=14, c=C1, alpha=0.6,
               edgecolors="none", label="M1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("area, well-spread masts")
    ax.set_ylabel("area, confined masts")
    ax.set_title("Geometry response")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=CG, lw=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_knobs.pdf"); plt.close(fig)
    print("fig12", flush=True)

    # ---- fig13 scaling ----
    rows = json.load(open(results_dir / "ch4_extras.json"))["scaling"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for mode, col, name in [("M0", C0, "M0 (point labels)"),
                            ("M1", C1, "M1 (distilled)")]:
        xs_ = [r["size"] for r in rows if r["mode"] == mode]
        ys = [r["ineff_median_identifiable"] for r in rows if r["mode"] == mode]
        ax.plot(xs_, ys, "o-", color=col, lw=1.8, ms=4, label=name)
    ax.axhline(1.0, color=CM, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2500, 10000, 40000])
    ax.set_xticklabels(["2.5k", "10k", "40k"])
    ax.set_xlabel("training scenarios")
    ax.set_ylabel("median identifiable-scenario inefficiency")
    ax.legend(frameon=False)
    ax.grid(color=CG, lw=0.5, which="both")
    fig.savefig(fig_dir / "fig13_scaling.pdf"); plt.close(fig)
    print("fig13", flush=True)

    # ---- fig14 stratified ----
    strata = json.load(open(results_dir / "ch4_extras2.json"))["strata"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    deciles = np.quantile(e_area, np.linspace(0, 1, 11))
    xs_, y0, y1 = [], [], []
    for k in range(10):
        m = (e_area >= deciles[k]) & (e_area <= deciles[k + 1])
        xs_.append(np.median(e_area[m]))
        y0.append(np.median(ratios["M0"][m]))
        y1.append(np.median(ratios["M1"][m]))
    ax.plot(xs_, y0, "o-", color=C0, lw=1.6, ms=3.5, label="M0")
    ax.plot(xs_, y1, "o-", color=C1, lw=1.6, ms=3.5, label="M1")
    ax.axhline(1.0, color=CM, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle area (difficulty decile median)")
    ax.set_ylabel("median inefficiency")
    ax.set_title("Where the gap lives")
    ax.legend(frameon=False)
    ax.grid(color=CG, lw=0.5, which="both")
    ax = axes[1]
    names = list(strata.keys())
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, [strata[n]["cov_M0"] for n in names], 0.36,
           color=C0, label="M0")
    ax.bar(xpos + 0.18, [strata[n]["cov_M1"] for n in names], 0.36,
           color=C1, label="M1")
    ax.axhline(0.90, color=CM, lw=0.8, ls="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=6)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("conformal coverage")
    ax.set_title("Conditional coverage by stratum")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=CG, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig14_stratified.pdf"); plt.close(fig)
    print("fig14", flush=True)

    # ---- mode recovery rerun at sigma = 1e-6 ppm ----
    hits, tried, attempts = 0, 0, 0
    mrng = np.random.default_rng(11)
    while tried < 24 and attempts < 500:
        attempts += 1
        srng, xs_b, q_b, u_b, stab_b, sig_b, master = base_scenario(2000 + attempts)
        cell = pos_to_cell(xs_b[None], N)[0]
        xs_c = np.array([(cell % N + 0.5) / N, (cell // N + 0.5) / N])
        g = plume_ppm_per_kgh(torch.tensor(master[:8]),
                              torch.tensor(xs_c)[None],
                              torch.tensor(u_b)[None],
                              torch.tensor([stab_b])).numpy()
        if (q_b * g).max() < 1.0:
            continue
        tried += 1
        d1, _ = one_scenario(xs_c, q_b, u_b, stab_b, 1e-6, master[:8], mrng)
        pe, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
        hits += int(pe[0].argmax() == cell)
    print(f"mode recovery @1e-6 ppm: {hits}/{tried}", flush=True)
    update_json(resolve(cfg, "results_dir") / "gates.json",
                {"CH4_extras2": {"mode_recovery_1e6": f"{hits}/{tried}"}})


if __name__ == "__main__":
    main()
