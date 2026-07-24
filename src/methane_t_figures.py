"""CH4-T figures: audit scatter (radii), a wind-meander plume example, and
seed-stability bars.  Uses saved audit sizes + exact posteriors + campaign json
(no model checkpoints needed).  -> figures/figT1_audit.pdf, figT2_example.pdf
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm

from common import load_config, resolve
from methane_model import plume_ppm_per_kgh
from methane_t import N_GRID, T, stab_of

L = 500.0
C_BASE, C_IMPR, C_EXACT, C_MUT, C_GRID = "#2a78d6", "#e87ba4", "#008300", "#52514e", "#e3e2df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    fd = resolve(cfg, "figures_dir")
    rr = resolve(cfg, "results_dir")
    z0 = np.load(dd / "ch4t_audit_M0.npz")
    z1 = np.load(dd / "ch4t_audit_M1.npz")
    e = z0["exact_sizes"] / 4096
    a0 = z0["sizes"] / 4096
    a1 = z1["sizes"] / 4096
    camp = json.load(open(rr / "ch4t_campaign.json"))

    # ---- figT1: audit scatter (2 panels: area-ratio + seed radii) --------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axes[0]
    lim = [3e-4, 1.3]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8)
    for a, col, nm in [(a0, C_BASE, "M0 (standard)"), (a1, C_IMPR, "M1 (distilled)")]:
        ax.scatter(e, a, s=4, alpha=0.2, c=col, edgecolors="none",
                   rasterized=True,
                   label=f"{nm}: {np.sqrt(np.median(a)/np.pi)*L:.0f} m")
    ax.scatter([np.median(e)], [np.median(e)], marker="*", s=60, c=C_EXACT,
               zorder=5, label=f"Bayes limit: {np.sqrt(np.median(e)/np.pi)*L:.0f} m")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("exact-posterior region area")
    ax.set_ylabel("learned region area")
    ax.set_title("CH4-T: sharpness at matched 90% coverage")
    ax.legend(frameon=False, loc="lower right", markerscale=2)
    ax.grid(color=C_GRID, lw=0.5)

    ax = axes[1]
    seeds = [1, 2, 3]
    m0 = [s["radius_m"] for s in camp["seeds"] if s["mode"] == "M0"]
    m1 = [s["radius_m"] for s in camp["seeds"] if s["mode"] == "M1"]
    x = np.arange(3)
    ax.bar(x - 0.2, m0, 0.4, color=C_BASE, label="M0")
    ax.bar(x + 0.2, m1, 0.4, color=C_IMPR, label="M1")
    ax.axhline(camp["oracle_radius_m"], color=C_EXACT, lw=1.2, ls="--",
               label=f"Bayes limit ({camp['oracle_radius_m']:.0f} m)")
    ax.axhline(50, color=C_MUT, lw=0.8, ls=":", label="EPA 50 m radius")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("median 90% region radius (m)")
    ax.set_title("Seed stability")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(color=C_GRID, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(fd / "figT1_audit.pdf")
    plt.close(fig)
    print("figT1 done", flush=True)

    # ---- figT2: wind-meander plume example -------------------------------
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    P_ex = np.load(dd / "ch4t_test_posterior.npz")["probs"]
    # pick an identifiable, well-sensed scenario
    sharp = np.argsort(z0["exact_sizes"])
    i = int(sharp[len(sharp) // 20])
    ng = 160
    cc = (np.arange(ng) + 0.5) / ng
    X, Y = np.meshgrid(cc, cc)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
    ns = int(d["n_sensors"][i])
    stab = stab_of(d, i)
    # time-integrated plume envelope (sum over the wind sequence)
    field = np.zeros(ng * ng)
    for k in range(T):
        u = torch.tensor(d["u_seq"][i, k])[None]
        field += plume_ppm_per_kgh(pts, torch.tensor(d["xs"][i])[None],
                                   u.expand(ng * ng, 2),
                                   torch.tensor([stab]).expand(ng * ng)).numpy()
    field *= d["q"][i] / T

    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(4.0, 3.7))
    ax.imshow(field.reshape(ng, ng), origin="lower", extent=[0, L, 0, L],
              cmap="Purples", norm=PowerNorm(0.4))
    ax.scatter(d["sensors"][i, :ns, 0] * L, d["sensors"][i, :ns, 1] * L,
               marker="^", s=26, c="#0b0b0b", edgecolors="white", lw=0.4, zorder=5)
    sx, sy = d["xs"][i, 0] * L, d["xs"][i, 1] * L
    for k in range(0, T, 5):
        u = d["u_seq"][i, k] * 8.0
        ax.annotate("", xy=(sx + u[0] * 11, sy + u[1] * 11), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#d55181", lw=1,
                                    alpha=0.6))
    # median 90% regions as equivalent-radius circles, drawn to scale
    for radius, col in [(camp["oracle_radius_m"], C_EXACT),
                        (46.0, C_IMPR),
                        ([s["radius_m"] for s in camp["seeds"]
                          if s["mode"] == "M0"][0], C_BASE)]:
        ax.add_patch(Circle((sx, sy), radius, fill=False, ec=col, lw=1.6,
                            zorder=4))
    ax.add_patch(Circle((sx, sy), 50.0, fill=False, ec=C_MUT, lw=0.9,
                        ls=":", zorder=4))
    ax.scatter(sx, sy, marker="*", s=120, c="white", edgecolors="#0b0b0b",
               lw=0.9, zorder=6)
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(f"Median 90% search region, drawn to scale\n"
                 f"({d['q'][i]:.0f} kg/h, class {'ABCDEF'[stab]}, "
                 f"{ns} masts)", fontsize=7.5)
    leg = [Line2D([], [], color=C_EXACT, lw=1.6,
                  label=f"Bayes limit {camp['oracle_radius_m']:.0f} m"),
           Line2D([], [], color=C_IMPR, lw=1.6, label="full recipe 44\u201349 m"),
           Line2D([], [], color=C_BASE, lw=1.6, label="labels only 94 m"),
           Line2D([], [], color=C_MUT, lw=0.9, ls=":", label="EPA 50 m")]
    ax.legend(handles=leg, frameon=True, framealpha=0.9, loc="lower left",
              fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fd / "figT2_example.pdf")
    plt.close(fig)
    print("figT2 done", flush=True)


if __name__ == "__main__":
    main()
