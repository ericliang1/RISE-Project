"""CH4-T architecture comparison figures: DeepSets vs GNN vs Set Transformer.

Left  : learned-vs-exact region area at matched 90% coverage (seed 1), the four
        distilled-target models M1/M2/M3 + baseline M0, with the Bayes star.
Right : median 90% region radius across three seeds for the two new
        architectures (GNN M2, Set Transformer M3), against the DeepSets M0/M1
        baselines and the 7 m Bayes / 50 m EPA lines.
figT2 : one scenario to scale, each model's median region as a circle.

Reads only saved audit .npz files -- no training, no data regen.
-> figures/figT1_st_audit.{pdf,png}, figT2_st_example.{pdf,png}
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from common import load_config, resolve
from methane_model import L_SITE as L, plume_ppm_per_kgh
from methane_t import N_GRID, T, stab_of

# paper palette + GNN purple + Set Transformer orange
C_M0, C_M1, C_EXACT, C_MUT, C_GRID = "#2a78d6", "#e87ba4", "#008300", "#52514e", "#e3e2df"
C_M2, C_M3 = "#5a3fa0", "#e2792b"
plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight"})

CELLS = N_GRID * N_GRID
radius_m = lambda s: np.sqrt((s / CELLS) / np.pi) * L
area_frac = lambda s: s / CELLS
SEEDS = [1, 2, 3]


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    fd = resolve(cfg, "figures_dir")
    SRC = "/projectnb/rise-tower/eric1/csr-data"

    m0 = np.load(f"{SRC}/ch4t_audit_M0.npz")
    m1 = np.load(f"{SRC}/ch4t_audit_M1.npz")
    m2 = {s: np.load(dd / f"ch4t_audit_M2_seed{s}.npz") for s in SEEDS}
    m3 = {s: np.load(dd / f"ch4t_audit_M3_seed{s}.npz") for s in SEEDS}
    exact = m1["exact_sizes"]
    r_m0 = float(np.median(radius_m(m0["sizes"])))
    r_m1 = float(np.median(radius_m(m1["sizes"])))
    r_bayes = float(np.median(radius_m(exact)))
    r2 = [np.median(radius_m(m2[s]["sizes"])) for s in SEEDS]
    r3 = [np.median(radius_m(m3[s]["sizes"])) for s in SEEDS]

    # ================= figT1 =================
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    lim = [8e-4, 1.3]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8, zorder=1)
    e = area_frac(exact)
    for z, col, lab in [
        (m0["sizes"], C_M0, f"M0 DeepSets: {r_m0:.0f} m"),
        (m1["sizes"], C_M1, f"M1 DeepSets+distill: {r_m1:.0f} m"),
        (m3[1]["sizes"], C_M3, f"M3 SetTransformer: {r3[0]:.0f} m"),
        (m2[1]["sizes"], C_M2, f"M2 GNN: {r2[0]:.0f} m"),
    ]:
        ax.scatter(e, area_frac(z), s=4, alpha=0.16, c=col, edgecolors="none", zorder=2)
        ax.scatter([], [], s=18, c=col, label=lab)
    ax.scatter([np.median(e)], [np.median(e)], marker="*", s=80, c=C_EXACT,
               zorder=6, edgecolors="white", lw=0.4, label=f"Bayes limit: {r_bayes:.0f} m")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="exact-posterior region area (frac.)",
           ylabel="learned region area (frac.)",
           title="Sharpness at matched 90% coverage")
    ax.grid(color=C_GRID, lw=0.5)
    ax.legend(fontsize=6.0, loc="lower right", framealpha=0.9)

    ax = axes[1]
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.2, r3, 0.4, color=C_M3, label="M3 SetTransformer (distill)")
    ax.bar(x + 0.2, r2, 0.4, color=C_M2, label="M2 GNN (distill)")
    ax.axhline(r_m0, color=C_M0, lw=1.1, ls="--", label=f"M0 DeepSets ({r_m0:.0f} m)")
    ax.axhline(r_m1, color=C_M1, lw=1.1, ls="--", label=f"M1 DeepSets+distill ({r_m1:.0f} m)")
    ax.axhline(50, color=C_MUT, lw=0.9, ls=":", label="EPA 50 m radius")
    ax.axhline(r_bayes, color=C_EXACT, lw=1.2, ls="--", label=f"Bayes limit ({r_bayes:.0f} m)")
    ax.set(xticks=x, xticklabels=[f"seed {s}" for s in SEEDS], ylim=(0, 100),
           ylabel="median 90% region radius (m)", title="Seed stability")
    ax.grid(color=C_GRID, lw=0.5, axis="y")
    ax.legend(fontsize=5.8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fd / f"figT1_st_audit.{ext}")
    plt.close(fig)
    print(f"saved {fd}/figT1_st_audit.pdf (+.png)")

    # ================= figT2: one scenario to scale =================
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    i = int(np.argsort(exact)[len(exact) // 20])
    ns = int(d["n_sensors"][i]); stab = stab_of(d, i)
    ng = 160
    cc = (np.arange(ng) + 0.5) / ng
    X, Y = np.meshgrid(cc, cc)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
    field = np.zeros(ng * ng)
    for k in range(T):
        u = torch.tensor(d["u_seq"][i, k])[None]
        field += plume_ppm_per_kgh(pts, torch.tensor(d["xs"][i])[None],
                                   u.expand(ng * ng, 2),
                                   torch.tensor([stab]).expand(ng * ng)).numpy()
    field *= d["q"][i] / T

    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    ax.imshow(field.reshape(ng, ng), origin="lower", extent=[0, L, 0, L],
              cmap="Purples", norm=PowerNorm(0.4))
    ax.scatter(d["sensors"][i, :ns, 0] * L, d["sensors"][i, :ns, 1] * L,
               marker="^", s=26, c="#0b0b0b", edgecolors="white", lw=0.4, zorder=5)
    sx, sy = d["xs"][i, 0] * L, d["xs"][i, 1] * L
    for k in range(0, T, 5):
        u = d["u_seq"][i, k] * 8.0
        ax.annotate("", xy=(sx + u[0] * 11, sy + u[1] * 11), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#d55181", lw=1, alpha=0.6))
    for radius, col in [(r_m0, C_M0), (r_m1, C_M1), (r3[0], C_M3),
                        (r2[0], C_M2), (r_bayes, C_EXACT)]:
        ax.add_patch(Circle((sx, sy), radius, fill=False, ec=col, lw=1.7, zorder=4))
    ax.scatter(sx, sy, marker="*", s=120, c="white", edgecolors="#0b0b0b", lw=0.9, zorder=6)
    ax.set(xlim=(0, L), ylim=(0, L), xlabel="m", ylabel="m")
    ax.set_title(f"Median 90% search region, drawn to scale\n"
                 f"({d['q'][i]:.0f} kg/h, class {'ABCDEF'[stab]}, {ns} masts)", fontsize=7.5)
    leg = [Line2D([], [], color=C_EXACT, lw=1.7, label=f"Bayes limit {r_bayes:.0f} m"),
           Line2D([], [], color=C_M2, lw=1.7, label=f"M2 GNN {r2[0]:.0f} m"),
           Line2D([], [], color=C_M3, lw=1.7, label=f"M3 SetTransformer {r3[0]:.0f} m"),
           Line2D([], [], color=C_M1, lw=1.7, label=f"M1 DeepSets+distill {r_m1:.0f} m"),
           Line2D([], [], color=C_M0, lw=1.7, label=f"M0 DeepSets {r_m0:.0f} m")]
    ax.legend(handles=leg, frameon=True, framealpha=0.9, loc="lower left", fontsize=6.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fd / f"figT2_st_example.{ext}")
    plt.close(fig)
    print(f"saved {fd}/figT2_st_example.pdf (+.png)")


if __name__ == "__main__":
    main()
