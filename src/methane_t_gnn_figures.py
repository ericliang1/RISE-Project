"""CH4-T GNN audit: three-seed table + paper-style figure (M0/M1/M2base/M2).

Left panel : learned-vs-exact region AREA at matched 90% coverage (seed 1),
             all four models, with the Bayes-limit star.
Right panel: median 90% region radius across three seeds for the two GNN
             models, against the DeepSets M0/M1 baselines and the 7 m Bayes /
             50 m EPA reference lines.

Reuses the paper's figure conventions (methane_t_figures.py): same palette for
M0 (blue) and M1 (pink), Bayes green; the new relational models get a purple
family.  Reads only saved audit .npz files -- no training, no data regen.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_config, resolve
from methane_model import L_SITE as L
from methane_t import N_GRID

# paper palette + two new purples for the GNN (relational) models
C_M0, C_M1, C_EXACT, C_MUT, C_GRID = "#2a78d6", "#e87ba4", "#008300", "#52514e", "#e3e2df"
C_M2base, C_M2 = "#9a86c4", "#5a3fa0"
plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight"})

CELLS = N_GRID * N_GRID
radius_m = lambda sizes: np.sqrt(((sizes / CELLS)) / np.pi) * L
area_frac = lambda sizes: sizes / CELLS


def main():
    cfg = load_config()
    data_dir = resolve(cfg, "data_dir")
    fig_dir = resolve(cfg, "figures_dir")
    res_dir = resolve(cfg, "results_dir")
    SRC = "/projectnb/rise-tower/eric1/csr-data"  # frozen DeepSets M0/M1 audits

    # ---- load audits ----
    m0 = np.load(f"{SRC}/ch4t_audit_M0.npz")       # DeepSets one-hot (seed 1)
    m1 = np.load(f"{SRC}/ch4t_audit_M1.npz")       # DeepSets distill (seed 1)
    seeds = [1, 2, 3]
    m2b = {s: np.load(data_dir / f"ch4t_audit_M2base_seed{s}.npz") for s in seeds}
    m2 = {s: np.load(data_dir / f"ch4t_audit_M2_seed{s}.npz") for s in seeds}
    exact = m1["exact_sizes"]                       # shared reference

    # ---- three-seed table ----
    def rng_str(dct):
        r = [np.median(radius_m(dct[s]["sizes"])) for s in seeds]
        return min(r), max(r), r
    b_lo, b_hi, b_all = rng_str(m2b)
    d_lo, d_hi, d_all = rng_str(m2)
    table = {
        "bayes_median_radius_m": float(np.median(radius_m(exact))),
        "M0_DeepSets_onehot_m": float(np.median(radius_m(m0["sizes"]))),
        "M1_DeepSets_distill_m": float(np.median(radius_m(m1["sizes"]))),
        "M2base_GNN_onehot_m": {"per_seed": [round(x, 1) for x in b_all],
                                 "range": [round(b_lo, 1), round(b_hi, 1)]},
        "M2_GNN_distill_m": {"per_seed": [round(x, 1) for x in d_all],
                              "range": [round(d_lo, 1), round(d_hi, 1)]},
    }
    print("=== CH4-T median 90% region radius (m) ===")
    print(f"  Bayes limit           {table['bayes_median_radius_m']:5.1f}")
    print(f"  M0  DeepSets one-hot  {table['M0_DeepSets_onehot_m']:5.1f}")
    print(f"  M1  DeepSets distill  {table['M1_DeepSets_distill_m']:5.1f}")
    print(f"  M2base GNN one-hot    {b_lo:5.1f}-{b_hi:.1f}  seeds {[round(x,1) for x in b_all]}")
    print(f"  M2  GNN distill       {d_lo:5.1f}-{d_hi:.1f}  seeds {[round(x,1) for x in d_all]}")
    print(f"  M0/M1 never overlap M2: "
          f"{max(d_all) < table['M1_DeepSets_distill_m']}")
    with open(res_dir / "ch4t_gnn_table.json", "w") as f:
        json.dump(table, f, indent=2)

    # ================= figure =================
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # ---- left: learned vs exact area (seed 1) ----
    ax = axes[0]
    lim = [8e-4, 1.3]
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8, zorder=1)
    e = area_frac(exact)
    for sizes, col, lab in [
        (m0["sizes"], C_M0, f"M0 DeepSets: {np.median(radius_m(m0['sizes'])):.0f} m"),
        (m1["sizes"], C_M1, f"M1 DeepSets+distill: {np.median(radius_m(m1['sizes'])):.0f} m"),
        (m2b[1]["sizes"], C_M2base, f"M2base GNN: {np.median(radius_m(m2b[1]['sizes'])):.0f} m"),
        (m2[1]["sizes"], C_M2, f"M2 GNN+distill: {np.median(radius_m(m2[1]['sizes'])):.0f} m"),
    ]:
        ax.scatter(e, area_frac(sizes), s=4, alpha=0.18, c=col,
                   edgecolors="none", zorder=2)
        ax.scatter([], [], s=18, c=col, label=lab)          # legend proxy
    ax.scatter([np.median(e)], [np.median(e)], marker="*", s=80, c=C_EXACT,
               zorder=6, edgecolors="white", lw=0.4,
               label=f"Bayes limit: {radius_m(np.median(exact)):.0f} m")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="exact-posterior region area (frac.)",
           ylabel="learned region area (frac.)",
           title="Sharpness at matched 90% coverage")
    ax.grid(color=C_GRID, lw=0.5)
    ax.legend(fontsize=6.0, loc="lower right", framealpha=0.9)

    # ---- right: seed-stability bars (GNN models) vs baselines ----
    ax = axes[1]
    x = np.arange(len(seeds))
    ax.bar(x - 0.2, b_all, 0.4, color=C_M2base, label="M2base GNN (one-hot)")
    ax.bar(x + 0.2, d_all, 0.4, color=C_M2, label="M2 GNN (distill)")
    ax.axhline(table["M0_DeepSets_onehot_m"], color=C_M0, lw=1.1, ls="--",
               label=f"M0 DeepSets ({table['M0_DeepSets_onehot_m']:.0f} m)")
    ax.axhline(table["M1_DeepSets_distill_m"], color=C_M1, lw=1.1, ls="--",
               label=f"M1 DeepSets+distill ({table['M1_DeepSets_distill_m']:.0f} m)")
    ax.axhline(50, color=C_MUT, lw=0.9, ls=":", label="EPA 50 m radius")
    ax.axhline(table["bayes_median_radius_m"], color=C_EXACT, lw=1.2, ls="--",
               label=f"Bayes limit ({table['bayes_median_radius_m']:.0f} m)")
    ax.set(xticks=x, xticklabels=[f"seed {s}" for s in seeds], ylim=(0, 100),
           ylabel="median 90% region radius (m)", title="Seed stability")
    ax.grid(color=C_GRID, lw=0.5, axis="y")
    ax.legend(fontsize=6.0, loc="upper right", ncol=1, framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"figT1_gnn_audit.{ext}")
    print(f"saved {fig_dir}/figT1_gnn_audit.pdf (+.png)")


if __name__ == "__main__":
    main()
