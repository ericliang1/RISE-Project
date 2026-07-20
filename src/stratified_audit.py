"""Stratified audit + significance tests (pure analysis of existing artifacts).

Strata: sensor count (4-6 / 7-9 / 10-12), noise terciles, oracle-difficulty
deciles.  Per stratum: conformal coverage and median inefficiency for M0/M1.
Significance: Wilcoxon signed-rank on paired per-scenario inefficiency ratios
and bootstrap CIs on median inefficiency.  Outputs
results/stratified_audit.json, results/significance.json, and
figures/fig14_stratified.pdf.

Usage:  python src/stratified_audit.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from common import load_config, resolve
from inference import load_split

C_BASE = "#2a78d6"
C_IMPR = "#e87ba4"
C_MUT = "#52514e"
C_GRID = "#e3e2df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def boot_ci_median(x, n=20000, seed=0, conf=0.95):
    rng = np.random.default_rng(seed)
    meds = np.median(
        x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    lo, hi = np.quantile(meds, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(np.median(x)), float(lo), float(hi)


def main():
    cfg = load_config()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig_dir = resolve(cfg, "figures_dir")
    d = load_split(data_dir, "test")
    z0 = np.load(data_dir / "audit_scatter_model_seed1.npz")
    z1 = np.load(data_dir / "audit_scatter_model2_seed1.npz")
    r0 = z0["area_learned"] / z0["area_exact"]
    r1 = z1["area_learned"] / z1["area_exact"]
    cov0 = np.load(data_dir / "regions_model_seed1_test.npz")["sizes"]  # sizes
    # coverage indicators recomputed from regions npz is not stored; use tails:
    # simpler — recompute covered flags from stored region sizes and heatmaps
    # is heavy; instead use per-stratum coverage from the stored covered masks
    # via conformal regions npz absence -> recompute cheaply from heatmaps:
    from conformal import regions, tail_scores, tail_threshold
    alpha = cfg["conformal"]["alpha"]
    d_calib = load_split(data_dir, "calib")
    cov = {}
    for tag, key in [("model_seed1", "M0"), ("model2_seed1", "M1")]:
        P_c = np.load(data_dir / f"heatmaps_{tag}_calib.npz")["probs"].astype(np.float64)
        P_t = np.load(data_dir / f"heatmaps_{tag}_test.npz")["probs"].astype(np.float64)
        rng = np.random.default_rng(cfg["conformal"]["score_seed"])
        t_hat = tail_threshold(tail_scores(P_c, d_calib["true_cell"], rng), alpha)
        cov[key] = regions(P_t, t_hat, d_test_true := d["true_cell"])["covered"]

    # ---------------- strata definitions ---------------------------------
    strata = {}
    N = d["n_sensors"]
    strata["N 4-6"] = N <= 6
    strata["N 7-9"] = (N >= 7) & (N <= 9)
    strata["N 10-12"] = N >= 10
    s_t = np.quantile(d["sigma"], [1 / 3, 2 / 3])
    strata["low noise"] = d["sigma"] <= s_t[0]
    strata["mid noise"] = (d["sigma"] > s_t[0]) & (d["sigma"] <= s_t[1])
    strata["high noise"] = d["sigma"] > s_t[1]

    table = {}
    for name, m in strata.items():
        table[name] = {
            "n": int(m.sum()),
            "coverage_M0": float(cov["M0"][m].mean()),
            "coverage_M1": float(cov["M1"][m].mean()),
            "ineff_med_M0": float(np.median(r0[m])),
            "ineff_med_M1": float(np.median(r1[m])),
        }

    # difficulty deciles by oracle area
    ae = z0["area_exact"]
    deciles = np.quantile(ae, np.linspace(0, 1, 11))
    dec_rows = []
    for i in range(10):
        m = (ae >= deciles[i]) & (ae <= deciles[i + 1])
        dec_rows.append({
            "decile": i + 1, "oracle_area_median": float(np.median(ae[m])),
            "ineff_med_M0": float(np.median(r0[m])),
            "ineff_med_M1": float(np.median(r1[m])),
            "coverage_M0": float(cov["M0"][m].mean()),
            "coverage_M1": float(cov["M1"][m].mean()),
        })

    with open(results_dir / "stratified_audit.json", "w") as f:
        json.dump({"strata": table, "difficulty_deciles": dec_rows}, f, indent=2)

    # ---------------- significance ---------------------------------------
    w = wilcoxon(np.log(r0), np.log(r1), alternative="greater")
    m0 = boot_ci_median(r0)
    m1 = boot_ci_median(r1)
    diff = boot_ci_median(np.log(r0 / r1))
    sig = {
        "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "test": "one-sided Wilcoxon signed-rank on paired log inefficiency "
                "ratios (H1: M0 > M1)",
        "ineff_median_M0_ci95": m0, "ineff_median_M1_ci95": m1,
        "median_log_ratio_improvement_ci95": diff,
        "frac_scenarios_M1_sharper": float((r1 < r0).mean()),
    }
    with open(results_dir / "significance.json", "w") as f:
        json.dump(sig, f, indent=2)
    print(json.dumps(sig, indent=2))

    # ---------------- figure ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    x = [row["oracle_area_median"] for row in dec_rows]
    ax.plot(x, [row["ineff_med_M0"] for row in dec_rows], "o-", color=C_BASE,
            lw=1.6, ms=3.5, label="M0")
    ax.plot(x, [row["ineff_med_M1"] for row in dec_rows], "o-", color=C_IMPR,
            lw=1.6, ms=3.5, label="M1")
    ax.axhline(1.0, color=C_MUT, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle region area (difficulty decile median)")
    ax.set_ylabel("median inefficiency")
    ax.set_title("Where the gap lives")
    ax.legend(frameon=False)
    ax.grid(color=C_GRID, lw=0.5, which="both")

    ax = axes[1]
    names = list(strata.keys())
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.18, [table[n]["coverage_M0"] for n in names], 0.36,
           color=C_BASE, label="M0")
    ax.bar(xpos + 0.18, [table[n]["coverage_M1"] for n in names], 0.36,
           color=C_IMPR, label="M1")
    ax.axhline(0.90, color=C_MUT, lw=0.8, ls="--")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("conformal coverage")
    ax.set_title("Conditional coverage by stratum")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=C_GRID, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig14_stratified.pdf")
    plt.close(fig)
    print("fig14 done")


if __name__ == "__main__":
    main()
