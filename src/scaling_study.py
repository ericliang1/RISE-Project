"""Data-scaling study: median inefficiency vs training-set size, M0 vs M1.

Trains both supervision modes at sizes {2500, 5000, 10000, 40000} (the 40k
point uses train + train_xl, both from the same prior; the frozen benchmark
splits are untouched), evaluates each with the identical conformal machinery,
and reports coverage + inefficiency.  Produces results/scaling_study.json and
figures/fig13_scaling.pdf.

Usage:  python src/scaling_study.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from inference import heatmaps, load_model, load_split
from train import train_one_seed

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

SIZES = [2500, 5000, 10000, 40000]


def evaluate(stem, cfg, device, d_calib, d_test, exact_sizes, alpha):
    ckpt = resolve(cfg, "checkpoints_dir") / f"{stem}.pt"
    model, _ = load_model(cfg, ckpt, device)
    P_c = heatmaps(model, d_calib, device).astype(np.float64)
    P_t = heatmaps(model, d_test, device).astype(np.float64)
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    t_hat = tail_threshold(tail_scores(P_c, d_calib["true_cell"], rng), alpha)
    reg = regions(P_t, t_hat, d_test["true_cell"])
    n_cells = P_t.shape[1]
    ratio = (reg["sizes"] / n_cells) / (exact_sizes / n_cells)
    return {
        "coverage": float(reg["covered"].mean()),
        "area_median": float(np.median(reg["sizes"] / n_cells)),
        "ineff_median": float(np.median(ratio)),
        "ineff_mean": float(ratio.mean()),
    }


def main():
    cfg = load_config()
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig_dir = resolve(cfg, "figures_dir")
    alpha = cfg["conformal"]["alpha"]
    d_calib = load_split(data_dir, "calib")
    d_test = load_split(data_dir, "test")
    exact_sizes = np.load(data_dir / "regions_exact_test.npz")["sizes"]

    rows = []
    for size in SIZES:
        use_xl = size > 10000
        for mode, distill in [("M0", False), ("M1", True)]:
            stem = f"scale_{mode.lower()}_{size}"
            print(f"=== {mode} @ {size} ===", flush=True)
            if not (resolve(cfg, "checkpoints_dir") / f"{stem}.pt").exists():
                if distill:
                    cfg["distill"]["mix_lambda"] = 1.0
                    cfg["distill"]["teacher_blur_std_cells"] = 0.75
                train_one_seed(cfg, 1, device, distill=distill,
                               train_size=size, use_xl=use_xl,
                               stem_override=stem)
            r = evaluate(stem, cfg, device, d_calib, d_test, exact_sizes, alpha)
            r.update(size=size, mode=mode)
            rows.append(r)
            print(r, flush=True)

    with open(results_dir / "scaling_study.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for mode, col, name in [("M0", C_BASE, "M0 (point labels)"),
                            ("M1", C_IMPR, "M1 (distilled)")]:
        xs = [r["size"] for r in rows if r["mode"] == mode]
        ys = [r["ineff_median"] for r in rows if r["mode"] == mode]
        ax.plot(xs, ys, "o-", color=col, lw=1.8, ms=4, label=name)
        for x, y, r in zip(xs, ys, [r for r in rows if r["mode"] == mode]):
            ax.annotate(f'{r["coverage"]:.2f}', (x, y), fontsize=5.5,
                        color=C_MUT, textcoords="offset points", xytext=(3, 4))
    ax.axhline(1.0, color=C_MUT, lw=0.8, ls="--")
    ax.annotate("oracle parity", (SIZES[0], 1.05), fontsize=6.5, color=C_MUT)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{s//1000}k" for s in SIZES])
    ax.set_xlabel("training scenarios")
    ax.set_ylabel("median inefficiency (learned / oracle area)")
    ax.legend(frameon=False)
    ax.grid(color=C_GRID, lw=0.5, which="both")
    fig.savefig(fig_dir / "fig13_scaling.pdf")
    plt.close(fig)
    print("fig13 done")


if __name__ == "__main__":
    main()
