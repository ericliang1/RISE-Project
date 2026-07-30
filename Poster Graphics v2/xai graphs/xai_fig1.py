"""XAI Figure 1 — physics-to-region explanation panel, from the REAL model.

Loads the retrained paper checkpoint (PhysHeadNet = DeepSetsT + zero-init
conv head, views=noisy, maps=ensr, seed=1), rebuilds the exact test/calib
views with make_view, recomputes softmax heatmaps and the split-conformal
threshold with the repo's own conformal.py, and cross-checks the resulting
regions against the frozen audit before drawing anything.

Layout: top row = the four physics feature maps exactly as fed to the
network; bottom = the model's softmax probability map with the calibrated
90% region, masts, true source, argmax, and a scale bar.

Usage (from repo root):
  python "Poster Graphics v2/xai graphs/xai_fig1.py"
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from common import load_config, get_device, resolve            # noqa: E402
import paired_wind as pw                                       # noqa: E402
from methane_t import N_GRID, to_batch_t                       # noqa: E402
from methane_t_uncertain import PhysHeadNet                    # noqa: E402
from conformal import (region_mask, regions, tail_scores,      # noqa: E402
                       tail_threshold)

OUT = pathlib.Path(__file__).resolve().parent
FROZEN = pathlib.Path("/projectnb/rise-tower/eric1/csr-data")
N_CELLS = N_GRID * N_GRID
SITE = 500.0

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF = "#fcfcfb"
GOLD, REDX = "#f2c230", "#e04343"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 16, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})


def forward_probs(model, d, stats, device):
    """Softmax heatmaps for a split, identical to stage_train's probs()."""
    out = []
    n = len(d["ids"])
    with torch.no_grad():
        for lo in range(0, n, 512):
            idx = np.arange(lo, min(lo + 512, n))
            b = to_batch_t(d, idx, device)
            b["stats"] = stats[idx].to(device).view(len(idx), -1, N_GRID,
                                                    N_GRID)
            out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    alpha = cfg["conformal"]["alpha"]

    ckpt_path = dd / "checkpoints" / "pw_noisy_ensr_seed1.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PhysHeadNet(cfg, 4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"checkpoint: {ckpt_path} (val_nll {ckpt['val_nll']:.4f})")

    views = {}
    for split in ("calib", "test"):
        d_clean = dict(np.load(dd / f"ch4t_{split}.npz", allow_pickle=True))
        views[split] = pw.make_view(cfg, dd, (split, d_clean), "noisy",
                                    "ensr")
    d_cal, st_cal = views["calib"]
    d_tst, st_tst = views["test"]
    Pc = forward_probs(model, d_cal, st_cal, device)
    Pt = forward_probs(model, d_tst, st_tst, device)

    rngc = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(Pc, d_cal["true_cell"], rngc), alpha)
    reg = regions(Pt, th, d_tst["true_cell"])
    radii = np.sqrt(reg["sizes"] / N_CELLS / np.pi) * SITE
    print(f"conformal tail threshold t_hat = {th:.6e}  (alpha={alpha})")
    print(f"retrained model: coverage {reg['covered'].mean():.4f}, "
          f"median radius {np.median(radii):.1f} m, "
          f"<50 m {(radii < 50).mean() * 100:.1f}%")

    fro = np.load(FROZEN / "pw_audit_pw_noisy_ensr_seed1_noisy.npz")
    fr = np.sqrt(fro["sizes"] / N_CELLS / np.pi) * SITE
    same = (fro["sizes"] == reg["sizes"]).mean()
    print(f"frozen audit:    coverage {fro['covered'].mean():.4f}, "
          f"median radius {np.median(fr):.1f} m, "
          f"<50 m {(fr < 50).mean() * 100:.1f}%")
    print(f"per-scenario region sizes identical to frozen audit: "
          f"{same * 100:.1f}% of 2,000 "
          f"(median |Delta radius| "
          f"{np.median(np.abs(radii - fr)):.1f} m)")

    # ---- scenario pick: 8 masts, covered, radius nearest the 71 m median
    ns = d_tst["n_sensors"].astype(int)
    ok = (ns == 8) & reg["covered"]
    i = int(np.argmin(np.where(ok, np.abs(radii - 71.0), np.inf)))
    sid = str(d_tst["ids"][i])
    print(f"scenario: index {i}, id {sid}, {ns[i]} masts, "
          f"q {d_tst['q'][i]:.0f} kg/h, region radius {radii[i]:.1f} m, "
          f"covered {bool(reg['covered'][i])}")

    maps4 = st_tst[i].numpy().reshape(4, N_GRID, N_GRID)
    prob = Pt[i].reshape(N_GRID, N_GRID)
    mask = region_mask(Pt[i], th).reshape(N_GRID, N_GRID)
    sx = d_tst["sensors"][i, :ns[i]] * SITE
    tc = int(d_tst["true_cell"][i])
    tx = ((tc % N_GRID + 0.5) / N_GRID * SITE,
          (tc // N_GRID + 0.5) / N_GRID * SITE)
    am = int(Pt[i].argmax())
    ax_ = ((am % N_GRID + 0.5) / N_GRID * SITE,
           (am // N_GRID + 0.5) / N_GRID * SITE)

    # ---- figure
    fig = plt.figure(figsize=(10.4, 13.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 3.0], hspace=0.24,
                          wspace=0.14, left=0.045, right=0.955, top=0.90,
                          bottom=0.045)
    panels = [  # title, channel index, cmap, note
        ("Source evidence", 0, "viridis", "input ch. 1"),
        ("Sensor visibility", 2, "viridis", "input ch. 3"),
        ("Wind-error\nsensitivity", 1, "viridis", "input ch. 2"),
        ("Fit quality", 3, "viridis_r",
         "input ch. 4 — bright = good fit"),
    ]
    ext = [0, SITE, 0, SITE]
    for k, (title, ch, cm, note) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        m = maps4[ch]
        lo, hi = np.percentile(m, [1, 99])
        im = ax.imshow(np.clip(m, lo, hi), origin="lower", cmap=cm,
                       extent=ext, interpolation="bilinear")
        ax.scatter(sx[:, 0], sx[:, 1], marker="^", s=52, c="white",
                   edgecolors=INK, linewidths=1.0, zorder=6)
        ax.plot(*tx, marker="*", color=GOLD, ms=13, mec=INK, mew=0.9,
                zorder=7)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=7,
                     linespacing=1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.055, pad=0.045, aspect=16)
        cb.ax.tick_params(labelsize=11.5)
        cb.outline.set_visible(False)
        cb.set_label(note, fontsize=12, color=MUTED)

    axb = fig.add_subplot(gs[1, :])
    im = axb.imshow(prob, origin="lower", cmap="magma", extent=ext,
                    norm=LogNorm(vmin=1e-6, vmax=prob.max()),
                    interpolation="bilinear")
    axb.contourf(mask.astype(float), levels=[0.5, 1.5], colors=["white"],
                 alpha=0.18, extent=ext, zorder=3)
    axb.contour(mask.astype(float), levels=[0.5], colors=["white"],
                linewidths=3.0, extent=ext, zorder=4)
    axb.scatter(sx[:, 0], sx[:, 1], marker="^", s=170, c="white",
                edgecolors=INK, linewidths=1.4, zorder=6,
                label=f"sensor masts ({ns[i]})")
    axb.plot(*tx, marker="*", color=GOLD, ms=30, mec=INK, mew=1.5, lw=0,
             zorder=7, label="true source")
    axb.plot(*ax_, marker="x", color=REDX, ms=20, mew=4.0, lw=0, zorder=7,
             label="argmax prediction",
             path_effects=[pe.withStroke(linewidth=6, foreground="white")])
    axb.plot([], [], color="white", lw=3, label="90% conformal region")

    axb.plot([20, 120], [24, 24], color="white", lw=4,
             solid_capstyle="butt", zorder=8)
    axb.annotate("100 m", xy=(70, 34), ha="center", fontsize=15,
                 color="white", zorder=8)
    axb.annotate(f"90% region radius {radii[i]:.0f} m",
                 xy=(0.985, 0.975), xycoords="axes fraction", ha="right",
                 va="top", fontsize=17, fontweight="bold", color="white",
                 zorder=8)
    axb.set_title("Combined softmax probability map "
                  "$\\hat{p}(c \\mid x)$ and calibrated region",
                  fontsize=19, fontweight="bold", loc="left", pad=10)
    axb.set_xticks([0, 250, 500])
    axb.set_yticks([0, 250, 500])
    axb.set_xlabel("site coordinate (m)", fontsize=15)
    axb.tick_params(labelsize=13)
    cb = fig.colorbar(im, ax=axb, fraction=0.043, pad=0.02)
    cb.set_label("predicted source probability (log scale)", fontsize=14,
                 color=INK2)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)
    axb.legend(loc="upper left", frameon=True, fontsize=13.5,
               framealpha=0.55, facecolor="black", edgecolor="none",
               labelcolor="white")
    fig.suptitle("From physics maps to a guaranteed region",
                 fontsize=22, fontweight="bold", x=0.045, ha="left",
                 y=0.975)
    fig.text(0.045, 0.945, "physics-guided DeepSets (ensr, seed 1) on "
             f"held-out scenario {sid} — {ns[i]} masts, "
             f"{d_tst['q'][i]:.0f} kg h$^{{-1}}$, region radius "
             f"{radii[i]:.0f} m (subgroup median ≈ 71 m)",
             fontsize=13.5, color=MUTED, ha="left")

    for ext_ in ("pdf", "png"):
        p = OUT / f"fig1_physics_to_region.{ext_}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
