"""XAI Figure 1 — physics-to-region explanation panel, from the REAL model.

Loads the retrained paper checkpoint (PhysHeadNet = DeepSetsT + zero-init
conv head, views=noisy, maps=ensr, seed=1), rebuilds the exact test/calib
views with make_view, recomputes softmax heatmaps and the split-conformal
threshold with the repo's own conformal.py, and cross-checks the resulting
regions against the frozen audit before drawing anything.

Layout: top row = the four physics feature maps in the canonical order
(source evidence, sensor visibility, wind-error sensitivity, fit quality),
exactly as fed to the network; bottom = the model's softmax probability map
with the calibrated 90% region, masts, true source, argmax, and scale bar.
All styling comes from xai_style.py (shared with xai_fig23.py).

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

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from common import load_config, get_device, resolve            # noqa: E402
import paired_wind as pw                                       # noqa: E402
from methane_t import N_GRID, to_batch_t                       # noqa: E402
from methane_t_uncertain import PhysHeadNet                    # noqa: E402
from conformal import (region_mask, regions, tail_scores,      # noqa: E402
                       tail_threshold)
from xai_style import (CMAP_MAG, CMAP_MAG_R, INK2, MAP_ORDER,  # noqa: E402
                       MUTED, apply_style, draw_argmax, draw_masts,
                       draw_source, frame, header, legend_handles,
                       save_all)

OUT = HERE
FROZEN = pathlib.Path("/projectnb/rise-tower/eric1/csr-data")
N_CELLS = N_GRID * N_GRID
SITE = 500.0

apply_style()


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

    # ---- figure (presentation only below this line)
    # constrained layout (global default from xai_style) fights the
    # equal-aspect hero panel here, so this one figure is laid out by hand
    fig = plt.figure(figsize=(10.4, 12.9))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 2.8], hspace=0.16,
                          wspace=0.14, left=0.045, right=0.955, top=0.965,
                          bottom=0.045)
    ext = [0, SITE, 0, SITE]
    for k, (title, ch) in enumerate(MAP_ORDER):
        ax = fig.add_subplot(gs[0, k])
        m = maps4[ch]
        lo, hi = np.percentile(m, [1, 99])
        cm = CMAP_MAG_R if title == "Fit quality" else CMAP_MAG
        im = ax.imshow(np.clip(m, lo, hi), origin="lower", cmap=cm,
                       extent=ext, interpolation="bilinear")
        draw_masts(ax, sx)
        draw_source(ax, tx)
        ax.set_title(title.replace("Wind-error ", "Wind-error\n"),
                     fontsize=16, pad=7, linespacing=1.15)
        frame(ax)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.055, pad=0.03, aspect=15)
        cb.ax.tick_params(labelsize=11.5)
        cb.outline.set_visible(False)
        if title == "Fit quality":
            cb.set_label("bright = good fit", fontsize=12, color=MUTED)

    axb = fig.add_subplot(gs[1, :])
    im = axb.imshow(prob, origin="lower", cmap=CMAP_MAG, extent=ext,
                    norm=LogNorm(vmin=1e-6, vmax=prob.max()),
                    interpolation="bilinear")
    axb.contourf(mask.astype(float), levels=[0.5, 1.5], colors=["white"],
                 alpha=0.18, extent=ext, zorder=3)
    axb.contour(mask.astype(float), levels=[0.5], colors=["white"],
                linewidths=3.0, extent=ext, zorder=4)
    draw_masts(axb, sx, hero=True)
    draw_source(axb, tx, hero=True)
    draw_argmax(axb, ax_, hero=True)
    hleg = legend_handles(ns[i]) + [
        plt.Line2D([], [], color="white", lw=3,
                   label="90% conformal region")]
    axb.legend(handles=hleg, loc="upper left", frameon=True, fontsize=13.5,
               framealpha=0.55, facecolor="black", edgecolor="none",
               labelcolor="white")

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
                  fontsize=19, loc="left", pad=10)
    axb.set_xticks([0, 250, 500])
    axb.set_yticks([0, 250, 500])
    axb.set_xlabel("site coordinate (m)", fontsize=15)
    axb.tick_params(labelsize=13)
    cb = fig.colorbar(im, ax=axb, fraction=0.043, pad=0.02)
    cb.set_label("predicted source probability (log scale)", fontsize=14,
                 color=INK2)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)

    header(fig, "From physics maps to a guaranteed region",
           f"DeepSets ensr, seed 1 · scenario {sid} · {ns[i]} masts · "
           f"{d_tst['q'][i]:.0f} kg h$^{{-1}}$ · region radius "
           f"{radii[i]:.0f} m ≈ subgroup median",
           ty=1.022, sy=1.001, x=0.045)

    save_all(fig, str(OUT / "fig1_physics_to_region"))
    plt.close(fig)


if __name__ == "__main__":
    main()
