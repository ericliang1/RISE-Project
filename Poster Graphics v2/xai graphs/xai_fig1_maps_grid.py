"""XAI Figure 1b — the four physics feature maps alone, as a 2x2 grid.

Compact poster inset for the Methodology section: the same four maps shown
in the top row of fig1_physics_to_region (same scenario ch4t-test-001811,
same data files, same styling from xai_style.py), but arranged as a 2x2
grid with short labels and a single caption line — no colorbars, no
header — to replace a dense text block on the poster.

The maps are loaded straight from the frozen npz files exactly as
paired_wind.make_view assembles the "ensr" input (3 ensemble channels +
marginalized residual / R_MAX); no model or GPU is needed.

Usage (from repo root, after `source scripts/env.sh`):
  python "Poster Graphics v2/xai graphs/xai_fig1_maps_grid.py"
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from common import load_config, resolve                        # noqa: E402
from methane_t import N_GRID                                   # noqa: E402
from paired_wind import R_MAX                                  # noqa: E402
from xai_style import (CMAP_MAG, CMAP_MAG_R, MUTED,            # noqa: E402
                       apply_style, draw_masts, draw_source, frame,
                       save_all)

OUT = HERE
SITE = 500.0
SCENARIO_ID = "ch4t-test-001811"   # same scenario as fig1_physics_to_region

# Short poster names -> ensr channel index (same channels as MAP_ORDER in
# xai_style; "Wind sensitivity" is the compact label for the poster).
GRID_ORDER = [("Source evidence", 0), ("Sensor visibility", 2),
              ("Wind sensitivity", 1), ("Fit quality", 3)]

CAPTION = ("Physics-derived feature maps for one test scenario — the four "
           "inputs the network sees\n(▲ sensor masts, ★ true source; "
           "bright = high value, Fit quality inverted so bright = good "
           "fit).")

apply_style()


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")

    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    ids = np.asarray([str(s) for s in d["ids"]])
    i = int(np.flatnonzero(ids == SCENARIO_ID)[0])
    ns = int(d["n_sensors"][i])

    # ensr maps, assembled exactly as paired_wind.make_view does
    ens = np.load(dd / "ch4tu_test_maps_ens.npz")["maps"][i]
    r = np.load(dd / "pw_test_resid_marg.npz")["noisy"][i]
    r = np.minimum(r.astype(np.float32) / R_MAX, 1.0)[None, :]
    maps4 = np.concatenate([ens, r], 0).reshape(4, N_GRID, N_GRID)

    sx = d["sensors"][i, :ns] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % N_GRID + 0.5) / N_GRID * SITE,
          (tc // N_GRID + 0.5) / N_GRID * SITE)
    print(f"scenario {SCENARIO_ID}: index {i}, {ns} masts, "
          f"q {d['q'][i]:.0f} kg/h")

    fig = plt.figure(figsize=(7.2, 8.0))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(2, 2, left=0.03, right=0.97, top=0.94,
                          bottom=0.095, wspace=0.08, hspace=0.17)
    ext = [0, SITE, 0, SITE]
    for k, (title, ch) in enumerate(GRID_ORDER):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        m = maps4[ch]
        lo, hi = np.percentile(m, [1, 99])
        cm = CMAP_MAG_R if title == "Fit quality" else CMAP_MAG
        ax.imshow(np.clip(m, lo, hi), origin="lower", cmap=cm, extent=ext,
                  interpolation="bilinear")
        draw_masts(ax, sx)
        draw_source(ax, tx)
        ax.set_title(title, fontsize=16, pad=7)
        frame(ax)

    cap = fig.text(0.5, 0.032, CAPTION, ha="center", va="center",
                   fontsize=12.5, color=MUTED, linespacing=1.45)
    # auto-shrink the caption if it would overflow the canvas width
    fig.canvas.draw()
    wpx = fig.get_size_inches()[0] * fig.dpi
    extc = cap.get_window_extent()
    if extc.width > 0.97 * wpx:
        cap.set_fontsize(cap.get_fontsize() * 0.97 * wpx / extc.width)

    save_all(fig, str(OUT / "fig1_maps_grid"))
    plt.close(fig)


if __name__ == "__main__":
    main()
