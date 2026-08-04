"""XAI Figure 1b (v3) — the three physics feature maps alone, as a 1x3
strip.

Compact poster inset for the Methodology section: the same three maps shown
in the top row of fig1_physics_to_region (same scenario — read from
fig1_scenario.json, written by xai_fig1.py — same data files, same styling
from xai_style.py), but arranged as a single row with short labels and one
caption line — no colorbars, no header — to replace a dense text block on
the poster.

The maps are loaded straight from the frozen npz file exactly as
paired_wind.make_view assembles the "ens" input (3 ensemble channels; the
v2 residual channel no longer exists in the method); no model or GPU is
needed.

Usage (from repo root, after `source scripts/env.sh` and xai_fig1.py):
  python "Poster Graphics v3/xai graphs/xai_fig1_maps_grid.py"
"""
import json
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
from xai_style import (CMAP_MAG, MUTED,                        # noqa: E402
                       apply_style, draw_masts, draw_source, frame,
                       save_all)

OUT = HERE

# Short poster names -> ens channel index (same channels as MAP_ORDER in
# xai_style; "Wind sensitivity" is the compact label for the poster).
GRID_ORDER = [("Source evidence", 0), ("Sensor visibility", 2),
              ("Wind sensitivity", 1)]

CAPTION = ("Physics-derived feature maps for one test scenario — the three "
           "inputs the network sees\n(▲ sensor masts, ★ true source; "
           "bright = high value).")

apply_style()


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")

    sc = json.load(open(HERE / "fig1_scenario.json"))
    i, sid = int(sc["index"]), sc["id"]

    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    assert str(d["ids"][i]) == sid, "fig1_scenario.json is stale"
    ns = int(d["n_sensors"][i])

    # ens maps, exactly the "ens" input paired_wind.make_view assembles
    maps3 = (np.load(dd / "ch4tu_test_maps_ens.npz")["maps"][i]
             .reshape(3, N_GRID, N_GRID))

    SITE = 500.0
    sx = d["sensors"][i, :ns] * SITE
    tc = int(d["true_cell"][i])
    tx = ((tc % N_GRID + 0.5) / N_GRID * SITE,
          (tc // N_GRID + 0.5) / N_GRID * SITE)
    print(f"scenario {sid}: index {i}, {ns} masts, "
          f"q {d['q'][i]:.0f} kg/h")

    fig = plt.figure(figsize=(10.5, 4.5))
    fig.set_layout_engine("none")
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.885,
                          bottom=0.16, wspace=0.08)
    ext = [0, SITE, 0, SITE]
    for k, (title, ch) in enumerate(GRID_ORDER):
        ax = fig.add_subplot(gs[0, k])
        m = maps3[ch]
        lo, hi = np.percentile(m, [1, 99])
        ax.imshow(np.clip(m, lo, hi), origin="lower", cmap=CMAP_MAG,
                  extent=ext, interpolation="bilinear")
        draw_masts(ax, sx)
        draw_source(ax, tx)
        ax.set_title(title, fontsize=16, pad=7)
        frame(ax)

    cap = fig.text(0.5, 0.055, CAPTION, ha="center", va="center",
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
