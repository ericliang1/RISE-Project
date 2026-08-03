"""Minimal Monte Carlo process figure for the poster: sample K=8 winds
around the measurement, compute the evidence map under each, average.
Usage: python scripts/fig_mc_simple.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "src")
from common import cell_centers, load_config, resolve
from methane_t import N_GRID, cell_responses_t, stab_of
from methane_t_uncertain import K_MAPS, perturb_wind, split_tag, zmap

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
BASE = "#c3c2b7"
CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#9ec5f4",
    "#6da7ec", "#3987e5", "#256abf", "#104281"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12})

cfg = load_config()
dd = resolve(cfg, "data_dir")
root = cfg["seeds"]["root_entropy"]
tag = split_tag("test")
d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
u_obs = np.load(dd / "ch4tu_test_uobs.npy")

z = np.load(dd / "pw_audit_pw_noisy_ens_gnn_seed1_noisy.npz")
r = np.sqrt(z["sizes"].astype(float) / 4096 / np.pi) * 500
grp = d["n_sensors"].astype(int) == 7
i = int(np.argmin(np.where(grp, np.abs(r - np.median(r[grp])), np.inf)))

ns = int(d["n_sensors"][i])
keep = d["keep"][i, :ns]
y = torch.tensor(d["readings"][i, :ns][keep], dtype=torch.float64)
cells = torch.tensor(cell_centers(N_GRID), dtype=torch.float64)

draws, zmaps = [], []
for k in range(K_MAPS):
    uk = perturb_wind(u_obs[i], np.random.default_rng([root, 92, tag, i, k]))
    draws.append(uk)
    g = cell_responses_t(cells, d["sensors"][i, :ns], uk, stab_of(d, i),
                         torch.device("cpu"))[:, torch.tensor(keep)]
    a = (g @ y).numpy()[None]
    b = (g * g).sum(1).numpy()[None]
    zmaps.append(zmap(a, b, d["sigma"][i:i + 1])[0].reshape(64, 64))
zmean = np.stack(zmaps).mean(0)

fig = plt.figure(figsize=(10.6, 3.3))

# ---- step 1: wind compass, measured arrow + 8 sampled arrows
ax1 = fig.add_axes([0.03, 0.16, 0.24, 0.66])
ax1.set_xlim(-1.15, 1.15); ax1.set_ylim(-1.15, 1.15)
ax1.set_aspect("equal"); ax1.axis("off")
ax1.add_patch(plt.Circle((0, 0), 1.02, fc="none", ec=BASE, lw=1.2))
T_SHOW = 10                      # one time step: the per-step error fan
um = u_obs[i][T_SHOW]
sm = np.linalg.norm(um)
for uk in draws:
    v = uk[T_SHOW]
    v = v / sm * 0.88                # length shows the 10% speed error too
    ax1.annotate("", xy=v, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7,
                                 alpha=0.55, mutation_scale=16))
vm = um / sm * 0.88
ax1.annotate("", xy=vm, xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.8,
                             mutation_scale=20))
ax1.text(0, -1.32, "measured wind (black),\n$K{=}8$ samples from its\n"
         "error model (blue)", ha="center", fontsize=10.5, color=INK2)
ax1.set_title("1. sample plausible winds", fontsize=12, color=INK, pad=10)

def show_map(ax, img):
    lo, hi = np.percentile(img, [2, 99])
    ax.imshow(np.clip(img, lo, hi), origin="lower", cmap=CMB,
              extent=[0, 1, 0, 1], interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("white"); sp.set_linewidth(1.8)

# ---- step 2: stacked per-draw maps
x0, y0, w, h = 0.40, 0.20, 0.155, 0.52
for j, k in enumerate((7, 5, 3, 0)):
    ax = fig.add_axes([x0 + 0.016 * j, y0 + 0.045 * j, w, h])
    show_map(ax, zmaps[k])
fig.text(x0 + 0.10, 0.075, "one evidence map\nper sampled wind",
         ha="center", fontsize=10.5, color=INK2)
fig.text(x0 + 0.10, 0.93, "2. Monte Carlo wind ensemble",
         ha="center", fontsize=12, color=INK)

# ---- step 3: the averaged map
ax3 = fig.add_axes([0.755, 0.20, 0.185, 0.62])
show_map(ax3, zmean)
for sp in ax3.spines.values():
    sp.set_color(BLUE); sp.set_linewidth(2.2)
fig.text(0.848, 0.075, "wind-uncertainty-aware\ninput for the network",
         ha="center", fontsize=10.5, color=INK2)
fig.text(0.848, 0.93, "3. average the $K$ maps", ha="center",
         fontsize=12, color=INK)

for xc in (0.325, 0.685):
    fig.text(xc, 0.48, "$\\Rightarrow$", fontsize=22, color=INK2,
             ha="center", va="center")

out = pathlib.Path("figures/paper")
fig.savefig(out / "fig_mc_simple.pdf", bbox_inches="tight")
fig.savefig(out / "fig_mc_simple.png", dpi=220, bbox_inches="tight")
print("wrote fig_mc_simple (scenario %d)" % i)
