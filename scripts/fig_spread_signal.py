"""Poster figure: the Monte Carlo spread is a signal, not noise.

Two real cells from one test scenario with near-identical MEAN evidence but
very different SPREAD across the K=8 wind draws: per-draw traces (left), the
mean map where they look alike (center), the spread map where they don't
(right).  Usage: python scripts/fig_spread_signal.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

sys.path.insert(0, "src")
from common import cell_centers, load_config, resolve
from methane_t import N_GRID, cell_responses_t, stab_of
from methane_t_uncertain import K_MAPS, perturb_wind, split_tag, zmap

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#9ec5f4",
    "#6da7ec", "#3987e5", "#256abf", "#104281"])
CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5", "#f6b899",
    "#eb6834", "#a03c14"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11})

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
sx = d["sensors"][i, :ns] * 500
y = torch.tensor(d["readings"][i, :ns][keep], dtype=torch.float64)
cells = torch.tensor(cell_centers(N_GRID), dtype=torch.float64)

Z = []
for k in range(K_MAPS):
    uk = perturb_wind(u_obs[i], np.random.default_rng([root, 92, tag, i, k]))
    g = cell_responses_t(cells, d["sensors"][i, :ns], uk, stab_of(d, i),
                         torch.device("cpu"))[:, torch.tensor(keep)]
    a = (g @ y).numpy()[None]
    b = (g * g).sum(1).numpy()[None]
    Z.append(zmap(a, b, d["sigma"][i:i + 1])[0])
Z = np.stack(Z)                       # (8, 4096)
m, s = Z.mean(0), Z.std(0)

# pick two cells: near-identical mean (upper band), spread ratio maximized,
# spatially separated so the markers do not collide
band = (m > np.percentile(m, 88)) & (m < np.percentile(m, 97))
idx = np.where(band)[0]
best, pair = 0.0, None
for a_ in idx[np.argsort(s[idx])[:40]]:          # low-spread candidates
    for b_ in idx[np.argsort(s[idx])[-40:]]:     # high-spread candidates
        if abs(m[a_] - m[b_]) > 0.05 * (m.max() - m.min()):
            continue
        da = np.hypot(a_ % 64 - b_ % 64, a_ // 64 - b_ // 64)
        score = (s[b_] / (s[a_] + 1e-9)) * min(da / 10, 1.0)
        if score > best:
            best, pair = score, (a_, b_)
ca, cb = pair
pos = lambda c: ((c % 64 + 0.5) / 64 * 500, (c // 64 + 0.5) / 64 * 500)
pa, pb = pos(ca), pos(cb)

fig = plt.figure(figsize=(11.8, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], left=0.06,
                      right=0.985, top=0.80, bottom=0.16, wspace=0.22)

# ---- left: per-draw evidence for the two cells
ax = fig.add_subplot(gs[0, 0])
ks = np.arange(1, 9)
ax.axhline(m[ca], color=BLUE, lw=1.2, ls=(0, (4, 3)), alpha=0.8)
ax.axhline(m[cb], color=ORANGE, lw=1.2, ls=(0, (4, 3)), alpha=0.8)
ax.plot(ks, Z[:, ca], "o-", color=BLUE, lw=1.6, ms=7, mec="white", mew=1.2,
        label="stable cell", zorder=4)
ax.plot(ks, Z[:, cb], "s-", color=ORANGE, lw=1.6, ms=7, mec="white", mew=1.2,
        label="fragile cell", zorder=4)
ax.annotate("same mean", xy=(8.35, (m[ca] + m[cb]) / 2), fontsize=9.5,
            color=INK2, ha="left", va="center", annotation_clip=False)
ax.set_xlabel("wind draw $k$")
ax.set_ylabel("evidence at the cell")
ax.set_xticks(ks)
ax.set_xlim(0.6, 8.4)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=10)
ax.set_title("evidence across the 8 plausible winds", fontsize=10.5)

def map_panel(ax, img, cm, title):
    lo, hi = np.percentile(img, [2, 99])
    ax.imshow(np.clip(img, lo, hi).reshape(64, 64), origin="lower", cmap=cm,
              extent=[0, 500, 0, 500], interpolation="bilinear")
    ax.scatter(sx[:, 0], sx[:, 1], s=13, c=INK, edgecolors="white",
               linewidths=0.7, zorder=5)
    ax.scatter(*pa, marker="o", s=130, fc="none", ec=BLUE, lw=2.6, zorder=6)
    ax.scatter(*pb, marker="s", s=130, fc="none", ec=ORANGE, lw=2.6, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BASE)
    ax.set_title(title, fontsize=10.5, pad=4)

ax2 = fig.add_subplot(gs[0, 1])
map_panel(ax2, m, CMB, "mean map: the cells look alike")
ax3 = fig.add_subplot(gs[0, 2])
map_panel(ax3, s, CMO, "spread map: only here they differ")

fig.suptitle("the Monte Carlo spread is a signal, not noise",
             fontsize=12.5, color=INK, y=0.965)

out = pathlib.Path("figures/paper")
fig.savefig(out / "fig_spread_signal.pdf", bbox_inches="tight")
fig.savefig(out / "fig_spread_signal.png", dpi=220, bbox_inches="tight")
print("wrote fig_spread_signal (scenario %d; cells %d, %d; spread ratio %.1f)"
      % (i, ca, cb, s[cb] / s[ca]))
