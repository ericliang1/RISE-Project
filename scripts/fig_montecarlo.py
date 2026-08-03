"""Monte Carlo wind-marginalization figure: one measured wind, K=8 plausible
winds, the per-draw evidence maps they induce, and the mean/spread summary
channels the network receives.  Real test scenario, real pipeline.

Usage: python scripts/fig_montecarlo.py -> figures/paper/fig_montecarlo.{pdf,png}
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
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
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

# representative scenario: 7 masts, method radius near that stratum's median
z = np.load(dd / "pw_audit_pw_noisy_ens_gnn_seed1_noisy.npz")
r = np.sqrt(z["sizes"].astype(float) / 4096 / np.pi) * 500
ns_all = d["n_sensors"].astype(int)
grp = ns_all == 7
i = int(np.argmin(np.where(grp, np.abs(r - np.median(r[grp])), np.inf)))

ns = int(d["n_sensors"][i])
keep = d["keep"][i, :ns]
sx = d["sensors"][i, :ns] * 500
tc = int(d["true_cell"][i])
tx = ((tc % 64 + 0.5) / 64 * 500, (tc // 64 + 0.5) / 64 * 500)
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
Z = np.stack(zmaps)
zmean, zspread = Z.mean(0), Z.std(0)

fig = plt.figure(figsize=(12.2, 3.6))
gs = fig.add_gridspec(2, 6, width_ratios=[2.7, 0.45, 1, 1, 0.45, 1],
                      left=0.05, right=0.985, top=0.80, bottom=0.13,
                      hspace=0.30, wspace=0.14)

# ---- panel A: measured wind + K=8 Monte Carlo draws (direction vs time)
axA = fig.add_subplot(gs[:, 0])
steps = np.arange(1, 31)
th_obs = np.degrees(np.unwrap(np.arctan2(u_obs[i][:, 1], u_obs[i][:, 0])))
for k, uk in enumerate(draws):
    th = np.degrees(np.unwrap(np.arctan2(uk[:, 1], uk[:, 0])))
    th += np.round((th_obs.mean() - th.mean()) / 360) * 360
    axA.plot(steps, th, color=BLUE, lw=1.1, alpha=0.45,
             label="8 plausible winds" if k == 0 else None)
axA.plot(steps, th_obs, color=INK, lw=2.6,
         label="measured wind $\\hat{w}$")
axA.set_xlabel("time step")
axA.set_ylabel("wind direction (deg)")
axA.set_xlim(1, 30)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.yaxis.grid(True, color=GRID, lw=0.8)
axA.set_axisbelow(True)
axA.legend(loc="upper left", frameon=False, fontsize=9.5)
axA.set_title("sample the anemometer's error model ($K{=}8$)",
              fontsize=10.5, color=INK)

def map_panel(ax, img, cm, title, tcol=INK):
    lo, hi = np.percentile(img, [2, 99])
    ax.imshow(np.clip(img, lo, hi), origin="lower", cmap=cm,
              extent=[0, 500, 0, 500], interpolation="bilinear")
    ax.scatter(sx[:, 0], sx[:, 1], s=13, c=INK, edgecolors="white",
               linewidths=0.7, zorder=5)
    ax.add_patch(Circle(tx, 18, ec=INK, fc="none", lw=1.3, zorder=6))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASE)
    ax.set_title(title, fontsize=9.8, color=tcol, pad=3)

# ---- panel B: four of the eight per-draw evidence maps (2x2)
for (row, col), k in zip(((0, 2), (0, 3), (1, 2), (1, 3)), (0, 2, 5, 7)):
    axT = fig.add_subplot(gs[row, col])
    map_panel(axT, zmaps[k], CMB, f"draw {k + 1}")
fig.text(0.565, 0.93, "evidence under each plausible wind (4 of 8 shown)",
         fontsize=10.5, color=INK, ha="center")

# ---- panel C: the two Monte Carlo summary channels
axM = fig.add_subplot(gs[0, 5])
map_panel(axM, zmean, CMB, "mean $\\rightarrow$ evidence map", BLUE)
axS = fig.add_subplot(gs[1, 5])
map_panel(axS, zspread, CMO, "std $\\rightarrow$ spread map", ORANGE)

# arrows between stages (in the spacer columns, vertically centered)
for xc in (0.415, 0.795):
    fig.text(xc, 0.47, "$\\Rightarrow$", fontsize=18, color=INK2,
             ha="center", va="center")

out = pathlib.Path("figures/paper")
fig.savefig(out / "fig_montecarlo.pdf", bbox_inches="tight")
fig.savefig(out / "fig_montecarlo.png", dpi=220, bbox_inches="tight")
print("wrote fig_montecarlo (scenario %d, %d masts)" % (i, ns))
