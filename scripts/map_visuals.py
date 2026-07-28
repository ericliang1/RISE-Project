"""Clean standalone visuals for each of the four physics feature maps
(same 8-mast measured-wind scenario as the paper figures).
Usage: python scripts/map_visuals.py -> figures/paper/fig_map_*.{pdf,png}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from common import load_config, resolve

INK, BASE, SURF = "#0b0b0b", "#c3c2b7", "#ffffff"
CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb",
    "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5",
    "#f6b899", "#eb6834", "#a03c14"])
plt.rcParams["font.family"] = "sans-serif"

cfg = load_config()
dd = resolve(cfg, "data_dir")
N_CELLS = 64 * 64
d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
zb = np.load(dd / "pw_audit_pw_noisy_nomaps_seed1_noisy.npz")
zo = np.load(dd / "pw_audit_pw_noisy_ensr_seed1_noisy.npz")


def rad(s):
    return np.sqrt(s.astype(float) / N_CELLS / np.pi) * 500


rb, ro = rad(zb["sizes"]), rad(zo["sizes"])
grp = d["n_sensors"].astype(int) == 8
dist = (np.log(rb / np.median(rb[grp])) ** 2
        + np.log(ro / np.median(ro[grp])) ** 2)
dist[~grp] = np.inf
i = int(np.argmin(dist))
ens = np.load(dd / "ch4tu_test_maps_ens.npz")["maps"][i]
rm = np.load(dd / "pw_test_resid_marg.npz")["noisy"][i].astype(float)
ns = int(d["n_sensors"][i])
sx = d["sensors"][i, :ns] * 500
tc = int(d["true_cell"][i])
tx = ((tc % 64 + 0.5) / 64 * 500, (tc // 64 + 0.5) / 64 * 500)

maps = [("evidence", ens[0].reshape(64, 64), CMB),
        ("fragility", ens[1].reshape(64, 64), CMO),
        ("sensitivity", ens[2].reshape(64, 64), CMB),
        ("fitquality", (10 - rm).reshape(64, 64), CMB)]

for name, img, cm in maps:
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lo, hi = np.percentile(img, [2, 99])
    ax.imshow(np.clip(img, lo, hi), origin="lower", cmap=cm,
              extent=[0, 500, 0, 500], interpolation="bilinear")
    ax.scatter(sx[:, 0], sx[:, 1], s=46, c=INK, edgecolors="white",
               linewidths=1.5, zorder=5)
    ax.add_patch(Circle(tx, 13, ec=INK, fc="none", lw=2.0, zorder=6))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASE); s.set_linewidth(1.2)
    fig.savefig(f"figures/paper/fig_map_{name}.pdf",
                bbox_inches="tight", facecolor=SURF, pad_inches=0.03)
    fig.savefig(f"figures/paper/fig_map_{name}.png", dpi=260,
                bbox_inches="tight", facecolor=SURF, pad_inches=0.03)
    plt.close(fig)
    print(f"wrote fig_map_{name}")
