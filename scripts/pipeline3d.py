"""3D-slab pipeline diagram (CNN-figure style): many sensor inputs ->
physics feature maps (real data as textured slabs) -> set network +
zero-init conv head -> one guaranteed region.
Usage: python scripts/pipeline3d.py -> figures/paper/fig_pipeline3d.*"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Polygon

from common import load_config, resolve

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, BASE, SURF = "#0b0b0b", "#52514e", "#898781", \
    "#c3c2b7", "#ffffff"
CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb",
    "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
CMO = LinearSegmentedColormap.from_list("o", ["#fcfcfb", "#fbe0d5",
    "#f6b899", "#eb6834", "#a03c14"])
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"]})

cfg = load_config()
dd = resolve(cfg, "data_dir")
N_CELLS = 64 * 64

# same scenario as the paper example (8 masts, measured wind)
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
det = np.load(dd / "ch4tu_test_maps_det.npz")["maps"][i]
rm = np.load(dd / "pw_test_resid_marg.npz")["noisy"][i].astype(float)
mo = np.unpackbits(zo["masks"][i])[:N_CELLS].reshape(64, 64)
ns = int(d["n_sensors"][i])
sx = d["sensors"][i, :ns]
um = d["u_mean"][i] / (np.linalg.norm(d["u_mean"][i]) + 1e-9)

fig, ax = plt.subplots(figsize=(13.2, 3.4))
ax.set_xlim(0, 132); ax.set_ylim(0, 34); ax.axis("off")
SK = 0.55       # 3d skew ratio


def slab(x, y, w, h, dep, face="#f4f4f2", img=None, cm=None, lw=1.1):
    """One 3d slab; optional heatmap texture on the front face."""
    top = [(x, y + h), (x + dep, y + h + dep * SK),
           (x + w + dep, y + h + dep * SK), (x + w, y + h)]
    side = [(x + w, y), (x + w + dep, y + dep * SK),
            (x + w + dep, y + h + dep * SK), (x + w, y + h)]
    ax.add_patch(Polygon(top, closed=True, fc="#e6e6e2", ec="#666",
                         lw=lw, zorder=4))
    ax.add_patch(Polygon(side, closed=True, fc="#dcdcd8", ec="#666",
                         lw=lw, zorder=4))
    if img is not None:
        lo, hi = np.percentile(img, [2, 99])
        ax.imshow(np.clip(img, lo, hi), origin="lower", cmap=cm,
                  extent=[x, x + w, y, y + h], interpolation="nearest",
                  zorder=5)
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h),
                              (x, y + h)], closed=True, fc="none",
                             ec="#666", lw=lw, zorder=6))
    else:
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h),
                              (x, y + h)], closed=True, fc=face,
                             ec="#666", lw=lw, zorder=5))


def arrow(x0, y0, x1, y1, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=14, color=INK2, lw=lw, zorder=8))


# ---- many inputs: masts + wind ----
ax.text(8.5, 31.5, "sensor readings", ha="center", fontsize=11,
        color=INK, fontweight="bold")
ends = np.linspace(10.5, 22.5, ns)
order = np.argsort(sx[:, 1])
for j, k in enumerate(order):
    mx, my = 3 + sx[k, 0] * 11, 8 + sx[k, 1] * 18
    ax.plot(mx, my, "o", ms=7, color=INK, mec="white", mew=1.2,
            zorder=7)
    arrow(mx + 0.8, my, 22.6, ends[j], lw=0.9)
ax.annotate("", xy=(11 + um[0] * 4, 4.4 + um[1] * 1.2),
            xytext=(11 - um[0] * 4, 4.4 - um[1] * 1.2),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
ax.text(11, 1.4, "wind $\\hat w$ + budget $\\varepsilon$", ha="center",
        fontsize=9.5, color=INK2)

# ---- physics feature maps as textured slabs ----
ax.text(34, 31.5, "physics feature maps", ha="center", fontsize=11,
        color=BLUE, fontweight="bold")
imgs = [(ens[0].reshape(64, 64), CMB), (ens[1].reshape(64, 64), CMO),
        (ens[2].reshape(64, 64), CMB), ((10 - rm).reshape(64, 64), CMB)]
labs = ["evidence", "fragility", "sensitivity", "fit"]
for k, ((img, cm), lb) in enumerate(zip(imgs, labs)):
    x0 = 24 + k * 6.4
    slab(x0, 8, 5.4, 16, 1.6, img=img, cm=cm)
ax.text(36, 4.7, "evidence · fragility · sensitivity · "
        "fit quality", ha="center", va="top", fontsize=9.5,
        color=INK2)
arrow(52.4, 16.5, 57.4, 16.5)

# ---- set network slabs (many-to-one narrowing) ----
ax.text(70, 31.5, "set network $f_\\theta$ + head $h_\\phi$",
        ha="center", fontsize=11, color=INK, fontweight="bold")
widths = [3.4, 3.0, 2.5, 2.0, 1.6]
heights = [20, 17, 14, 11, 8.5]
x0 = 59
for w, h in zip(widths, heights):
    slab(x0, 16.5 - h / 2 + 1.5, w, h, 1.8)
    x0 += w + 2.1
ax.text(69, 4.7, "zero-init head joins the logits",
        ha="center", va="top", fontsize=9.5, color=INK2)
arrow(x0 + 0.3, 16.5, x0 + 5.3, 16.5)

# ---- one output: guaranteed region ----
xo = x0 + 6.2
ax.text(xo + 10, 31.5, "one guaranteed region", ha="center",
        fontsize=11, color=INK, fontweight="bold")
slab(xo, 9.0, 18, 16.5, 2.0, img=np.where(mo, 1.0, 0.06),
     cm=LinearSegmentedColormap.from_list("m", ["#f7f9fc", "#6da7ec"]))
tc = int(d["true_cell"][i])
tx = (xo + (tc % 64 + 0.5) / 64 * 18,
      9.0 + (tc // 64 + 0.5) / 64 * 16.5)
ax.plot(*tx, marker="+", color=INK, ms=8, mew=1.8, zorder=9)
for k in range(ns):
    ax.plot(xo + sx[k, 0] * 18, 9.0 + sx[k, 1] * 16.5, "o", ms=3.5,
            color=INK, mec="white", mew=0.7, zorder=9)
ax.text(xo + 10.5, 4.7, "90% coverage, deployment-wind\n"
        "calibrated", ha="center", va="top", fontsize=9.5,
        color=INK2)

fig.savefig("figures/paper/fig_pipeline3d.pdf", bbox_inches="tight",
            facecolor="white")
fig.savefig("figures/paper/fig_pipeline3d.png", dpi=220,
            bbox_inches="tight", facecolor="white")
print("wrote fig_pipeline3d")
