"""Representative CH4-T simulator graphic: the true plume at three time
steps as the wind meanders, and the noisy readings the masts record.
Real test scenario, real forward model, stored readings.

Usage: python scripts/fig_simulator.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

sys.path.insert(0, "src")
from common import load_config, resolve
from methane_model import plume_ppm_per_kgh
from methane_t import stab_of

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CMB = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb",
    "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11})

cfg = load_config()
dd = resolve(cfg, "data_dir")
d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))

# scenario pick: 6 masts, strong source, big direction meander (visible sweep)
ns_all = d["n_sensors"].astype(int)
th = np.degrees(np.unwrap(np.arctan2(d["u_seq"][:, :, 1],
                                     d["u_seq"][:, :, 0]), axis=1))
swing = th.max(1) - th.min(1)
early_std = th[:, :6].std(1)         # steady start -> narrow early beam
cand = (ns_all == 6) & (d["q"] > 300) & (swing > 60) & (swing < 110) \
       & (early_std < 7)
i = int(np.where(cand)[0][0])

ns = int(d["n_sensors"][i])
sx = d["sensors"][i, :ns] * 500
xs = d["xs"][i]
q = float(d["q"][i])
u_seq = d["u_seq"][i]
stab = stab_of(d, i)

# concentration field on a 220x220 grid at three snapshots
R = 220
g1 = np.linspace(0, 1, R)
gx, gy = np.meshgrid(g1, g1)
pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1),
                   dtype=torch.float64)
src = torch.tensor(xs, dtype=torch.float64).expand(R * R, 2)
# running time-average: the pollution footprint grows as the wind meanders
SNAPS = (2, 10, 30)                 # minutes included in each panel
st = torch.full((R * R,), stab, dtype=torch.long)
csum, fields = np.zeros((R, R)), []
for m in range(30):
    u = torch.tensor(u_seq[m], dtype=torch.float64).expand(R * R, 2)
    csum += q * plume_ppm_per_kgh(pts, src, u, st).numpy().reshape(R, R)
    if (m + 1) in SNAPS:
        fields.append(csum.copy())          # cumulative exposure (ppm-min)
vmax = np.percentile(fields[-1], 99.5)
FOOTPRINT = 3.0                             # ppm-min detectability outline

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55),
                         gridspec_kw=dict(wspace=0.10),
                         layout="constrained")

for ax, t, c in zip(axes, SNAPS, fields):
    ax.imshow(np.clip(c, 0, vmax), origin="lower", cmap=CMB,
              extent=[0, 500, 0, 500],
              norm=PowerNorm(0.5, vmin=0, vmax=vmax),
              interpolation="bilinear")
    ax.contour(c, levels=[FOOTPRINT], extent=[0, 500, 0, 500],
               colors=[INK2], linewidths=1.3, linestyles="dashed")
    ax.scatter(sx[:, 0], sx[:, 1], s=42, c=INK, edgecolors="white",
               linewidths=1.3, zorder=5)
    ax.plot(xs[0] * 500, xs[1] * 500, marker="*", color=INK, ms=15,
            mec="white", mew=1.0, zorder=6)
    ax.set_title(f"First {t} Minutes" if t < 30 else "All 30 Minutes",
                 fontsize=12, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(INK2)
        sp.set_linewidth(1.3)
        sp.set_zorder(10)

out = pathlib.Path("figures/paper")
fig.savefig(out / "fig_simulator.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out / "fig_simulator.png", dpi=300, bbox_inches="tight")
from pptx import Presentation
from pptx.util import Inches
from PIL import Image
img = out / "fig_simulator.png"
w_px, h_px = Image.open(img).size
prs = Presentation()
prs.slide_width = Inches(w_px / 300)
prs.slide_height = Inches(h_px / 300)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.shapes.add_picture(str(img), 0, 0, width=prs.slide_width,
                         height=prs.slide_height)
prs.save(out / "fig_simulator.pptx")
print("wrote fig_simulator.{pdf,png,pptx} (scenario %d, q=%.0f kg/h, "
      "swing %.0f deg)" % (i, q, swing[i]))
