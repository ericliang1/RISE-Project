"""Wide per-model SHAP beeswarm from shap_map_contributions.csv (no GPU).

User-requested poster variant of Panel B: every model split out (3 maps x
3 models = 9 rows), stretched wide with a tight x-axis so the direction of
each map's contribution is unmistakable, dots coloured by feature value
(per-row percentile, the standard SHAP summary gradient).

Usage (from repo root):
  python "Poster Graphics v3/shap/shap_beeswarm_models.py"
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "xai graphs"))
from xai_style import CMAP_FEATVAL, INK, INK2, MUTED, apply_style  # noqa: E402

apply_style()

MODELS = ["DeepSets", "GNN", "Set Transformer"]
MAPS = [("Source evidence", "source_evidence"),
        ("Sensor visibility", "sensor_visibility"),
        ("Wind spread", "wind_error_sensitivity")]

rows = list(csv.DictReader(open(HERE / "shap_map_contributions.csv")))
data = {}
for mdl in MODELS:
    sub = [r for r in rows if r["model"] == mdl]
    data[mdl] = {slug: (np.array([float(r[f"shapley_{slug}"]) for r in sub]),
                        np.array([float(r[f"featval_{slug}"]) for r in sub]))
                 for _, slug in MAPS}

phi_all = np.concatenate([data[m][s][0] for m in MODELS for _, s in MAPS])
XLO, XHI = -3.0, 3.0

fig, ax = plt.subplots(figsize=(17.0, 6.8), layout="constrained")
rng = np.random.default_rng(0)

GROUP_GAP, ROW_H = 1.05, 0.62
ys, ylabels, group_mid = [], [], []
y = 0.0
for gi, (title, slug) in enumerate(MAPS):
    ys_group = []
    for mdl in MODELS:
        ys.append(y); ys_group.append(y); ylabels.append(mdl)
        y -= ROW_H
    group_mid.append(np.mean(ys_group))
    y -= GROUP_GAP

for (title, slug), gm in zip(MAPS, group_mid):
    ax.text(-0.145, gm, title, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=15, color=INK,
            fontweight="bold")

k = 0
for title, slug in MAPS:
    for mdl in MODELS:
        phi, fv = data[mdl][slug]
        pct = fv.argsort().argsort() / (len(fv) - 1)
        jit = rng.normal(0, 0.085, len(phi)).clip(-0.24, 0.24)
        sc = ax.scatter(phi, ys[k] + jit, c=pct, cmap=CMAP_FEATVAL,
                        vmin=0, vmax=1, s=8, linewidths=0, alpha=0.6,
                        rasterized=True)
        ax.plot(np.mean(phi), ys[k], marker="D", color=INK, ms=8, zorder=5)
        k += 1

ax.axvline(0, color=INK2, lw=1.5, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels(ylabels, fontsize=12.5)
ax.tick_params(left=False)
ax.set_xlim(XLO, XHI)
ax.set_ylim(min(ys) - 0.55, max(ys) + 0.55)
ax.set_xlabel("Shapley value (nats; contribution to log-probability of "
              "the true source cell)", fontsize=14)
ax.xaxis.grid(True, color="#e1e0d9", lw=0.9)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, ticks=[0, 1])
cb.ax.set_yticklabels(["low", "high"], fontsize=12)
cb.set_label("feature value (percentile within row)", fontsize=12.5,
             color=INK2)
cb.outline.set_visible(False)

ax.annotate("black diamond = mean;  axis clipped at $\\pm$3",
            xy=(0.99, 0.015), xycoords="axes fraction",
            ha="right", fontsize=11, color=MUTED)

for ext in ("pdf", "png", "svg"):
    fig.savefig(HERE / f"shap_beeswarm_models.{ext}", dpi=220,
                bbox_inches="tight")
print("wrote shap_beeswarm_models.{pdf,png,svg}  xlim [%.2f, %.2f]"
      % (XLO, XHI))
