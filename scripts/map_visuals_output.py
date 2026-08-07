"""Companion to the per-map visuals: the model's combined logits and the
resulting probability map, same scenario-selection rule (8 masts, jointly
closest to the no-maps and with-maps median radii), warmed DeepSets seed-1
checkpoint.  Usage: python scripts/map_visuals_output.py
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
from common import load_config, resolve, get_device
from methane_t import N_GRID, to_batch_t
from methane_t_uncertain import PhysHeadNet
from paired_wind import make_view

BLUE = "#2a78d6"
INK, INK2 = "#0b0b0b", "#52514e"
BASE = "#c3c2b7"
CMB = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb",
    "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white"})

cfg = load_config()
dd = resolve(cfg, "data_dir")
device = get_device()
rad = lambda s: np.sqrt(np.asarray(s, float) / 4096 / np.pi) * 500

d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
rb = rad(np.load(dd / "pw_audit_pw_noisy_nomaps_seed1_noisy.npz")["sizes"])
ro = rad(np.load(dd / "pw_audit_pw_noisy_ens_seed1_noisy.npz")["sizes"])
grp = d["n_sensors"].astype(int) == 8
dist = (np.log(rb / np.median(rb[grp])) ** 2
        + np.log(ro / np.median(ro[grp])) ** 2)
dist[~grp] = np.inf
i = int(np.argmin(dist))

ns = int(d["n_sensors"][i])
sx = d["sensors"][i, :ns] * 500
tc = int(d["true_cell"][i])
tx = ((tc % 64 + 0.5) / 64 * 500, (tc // 64 + 0.5) / 64 * 500)

dn, stats = make_view(cfg, dd, ("test", d), "noisy", "ens")
ck = torch.load(dd / "checkpoints" / "pw_noisy_ens_seed1.pt",
                map_location=device, weights_only=False)
model = PhysHeadNet(cfg, 3).to(device).eval()
model.load_state_dict(ck["model_state"])

with torch.no_grad():
    b = to_batch_t(dn, np.array([i]), device)
    b["stats"] = stats[np.array([i])].to(device).view(1, 3, N_GRID, N_GRID)
    logits = model(b).float()[0].cpu().numpy().reshape(64, 64)
    probs = torch.softmax(torch.tensor(logits.ravel()), -1).numpy()
probs2d = probs.reshape(64, 64)


def panel(img, name, label, norm=None, clip=(2, 99)):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    if clip is not None:
        lo, hi = np.percentile(img, clip)
        img = np.clip(img, lo, hi)
    ax.imshow(img, origin="lower", cmap=CMB, extent=[0, 500, 0, 500],
              interpolation="bilinear", norm=norm)
    ax.scatter(sx[:, 0], sx[:, 1], s=46, c=INK, edgecolors="white",
               linewidths=1.5, zorder=5)
    ax.add_patch(Circle(tx, 13, ec=INK, fc="none", lw=2.0, zorder=6))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BASE)
    out = pathlib.Path("figures/paper")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}  ({label})")


panel(logits, "fig_map_logits", "combined logits, base + head")
# probabilities are extremely peaked; show log10(p) so structure is visible
logp = np.log10(probs2d + 1e-12)
panel(np.clip(logp, -8, logp.max()), "fig_map_prob",
      "combined probability map (log10 scale)", clip=None)
print("scenario %d, %d masts; no-maps %.0f m -> with maps %.0f m; "
      "p(true cell) = %.3f, argmax %s truth"
      % (i, ns, rb[i], ro[i], probs2d.ravel()[tc],
         "==" if int(probs.argmax()) == tc else "!="))


# ---- final panel: the conformal region cut from the probability map ----
za = np.load(dd / "pw_audit_pw_noisy_ens_seed1_noisy.npz")
mask = np.unpackbits(za["masks"][i])[:4096].reshape(64, 64).astype(float)
r_m = rad(za["sizes"][i:i + 1])[0]

fig, ax = plt.subplots(figsize=(4.6, 4.6))
lp = np.clip(np.log10(probs2d + 1e-12), -8, None)
ax.imshow(lp, origin="lower", cmap=CMB, extent=[0, 500, 0, 500],
          interpolation="bilinear", alpha=0.45)
ax.imshow(np.where(mask > 0, 1.0, np.nan), origin="lower",
          extent=[0, 500, 0, 500], interpolation="nearest",
          cmap=LinearSegmentedColormap.from_list("r", ["#9ec5f4",
                                                       "#9ec5f4"]),
          alpha=0.55, zorder=2)
ax.contour(mask, levels=[0.5], extent=[0, 500, 0, 500], colors=[BLUE],
           linewidths=2.6, zorder=3)
ax.scatter(sx[:, 0], sx[:, 1], s=46, c=INK, edgecolors="white",
           linewidths=1.5, zorder=5)
ax.add_patch(Circle(tx, 13, ec=INK, fc="none", lw=2.0, zorder=6))
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color(BASE)
out = pathlib.Path("figures/paper")
fig.savefig(out / "fig_map_region.pdf", bbox_inches="tight")
fig.savefig(out / "fig_map_region.png", dpi=260, bbox_inches="tight")
plt.close(fig)
print("wrote fig_map_region  (90%% conformal region, %.0f m equivalent "
      "radius, covered=%s)" % (r_m, bool(za["covered"][i])))
