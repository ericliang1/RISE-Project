"""figM_method.pdf: the physics-guided posterior distillation recipe, drawn
with REAL artifacts (physics maps, teacher, region) from one test scenario."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import load_config, resolve
from train import blur_teacher

C_MUT, C_IMPR, C_EXACT, C_BASE = "#52514e", "#e87ba4", "#008300", "#2a78d6"
plt.rcParams.update({"font.size": 7.5, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
N = 64


def hpd_mask(p, mass=0.90):
    o = np.argsort(-p)
    k = 1 + (np.cumsum(p[o]) < mass).sum()
    m = np.zeros_like(p)
    m[o[:k]] = 1
    return m.reshape(N, N)


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    fd = resolve(cfg, "figures_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    z = np.load(dd / "ch4t_test_suffstats.npz")
    P_ex = np.load(dd / "ch4t_test_posterior.npz")["probs"]
    sizes = np.load(dd / "ch4t_audit_M0.npz")["exact_sizes"]
    i = int(np.argsort(sizes)[len(sizes) // 20])
    a, b, sig = z["a"][i].astype(np.float64), z["b"][i].astype(np.float64), d["sigma"][i]
    zmap = np.clip(a / (sig * np.sqrt(b) + 1e-30), -60, 60)
    logb = np.log10(b + 1e-30)
    teach = blur_teacher(torch.tensor(P_ex[i:i+1].astype(np.float32)), N,
                         0.75, device="cpu").numpy()[0]
    ns = int(d["n_sensors"][i])

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1.15, 1, 1.15],
                          hspace=0.42, wspace=0.28,
                          left=0.02, right=0.98, top=0.86, bottom=0.05)

    def hm(ax, img, cmap, title, gamma=0.5):
        v = img - img.min()
        ax.imshow(v.reshape(N, N), origin="lower", cmap=cmap,
                  norm=PowerNorm(gamma))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7, pad=2)

    # column 1: raw scenario + physics maps
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1], marker="^",
               s=22, c="#0b0b0b")
    for k in range(0, 30, 6):
        u = d["u_seq"][i, k]
        ax.annotate("", xy=(0.5 + u[0] * .55, 0.5 + u[1] * .55),
                    xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#d55181",
                                    lw=0.9, alpha=0.65))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("readings + wind record", fontsize=7, pad=2)
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.72, "known plume physics", ha="center", fontsize=7.5,
            style="italic")
    ax.text(0.5, 0.38, "$a_c=\\sum g_c y$\n$b_c=\\sum g_c^2$", ha="center",
            fontsize=8)
    ax.axis("off")

    hm(fig.add_subplot(gs[0, 1]), zmap, "Purples",
       "match map  $z_c$")
    hm(fig.add_subplot(gs[1, 1]), logb, "Greys", "sensitivity  $\\log b_c$")

    # column 3: encoder + head
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    for y0, txt, col in [(0.66, "set encoder\n(DeepSets / GNN /\nSet Transformer)", C_BASE),
                         (0.16, "conv head\n(zero-init)", C_IMPR)]:
        ax.add_patch(FancyBboxPatch((0.06, y0), 0.88, 0.24,
                                    boxstyle="round,pad=0.02", fc="white",
                                    ec=col, lw=1.4))
        ax.text(0.5, y0 + 0.12, txt, ha="center", va="center", fontsize=7.5)
    ax.annotate("", xy=(0.44, 0.505), xytext=(0.44, 0.655),
                arrowprops=dict(arrowstyle="-|>", color=C_BASE, lw=1.3))
    ax.annotate("", xy=(0.44, 0.465), xytext=(0.44, 0.405),
                arrowprops=dict(arrowstyle="<|-", color=C_IMPR, lw=1.3))
    ax.text(0.44, 0.485, "+", ha="center", va="center", fontsize=9,
            zorder=6, bbox=dict(boxstyle="circle,pad=0.15", fc="white",
                                ec=C_MUT, lw=1.1))
    ax.annotate("logits", xy=(0.92, 0.485), xytext=(0.505, 0.485),
                fontsize=7, color=C_MUT, va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_MUT, lw=1.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # column 4: teacher (training-time), zoomed so the tempered blob is visible
    ax = fig.add_subplot(gs[0, 3])
    pk = int(np.argmax(teach))
    py, px = pk // N, pk % N
    r = 8
    y0, y1 = max(0, py - r), min(N, py + r)
    x0, x1 = max(0, px - r), min(N, px + r)
    ax.imshow(teach.reshape(N, N)[y0:y1, x0:x1], origin="lower",
              cmap="Greens", norm=PowerNorm(0.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("teacher: exact posterior\n(tempered; zoomed)", fontsize=7,
                 pad=2)
    ax = fig.add_subplot(gs[1, 3])
    ax.text(0.5, 0.55, "distill (CE)\ntraining only", ha="center",
            fontsize=7.5, color=C_EXACT)
    ax.annotate("", xy=(0.5, 0.95), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle="-|>", color=C_EXACT, lw=1.3,
                                linestyle="--"))
    ax.axis("off")

    # column 5: the payoff -- median regions to scale on the site
    ax = fig.add_subplot(gs[:, 4])
    sx, sy = 250.0, 290.0     # schematic placement: regions shown to scale
    for radius, col, lab in [(94, C_BASE, "labels only: 94 m"),
                             (46, C_IMPR, "full recipe: 44--49 m"),
                             (7, C_EXACT, "Bayes limit: 7 m")]:
        ax.add_patch(plt.Circle((sx, sy), radius, fill=False, ec=col, lw=1.6))
    ax.scatter([sx], [sy], marker="*", s=120, c="white",
               edgecolors="#0b0b0b", lw=0.9, zorder=5)
    ax.add_patch(plt.Circle((sx, sy), 50, fill=False, ec=C_MUT, lw=0.9,
                            ls=":"))
    ax.text(sx, sy - 108, "EPA 50 m", ha="center", fontsize=6, color=C_MUT)
    hh = [plt.Line2D([], [], color=c, lw=1.6, label=l)
          for _, c, l in [(94, C_BASE, "labels only: 94 m"),
                          (46, C_IMPR, "full recipe: 44--49 m"),
                          (7, C_EXACT, "Bayes limit: 7 m")]]
    ax.legend(handles=hh, loc="lower left", fontsize=6, frameon=False)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("split conformal, 90% coverage\nmedian region, to scale",
                 fontsize=7, pad=2)
    fig.savefig(fd / "figM_method.pdf")
    fig.savefig(fd / "figM_method_prev.png", dpi=160)
    print("figM done")


if __name__ == "__main__":
    main()
