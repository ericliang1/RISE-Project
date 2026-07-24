"""figP_pipeline.pdf: methodology figure, IEEE full-width (7.1 x 3.1 in).

Two visual bands carry the method's core claim structurally:
  TOP  (gray)  = the inference path, runs in milliseconds at deployment
  BOTTOM (green) = training-only supervision (exact posterior -> temper -> CE)
Real artifacts throughout (scenario, matched-filter map, teacher, measured
radii)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import load_config, resolve
from train import blur_teacher

C_BLUE, C_MAG, C_EXACT = "#2a78d6", "#c04d7c", "#008300"
C_INK, C_MUT = "#0b0b0b", "#52514e"
BAND_TOP, BAND_BOT = "#f4f3f1", "#eef5ec"
N = 64
plt.rcParams.update({"font.size": 7, "figure.dpi": 150,
                     "savefig.bbox": "tight"})


def main():
    cfg = load_config()
    dd = resolve(cfg, "data_dir")
    fd = resolve(cfg, "figures_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    z = np.load(dd / "ch4t_test_suffstats.npz")
    P_ex = np.load(dd / "ch4t_test_posterior.npz")["probs"]
    sizes = np.load(dd / "ch4t_audit_M0.npz")["exact_sizes"]
    i = int(np.argsort(sizes)[len(sizes) // 20])
    a, b = z["a"][i].astype(np.float64), z["b"][i].astype(np.float64)
    zmap = np.clip(a / (d["sigma"][i] * np.sqrt(b) + 1e-30), -60, 60)
    raw = P_ex[i]
    teach = blur_teacher(torch.tensor(P_ex[i:i+1].astype(np.float32)), N,
                         0.75, device="cpu").numpy()[0]
    ns = int(d["n_sensors"][i])

    fig = plt.figure(figsize=(7.1, 3.05))

    # ---- bands on a background axes (created FIRST: panels layer above)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.add_patch(Rectangle((0.005, 0.40), 0.99, 0.575, fc=BAND_TOP,
                           ec="none"))
    bg.add_patch(Rectangle((0.245, 0.015), 0.50, 0.36, fc=BAND_BOT,
                           ec=C_EXACT, lw=0.8, ls=(0, (4, 3))))
    bg.text(0.012, 0.905, "INFERENCE\n(test time, ms)",
            fontsize=7, color=C_MUT, weight="bold", va="top")
    bg.text(0.253, 0.030, "TRAINING ONLY — physics-posterior supervision",
            fontsize=7.5, color=C_EXACT, weight="bold")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    def panel(rect):
        ax = fig.add_axes(rect)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(C_MUT)
        return ax

    PW, PY, PH = 0.15, 0.445, 0.36    # top-band panel geometry

    # 1 observed scenario
    ax = panel([0.115, PY, PW, PH])
    ax.scatter(d["sensors"][i, :ns, 0], d["sensors"][i, :ns, 1], marker="^",
               s=18, c=C_INK)
    for k in range(0, 30, 6):
        u = d["u_seq"][i, k]
        ax.annotate("", xy=(0.5 + u[0] * .5, 0.5 + u[1] * .5),
                    xytext=(0.5, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#d55181",
                                    lw=0.8, alpha=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("readings $y$ + wind $\\hat{u}$", fontsize=7, pad=2)

    # 2 physics maps
    ax = panel([0.29, PY, PW, PH])
    ax.imshow((zmap - zmap.min()).reshape(N, N), origin="lower",
              cmap="Purples", norm=PowerNorm(0.5), aspect="auto")
    ax.set_title("matched-filter maps\n$z, \\log b$   (closed form)",
                 fontsize=7, pad=2)

    # 3 model
    ax = fig.add_axes([0.485, PY - 0.02, 0.19, PH + 0.04])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.56), 0.80, 0.36,
                                boxstyle="round,pad=0.02", fc="white",
                                ec=C_BLUE, lw=1.3))
    ax.text(0.43, 0.75, "any localizer\nDeepSets / GNN / SetTr.",
            ha="center", va="center", fontsize=6.8)
    ax.add_patch(FancyBboxPatch((0.03, 0.08), 0.80, 0.36,
                                boxstyle="round,pad=0.02", fc="white",
                                ec=C_MAG, lw=1.3))
    ax.text(0.43, 0.27, "zero-init conv head\n(+0.2% params)",
            ha="center", va="center", fontsize=6.8)
    ax.annotate("", xy=(0.92, 0.50), xytext=(0.83, 0.74),
                arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=1.2))
    ax.annotate("", xy=(0.92, 0.50), xytext=(0.83, 0.26),
                arrowprops=dict(arrowstyle="-", color=C_MAG, lw=1.2))
    ax.text(0.92, 0.50, "+", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=C_MUT,
                      lw=1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("model (logits)", fontsize=7, pad=0)

    # 4 conformal region
    ax = panel([0.72, PY, PW, PH])
    sx, sy = 250, 270
    for radius, c in [(94, C_BLUE), (46, C_MAG), (7, C_EXACT)]:
        ax.add_patch(plt.Circle((sx, sy), radius, fill=False, ec=c, lw=1.5))
    ax.add_patch(plt.Circle((sx, sy), 50, fill=False, ec=C_MUT, lw=0.8,
                            ls=":"))
    ax.scatter([sx], [sy], marker="*", s=80, c="white", edgecolors=C_INK,
               lw=0.8, zorder=5)
    ax.set_xlim(0, 500); ax.set_ylim(0, 500); ax.set_aspect("equal")
    ax.set_title("conformal 90% region", fontsize=7, pad=2)
    ax.text(0.985, 0.03, "94 m $\\to$ 44–49 m\n(EPA: 50 m)",
            transform=ax.transAxes, ha="right", fontsize=6.3, color=C_MUT)

    # ---- training band: raw posterior -> temper -> CE ------------------
    pk = int(np.argmax(teach))
    py_, px_ = pk // N, pk % N
    r = 8
    sl = np.s_[max(0, py_-r):py_+r, max(0, px_-r):px_+r]
    ax = panel([0.33, 0.075, 0.11, 0.20])
    ax.imshow(raw.reshape(N, N)[sl], origin="lower", cmap="Greens",
              norm=PowerNorm(0.4), aspect="auto")
    ax.set_title("exact posterior\n(near-delta)", fontsize=6.5, pad=2,
                 color=C_EXACT)
    ax = panel([0.53, 0.075, 0.11, 0.20])
    ax.imshow(teach.reshape(N, N)[sl], origin="lower", cmap="Greens",
              norm=PowerNorm(0.5), aspect="auto")
    ax.set_title("tempered 0.75 cells\n(raw fails: 134 m)", fontsize=6.5,
                 pad=2, color=C_EXACT)

    arr = dict(transform=fig.transFigure, color=C_MUT, lw=1.3,
               arrowstyle="-|>", mutation_scale=11)
    for x0, x1, y in [(0.255, 0.285, 0.625), (0.445, 0.48, 0.625),
                      (0.68, 0.715, 0.625)]:
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), **arr))
    fig.patches.append(FancyArrowPatch((0.445, 0.175), (0.525, 0.175), **arr))
    fig.patches.append(FancyArrowPatch(
        (0.60, 0.31), (0.578, 0.425), transform=fig.transFigure,
        color=C_EXACT, lw=1.4, ls=(0, (4, 2)), arrowstyle="-|>",
        mutation_scale=11))
    fig.text(0.665, 0.335, "cross-entropy", fontsize=6.5, color=C_EXACT)

    fig.savefig(fd / "figP_pipeline.pdf")
    fig.savefig(fd / "figP_pipeline_prev.png", dpi=160)
    print("figP v2 done")


if __name__ == "__main__":
    main()
