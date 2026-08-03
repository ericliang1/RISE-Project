"""Poster figure 08: what "Monte Carlo" means in this pipeline.

The anemometer's wind is not the wind the plume felt.  The Bayes-correct
evidence for a leak at cell c integrates over every wind the true one
could have been,

    p(y | c) = E_{u ~ p(u | u_hat)} p(y | c, u)
             ~ (1/K) sum_k p(y | c, u_k),      u_k ~ p(u | u_hat),

and the pipeline approximates that integral the Monte Carlo way: draw
K = 8 plausible winds from the stated anemometer error model (10 deg
direction, 10% speed, iid per step) and average.  The same K = 8 budget
is used twice — the network's ensemble input channels (mean evidence and
wind spread over its 8 map draws, RNG stream [root, 92, ...]) and the
physics-only marginalized oracle (log-mean-exp of the 8 per-draw
likelihoods, stream [root, 91, ...]).

The figure shows the machinery on the SAME held-out example scenario as
figs 01/01.5, with the frozen RNG streams, so every panel is real data:

  A. the measured wind record and the eight Monte-Carlo wind draws;
  B. the wind-spread input channel (3 x std of evidence over the draws)
     with each draw's maximum-likelihood cell overlaid — on the
     super-emitter population every single-wind posterior is a
     near-delta, so the draws appear as eight confident point answers;
  C. zoom: the eight answers scatter 11-63 m around the true source —
     the Monte-Carlo scatter IS the wind-uncertainty the deterministic
     answer hides;
  D. plain-language explainer with the estimator.

The per-draw answers are recomputed at figure time with the frozen
streams and the recomputed marginal is checked against the stored
`ch4tu_test_oracle_marg.npz` (exact match is printed).

Usage:
  source scripts/env.sh && python "Poster Graphics v3/generate_monte_carlo_figure.py"
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generate_poster_graphics_v3 as pg  # house style + helpers
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import methane_t_uncertain as mtu

BLUE, ORANGE, INK, INK2, MUTED = pg.BLUE, pg.ORANGE, pg.INK, pg.INK2, pg.MUTED
GRID, BASE, SURF = pg.GRID, pg.BASE, pg.SURF
SITE = pg.SITE
CELL = SITE / 64.0
STAR = "#f2c230"


def per_draw_answers(d, i):
    """Recompute the K=8 marginalized-oracle likelihoods for scenario i
    with the frozen RNG streams; return wind draws, per-draw argmax cells,
    the deterministic argmax, and the recomputed marginal posterior."""
    u_obs = np.load(pg.DATA / "ch4tu_test_uobs.npy")[i]
    cfg = mtu.load_config()
    root = cfg["seeds"]["root_entropy"]
    tag = mtu.split_tag("test")
    cfg_q = mtu.ch4_cfg(cfg)
    device = torch.device("cpu")
    cells = torch.tensor(mtu.cell_centers(mtu.N_GRID), device=device,
                         dtype=torch.float64)
    ns = int(d["n_sensors"][i])
    keep = d["keep"][i, :ns]
    y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                     dtype=torch.float64)
    yy = float(torch.dot(y, y))
    M = y.shape[0]
    sig = float(d["sigma"][i])

    def evidence(u_seq):
        g = mtu.cell_responses_t(cells, d["sensors"][i, :ns], u_seq,
                                 mtu.stab_of(d, i), device)
        g = g[:, torch.tensor(keep, device=device)]
        a = g @ y
        b = (g * g).sum(1)
        return mtu.marginal_log_evidence(a, b, yy, sig, M, cfg_q, device)

    draws = [mtu.perturb_wind(u_obs, np.random.default_rng(
        [root, 91, tag, i, j])) for j in range(mtu.K_TEACHER)]
    lps = torch.stack([evidence(u) for u in draws])
    lm = torch.logsumexp(lps, 0) - np.log(mtu.K_TEACHER)
    P_marg = torch.exp(lm - torch.logsumexp(lm, 0)).numpy()
    frozen = np.load(pg.DATA / "ch4tu_test_oracle_marg.npz")["probs"][i]
    err = np.abs(P_marg - frozen).max()
    print(f"    recomputed marginal vs frozen oracle_marg: "
          f"max abs diff {err:.2e}")
    det_cell = int(evidence(u_obs).argmax())
    return (u_obs, draws, [int(l.argmax()) for l in lps], det_cell,
            int(frozen.argmax()))


def cell_xy(c):
    return ((c % 64 + 0.5) * CELL, (c // 64 + 0.5) * CELL)


def fig_monte_carlo(d):
    i = pg.pick_scenario(d)  # same scenario as figs 01 / 01.5
    ns = int(d["n_sensors"][i])
    sx = d["sensors"][i, :ns] * SITE
    tx = cell_xy(int(d["true_cell"][i]))
    u_obs, draws, cells_k, det_cell, marg_cell = per_draw_answers(d, i)
    pts = np.array([cell_xy(c) for c in cells_k])
    dxy = cell_xy(det_cell)
    dists = np.linalg.norm(pts - np.array(tx), axis=1)
    print(f"    per-draw answer distances to truth: "
          f"{np.array2string(np.sort(dists), precision=0)} m; "
          f"deterministic {np.linalg.norm(np.array(dxy) - tx):.0f} m")

    med_b = np.median(pg.pooled_radii("nomaps"))
    med_e = np.median(pg.pooled_radii(pg.MAPS_TAG))

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 12.6))
    axA, axB, axC, axD = axes.flat

    # ---------------- A. the measured wind and the eight draws
    def deg(u):
        return np.degrees(np.unwrap(np.arctan2(u[:, 1], u[:, 0])))

    t = np.arange(mtu.T)
    ref = deg(u_obs)
    for j, u in enumerate(draws):
        th = deg(u)
        th -= 360.0 * np.round((th.mean() - ref.mean()) / 360.0)
        axA.plot(t, th, color=BLUE, lw=1.3, alpha=0.75, zorder=3)
    axA.plot(t, ref, color=INK, lw=3.0, zorder=5)
    axA.annotate("measured wind $\\hat{u}$", xy=(t[-1], ref[-1]),
                 xytext=(6, 0), textcoords="offset points", fontsize=13.5,
                 color=INK, va="center", fontweight="bold")
    j_hi = int(np.argmax([deg(u)[-1] for u in draws]))
    axA.annotate("8 Monte-Carlo draws\n$u_k \\sim p(u\\,|\\,\\hat{u})$",
                 xy=(t[-1], deg(draws[j_hi])[-1]), xytext=(6, 14),
                 textcoords="offset points", fontsize=13.5, color=BLUE,
                 va="center", linespacing=1.3)
    axA.set_xlim(0, mtu.T * 1.28)
    axA.set_xlabel("time step (30-step record)", fontsize=13.5)
    axA.set_ylabel("wind direction (deg)", fontsize=13.5)
    axA.yaxis.grid(True, color=GRID, lw=0.9)
    axA.set_axisbelow(True)
    pg.despine(axA)
    axA.tick_params(labelsize=12)
    axA.set_title("A. One measured wind, eight simulated truths",
                  fontsize=17, fontweight="bold", loc="left", pad=9)
    axA.annotate("each draw re-perturbs every step by the anemometer's "
                 "stated error\n(10° direction, 10% speed) — a sample from "
                 "what the true wind could be",
                 xy=(0, -0.14), xycoords="axes fraction", fontsize=13,
                 color=MUTED, va="top", linespacing=1.35)

    # ---------------- B. the spread channel, with the eight point answers
    ens = np.load(pg.DATA / "ch4tu_test_maps_ens.npz")["maps"][i]
    m = ens[1].reshape(64, 64)
    lo, hi = np.percentile(m, [2, 99])
    axB.imshow(np.clip(m, lo, hi), origin="lower", cmap=pg.CMO,
               extent=[0, SITE, 0, SITE], interpolation="bilinear")
    axB.scatter(sx[:, 0], sx[:, 1], s=52, c=INK, edgecolors="white",
                linewidths=1.5, zorder=5)
    axB.scatter(pts[:, 0], pts[:, 1], s=64, c=BLUE, edgecolors="white",
                linewidths=1.4, zorder=6)
    axB.plot(*tx, marker="*", color=STAR, ms=24, mec=INK, mew=1.3, zorder=8)
    axB.set_xlim(0, SITE)
    axB.set_ylim(0, SITE)
    axB.set_aspect("equal")
    axB.set_xticks([])
    axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_color(BASE)
        sp.set_linewidth(1.2)
    axB.set_title("B. Each draw pins the leak somewhere else",
                  fontsize=17, fontweight="bold", loc="left", pad=9)
    axB.annotate("dots: the single most likely cell under each wind draw — "
                 "every one a\nnear-certain posterior.  Backdrop: the "
                 "wind-spread input channel, the\nsame Monte-Carlo "
                 "disagreement handed to the network as a map",
                 xy=(0, -0.035), xycoords="axes fraction", fontsize=13,
                 color=MUTED, va="top", linespacing=1.35)

    # ---------------- C. zoom on the scatter
    allp = np.vstack([pts, dxy, tx])
    c0 = allp.mean(0)
    half = max(90.0, np.abs(allp - c0).max() + 36.0)
    x0, x1 = c0[0] - half, c0[0] + half
    y0, y1 = c0[1] - half, c0[1] + half
    axC.set_facecolor("white")
    for g in np.arange(np.floor(x0 / CELL) * CELL, x1 + CELL, CELL):
        axC.axvline(g, color=GRID, lw=0.6, zorder=1)
    for g in np.arange(np.floor(y0 / CELL) * CELL, y1 + CELL, CELL):
        axC.axhline(g, color=GRID, lw=0.6, zorder=1)
    axC.add_patch(Circle(tx, 50, fc="none", ec=MUTED, lw=1.8,
                         ls=(0, (4, 3)), zorder=3))
    for c in cells_k:
        x, y = cell_xy(c)
        axC.add_patch(Rectangle((x - CELL / 2, y - CELL / 2), CELL, CELL,
                                fc=pg.BLUE_FILL, ec=BLUE, lw=1.6, zorder=4))
    axC.add_patch(Rectangle((dxy[0] - CELL / 2, dxy[1] - CELL / 2), CELL,
                            CELL, fc=pg.ORANGE_FILL, ec=ORANGE, lw=1.8,
                            zorder=5))
    mx, my = cell_xy(marg_cell)
    axC.add_patch(Circle((mx, my), CELL * 0.95, fc="none", ec=pg.BLUE_D,
                         lw=2.2, zorder=6))
    axC.annotate("evidence-weighted pick of\nthe marginalized oracle",
                 xy=(mx + CELL, my), xytext=(0.97, 0.05),
                 textcoords="axes fraction", ha="right", fontsize=12.5,
                 color=pg.BLUE_D, linespacing=1.3,
                 arrowprops=dict(arrowstyle="-|>", color=pg.BLUE_D, lw=1.6,
                                 shrinkA=4,
                                 connectionstyle="arc3,rad=0.25"))
    axC.plot(*tx, marker="*", color=STAR, ms=26, mec=INK, mew=1.3, zorder=8)
    axC.plot([x0 + 14, x0 + 64], [y0 + 13, y0 + 13], color=INK2, lw=2.6,
             zorder=7, solid_capstyle="butt")
    axC.annotate("50 m", xy=(x0 + 39, y0 + 19), ha="center", fontsize=12,
                 color=INK2)
    axC.set_xlim(x0, x1)
    axC.set_ylim(y0, y1)
    axC.set_aspect("equal")
    axC.set_xticks([])
    axC.set_yticks([])
    for sp in axC.spines.values():
        sp.set_color(BASE)
        sp.set_linewidth(1.2)
    axC.set_title("C. The scatter of the answers is the real uncertainty",
                  fontsize=17, fontweight="bold", loc="left", pad=9)
    axC.annotate("zoom near the source: eight 7.8 m cells, "
                 f"{dists.min():.0f}–{dists.max():.0f} m from the truth.  "
                 "One wind gives false\ncertainty; eight draws reveal the "
                 "wind-limited resolution the regions must cover",
                 xy=(0, -0.035), xycoords="axes fraction", fontsize=13,
                 color=MUTED, va="top", linespacing=1.35)

    # ---------------- D. explainer
    axD.axis("off")
    axD.text(0, 0.98, "What “Monte Carlo” means",
             fontsize=17, fontweight="bold", va="top", color=INK)
    axD.text(0, 0.89,
             "Estimate an integral you cannot solve by averaging over\n"
             "random samples of the unknown.  Here the unknown is the\n"
             "true wind behind the measured record:",
             fontsize=13.8, color=INK2, va="top", linespacing=1.5)
    axD.text(0.5, 0.68,
             "$p(y\\,|\\,c)\\;=\\;\\mathbb{E}_{u\\sim p(u|\\hat{u})}\\,"
             "p(y\\,|\\,c,u)\\;\\approx\\;\\frac{1}{K}\\sum_{k=1}^{K}"
             "p(y\\,|\\,c,u_k)$",
             fontsize=15.5, color=INK, ha="center", va="top")
    axD.text(0, 0.52,
             "The pipeline runs this estimator twice, at the same K = 8\n"
             "budget so baseline and method are comparable:\n\n"
             "•  input maps — mean (consensus) and spread (disagreement)\n"
             "    of the evidence over the network's 8 draws;\n"
             "•  physics-only oracle — average of the 8 per-draw\n"
             "    likelihoods, weighting each draw's answer by its\n"
             "    evidence (panel C, ring).",
             fontsize=13.8, color=INK2, va="top", linespacing=1.5)
    axD.text(0, 0.0,
             f"Payoff (2,000 scenarios × 3 seeds): the Monte-Carlo map\n"
             f"channels shrink the median 90% region from {med_b:.0f} m to "
             f"{med_e:.0f} m.",
             fontsize=13.8, color=INK, va="bottom", linespacing=1.5,
             fontweight="bold")

    h = [plt.Line2D([], [], color=INK, marker="o", lw=0, ms=9, mec="white",
                    mew=1.5, label=f"sensor masts ({ns})"),
         plt.Line2D([], [], color=STAR, marker="*", lw=0, ms=15, mec=INK,
                    mew=1.1, label="true methane source"),
         plt.Line2D([], [], color=BLUE, marker="s", lw=0, ms=10,
                    mfc=pg.BLUE_FILL, label="a wind draw's answer"),
         plt.Line2D([], [], color=ORANGE, marker="s", lw=0, ms=10,
                    mfc=pg.ORANGE_FILL, label="measured-wind answer"),
         plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 3)),
                    label="50 m facility scale")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False,
               fontsize=13.5, bbox_to_anchor=(0.5, -0.022))
    fig.suptitle("Monte Carlo: eight winds stand in for the one we "
                 "couldn't measure", fontsize=24, fontweight="bold",
                 x=0.055, ha="left", y=1.005)
    fig.text(0.055, 0.968, "the wind-marginalization machinery on the "
             f"held-out example scenario of figs 01/01.5 ({ns} masts, "
             f"{d['q'][i]:.0f} kg h$^{{-1}}$); identical RNG streams to "
             "the frozen dataset, recomputed marginal checked against the "
             "stored oracle", fontsize=14, color=MUTED, ha="left")
    fig.subplots_adjust(wspace=0.14, hspace=0.30, top=0.925, bottom=0.06,
                        left=0.055, right=0.97)
    pg.save(fig, "08_monte_carlo_wind")


if __name__ == "__main__":
    d = pg.load_test()
    print("figure 08 (monte carlo wind marginalization)")
    fig_monte_carlo(d)
    print("monte carlo figure done")
