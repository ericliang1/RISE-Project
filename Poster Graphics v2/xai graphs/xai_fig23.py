"""XAI Figures 2 & 3 — embedding projection and integrated gradients,
from the real retrained paper checkpoint (see xai_fig1.py).

Figure 2: the set-network embedding (the input to model.decoder — the
concatenated mean-pool / max-pool / context vector, captured with a forward
pre-hook) for all 2,000 seed-1 test scenarios, projected with UMAP
(n_neighbors=30, min_dist=0.1, random_state=1) AND PCA.  Two panels:
coloured by achieved 90% region radius and by mast count (the stability
panel was dropped — silhouette ~0, uniform blob).  The structure metrics
(silhouette of a radius split, Spearman of axis 1 vs masts / log radius,
and the within-mast-count correlation) are still computed, printed, and
quoted on the figure.  Note: mast count is an input to the context MLP, so
the embedding is expected to encode it.

Figure 3: integrated gradients of the predicted (argmax) cell's log-prob
w.r.t. the four input feature maps for the Figure-1 scenario
(ch4t-test-001811).  Baseline = zero maps, 50 midpoint steps; the raw
sensor tokens are held fixed, so this attributes the maps pathway.  The
completeness identity (sum of IG = logp(full) - logp(baseline)) is printed.
Panels follow the canonical map order shared with Figure 1 via
xai_style.MAP_ORDER.

Usage (from repo root):
  python "Poster Graphics v2/xai graphs/xai_fig23.py"
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from common import load_config, get_device, resolve            # noqa: E402
import paired_wind as pw                                       # noqa: E402
from methane_t import N_GRID, to_batch_t                       # noqa: E402
from methane_t_uncertain import PhysHeadNet                    # noqa: E402
from conformal import regions, tail_scores, tail_threshold     # noqa: E402
from xai_fig1 import forward_probs                             # noqa: E402
from xai_style import (BLUE, CMAP_COUNT, CMAP_DIV, CMAP_MAG,   # noqa: E402
                       CMAP_MAG_R, INK2, MAP_ORDER, MUTED, ORANGE,
                       apply_style, draw_argmax, draw_masts, draw_source,
                       frame, header, save_all)

N_CELLS = N_GRID * N_GRID
SITE = 500.0

apply_style()

FIG1_SCENARIO = 1811   # ch4t-test-001811, fixed in xai_fig1.py


def load_model_and_views():
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    ckpt = torch.load(dd / "checkpoints" / "pw_noisy_ensr_seed1.pt",
                      map_location=device, weights_only=False)
    model = PhysHeadNet(cfg, 4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    views = {}
    for split in ("calib", "test"):
        d_clean = dict(np.load(dd / f"ch4t_{split}.npz", allow_pickle=True))
        views[split] = pw.make_view(cfg, dd, (split, d_clean), "noisy",
                                    "ensr")
    return cfg, device, model, views


def embeddings_and_radii(cfg, device, model, views):
    d_cal, st_cal = views["calib"]
    d_tst, st_tst = views["test"]
    feats = []
    hook = model.decoder.register_forward_pre_hook(
        lambda mod, inp: feats.append(inp[0].detach().float().cpu()))
    Pc = forward_probs(model, d_cal, st_cal, device)
    n_cal = len(d_cal["ids"])
    Pt = forward_probs(model, d_tst, st_tst, device)
    hook.remove()
    emb = torch.cat(feats).numpy()
    emb_t = emb[n_cal:]                       # test rows, in order
    assert emb_t.shape[0] == len(d_tst["ids"])
    rngc = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(Pc, d_cal["true_cell"], rngc),
                        cfg["conformal"]["alpha"])
    reg = regions(Pt, th, d_tst["true_cell"])
    radii = np.sqrt(reg["sizes"] / N_CELLS / np.pi) * SITE
    print(f"embedding dim {emb_t.shape[1]}, test rows {emb_t.shape[0]}, "
          f"threshold {th:.6e}")
    return d_tst, st_tst, Pt, emb_t, radii


# ------------------------------------------------------------- Figure 2
def fig2(d_tst, st_tst, emb, radii):
    import umap
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from scipy.stats import spearmanr

    with np.errstate(all="ignore"):
        proj_umap = umap.UMAP(n_neighbors=30, min_dist=0.1,
                              random_state=1).fit_transform(emb)
    proj_pca = PCA(n_components=2, random_state=1).fit_transform(emb)
    np.savez(HERE / "fig2_embeddings.npz", embedding=emb, umap=proj_umap,
             pca=proj_pca, radii=radii,
             n_sensors=d_tst["n_sensors"].astype(int),
             stab=np.round(np.log(d_tst["D"]) * 5).astype(int))

    logr = np.log(radii)
    split = radii > np.median(radii)
    masts = d_tst["n_sensors"].astype(int)
    stats = {}
    for name, pr in (("UMAP", proj_umap), ("PCA", proj_pca)):
        sil = silhouette_score(pr, split)
        r1 = spearmanr(pr[:, 0], logr).statistic
        r2 = spearmanr(pr[:, 1], logr).statistic
        rm = spearmanr(pr[:, 0], masts).statistic
        strat = np.mean([spearmanr(pr[masts == k, 0],
                                   logr[masts == k]).statistic
                         for k in range(4, 13) if (masts == k).sum() > 30])
        stats[name] = (sil, r1, r2, rm, strat)
        print(f"{name}: silhouette(radius>median split) = {sil:.3f}; "
              f"Spearman(axis1, log r) = {r1:+.3f}, "
              f"(axis2, log r) = {r2:+.3f}, (axis1, masts) = {rm:+.3f}, "
              f"(axis1, log r | within mast count) = {strat:+.3f}")
    print("VERDICT: the embedding organizes strongly by mast count (a "
          "context input); radius structure is weak and mostly mediated "
          "by mast count — presented as such on the figure.")

    for name, pr in (("umap", proj_umap), ("pca", proj_pca)):
        sil, r1, r2, rm, strat = stats[name.upper()]
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.2))
        specs = [
            ("A. 90% region radius", radii, CMAP_MAG_R,
             LogNorm(vmin=30, vmax=300), "region radius (m), log"),
            ("B. Number of masts", masts, CMAP_COUNT, None, "masts"),
        ]
        for ax, (title, c, cmap, norm, cbl) in zip(axes, specs):
            sc = ax.scatter(pr[:, 0], pr[:, 1], c=c, cmap=cmap, norm=norm,
                            s=15, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_title(title, loc="left", pad=9)
            frame(ax)
            cb = fig.colorbar(sc, ax=ax, orientation="horizontal",
                              fraction=0.05, pad=0.04, aspect=30)
            if norm is not None:
                cb.set_ticks([50, 100, 200])
                cb.set_ticklabels(["50", "100", "200"])
            cb.set_label(cbl, fontsize=13, color=INK2)
            cb.ax.tick_params(labelsize=12)
            cb.outline.set_visible(False)

        # thumbnails of the source-evidence map for extreme-radius points:
        # white inner pad + white outer halo so they detach from the
        # scatter; borders match the smallest/largest legend colours;
        # overlapping picks are nudged apart (greedy, in data units)
        import matplotlib.patheffects as pe
        ev = st_tst[:, 0, :].numpy()
        order = np.argsort(radii)
        span = np.ptp(pr, axis=0)
        placed = []
        for rank, col in [(order[:3], BLUE), (order[-3:], ORANGE)]:
            for j in rank:
                m = ev[j].reshape(N_GRID, N_GRID)
                lo, hi = np.percentile(m, [2, 99])
                xy = np.array([pr[j, 0], pr[j, 1]], float)
                for _ in range(12):
                    if all(np.any(np.abs(xy - q) > 0.135 * span)
                           for q in placed):
                        break
                    xy[0] += 0.05 * span[0]
                placed.append(xy.copy())
                im = OffsetImage(np.clip(m, lo, hi), zoom=0.62,
                                 cmap=CMAP_MAG, origin="lower")
                ab = AnnotationBbox(
                    im, tuple(xy), frameon=True, pad=0.55,
                    bboxprops=dict(edgecolor=col, facecolor="white",
                                   linewidth=2.5))
                ab.patch.set_path_effects(
                    [pe.withStroke(linewidth=7, foreground="white")])
                axes[0].add_artist(ab)
        hthumb = [plt.Line2D([], [], marker="s", lw=0, ms=11, mfc="white",
                             mec=BLUE, mew=2.2,
                             label="3 smallest regions (evidence map)"),
                  plt.Line2D([], [], marker="s", lw=0, ms=11, mfc="white",
                             mec=ORANGE, mew=2.2,
                             label="3 largest regions (evidence map)")]
        axes[0].legend(handles=hthumb, loc="lower left", frameon=True,
                       fontsize=12.5, facecolor="white", framealpha=0.9,
                       edgecolor="none")

        header(fig, "The embedding tracks mast count — an input — "
               "not difficulty",
               f"{name.upper()} of the decoder-input embedding, DeepSets "
               f"ensr seed 1, 2,000 scenarios · silhouette {sil:.2f} · "
               f"Spearman axis 1: masts {rm:+.2f}, log radius {r1:+.2f} "
               f"({strat:+.2f} within a mast count)",
               ty=1.052, sy=1.014)
        save_all(fig, str(HERE / f"fig2_embedding_{name}"))
        plt.close(fig)


# ------------------------------------------------------------- Figure 3
def fmt_sig(v):
    """Signed 2-dp value, but never '-0.00': round first, sign only when
    the rounded magnitude is >= 0.005."""
    r = round(float(v), 2)
    return "0.00" if abs(r) < 0.005 else f"{r:+.2f}"


def fig3(cfg, device, model, d_tst, st_tst, Pt, radii, steps=50):
    i = FIG1_SCENARIO
    ns = int(d_tst["n_sensors"][i])
    sx = d_tst["sensors"][i, :ns] * SITE
    tc = int(d_tst["true_cell"][i])
    tx = ((tc % N_GRID + 0.5) / N_GRID * SITE,
          (tc // N_GRID + 0.5) / N_GRID * SITE)
    am = int(Pt[i].argmax())
    ax_ = ((am % N_GRID + 0.5) / N_GRID * SITE,
           (am // N_GRID + 0.5) / N_GRID * SITE)

    s_full = st_tst[i].to(device).view(1, 4, N_GRID, N_GRID).float()
    alphas = ((torch.arange(steps, device=device) + 0.5) / steps)
    stats = (alphas[:, None, None, None, None] * s_full).view(
        steps, 4, N_GRID, N_GRID).requires_grad_(True)
    b = to_batch_t(d_tst, np.full(steps, i), device)
    b["stats"] = stats
    logp = torch.log_softmax(model(b).float(), -1)[:, am]
    logp.sum().backward()
    ig = (s_full[0] * stats.grad.mean(0)).detach().cpu().numpy()

    with torch.no_grad():
        for a, nm in ((s_full, "full"), (torch.zeros_like(s_full), "zero")):
            bb = to_batch_t(d_tst, np.array([i]), device)
            bb["stats"] = a
            lp = float(torch.log_softmax(model(bb).float(), -1)[0, am])
            print(f"logp[argmax cell] with {nm} maps: {lp:.4f}")
            if nm == "full":
                lp_full = lp
            else:
                lp_zero = lp
    print(f"completeness: sum(IG) = {ig.sum():+.4f}  vs  "
          f"logp(full) - logp(zero) = {lp_full - lp_zero:+.4f}  "
          f"({steps} midpoint steps, zero-map baseline)")

    # symmetric colour limits: 99th percentile of |IG| across all four maps
    vmax = float(np.percentile(np.abs(ig), 99))
    ext = [0, SITE, 0, SITE]
    fig, axes = plt.subplots(1, 4, figsize=(16.6, 5.4))
    for ax, (title, ch) in zip(axes, MAP_ORDER):
        im = ax.imshow(ig[ch], origin="lower", cmap=CMAP_DIV, vmin=-vmax,
                       vmax=vmax, extent=ext, interpolation="bilinear")
        draw_masts(ax, sx)
        draw_source(ax, tx)
        draw_argmax(ax, ax_)
        ax.set_title(f"{title}\n$\\Sigma$IG = {fmt_sig(ig[ch].sum())}",
                     fontsize=16.5, linespacing=1.2, pad=8)
        frame(ax)
    cb = fig.colorbar(im, ax=axes, fraction=0.023, pad=0.015, shrink=0.70)
    cb.set_label("IG of log $\\hat{p}$(argmax cell)", fontsize=13.5,
                 color=INK2)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)

    header(fig, "Which physics map drove the decision? Integrated "
           "gradients on the Figure-1 scenario",
           f"scenario ch4t-test-{FIG1_SCENARIO:06d} ({ns} masts) · red = "
           "toward the predicted cell, blue = away · zero-map baseline, "
           f"{steps} midpoint steps · completeness |error| "
           f"{abs(ig.sum() - (lp_full - lp_zero)):.3f} nats",
           ty=1.085, sy=1.028)
    save_all(fig, str(HERE / "fig3_integrated_gradients"))
    plt.close(fig)


def main():
    torch.manual_seed(0)
    cfg, device, model, views = load_model_and_views()
    d_tst, st_tst, Pt, emb, radii = embeddings_and_radii(cfg, device, model,
                                                         views)
    fig2(d_tst, st_tst, emb, radii)
    fig3(cfg, device, model, d_tst, st_tst, Pt, radii)


if __name__ == "__main__":
    main()
