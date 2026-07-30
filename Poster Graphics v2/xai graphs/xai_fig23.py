"""XAI Figures 2 & 3 — embedding projection and integrated gradients,
from the real retrained paper checkpoint (see xai_fig1.py).

Figure 2: the set-network embedding (the input to model.decoder — the
concatenated mean-pool / max-pool / context vector, captured with a forward
pre-hook) for all 2,000 seed-1 test scenarios, projected with UMAP
(n_neighbors=30, min_dist=0.1, random_state=1) AND PCA.  Coloured by
achieved 90% region radius, mast count, and Pasquill stability class.
After plotting, the structure is quantified (silhouette of a radius split,
Spearman correlation of each projection axis with log radius) and reported
plainly.  Note: mast count and stability are inputs to the context MLP, so
the embedding is expected to encode them.

Figure 3: integrated gradients of the predicted (argmax) cell's log-prob
w.r.t. the four input feature maps for the Figure-1 scenario
(ch4t-test-001811).  Baseline = zero maps, 50 midpoint steps; the raw
sensor tokens are held fixed, so this attributes the maps pathway.  The
completeness identity (sum of IG = logp(full) - logp(baseline)) is printed.

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

N_CELLS = N_GRID * N_GRID
SITE = 500.0
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF = "#fcfcfb"
BLUE, ORANGE, GOLD, REDX = "#2a78d6", "#eb6834", "#f2c230", "#e04343"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 16, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})

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
          "by mast count — consider cutting or reframing this figure.")
    stab = np.round(np.log(d_tst["D"]) * 5).astype(int)
    stab_names = np.array(list("ABCDEF"))

    for name, pr in (("umap", proj_umap), ("pca", proj_pca)):
        sil, r1, r2, rm, strat = stats[name.upper()]
        fig, axes = plt.subplots(1, 3, figsize=(16.6, 6.4))
        specs = [
            ("A. 90% region radius", radii, "magma_r",
             LogNorm(vmin=30, vmax=300), "region radius (m), log"),
            ("B. Number of masts", masts, "viridis", None, "masts"),
            ("C. Stability class", stab, "cividis", None,
             "Pasquill class (A–F)"),
        ]
        for ax, (title, c, cmap, norm, cbl) in zip(axes, specs):
            sc = ax.scatter(pr[:, 0], pr[:, 1], c=c, cmap=cmap, norm=norm,
                            s=13, linewidths=0, alpha=0.85, rasterized=True)
            ax.set_title(title, fontsize=18, fontweight="bold", loc="left",
                         pad=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#c3c2b7")
                sp.set_linewidth(1.2)
            cb = fig.colorbar(sc, ax=ax, orientation="horizontal",
                              fraction=0.05, pad=0.05, aspect=30)
            if norm is not None:
                cb.set_ticks([50, 100, 200])
                cb.set_ticklabels(["50", "100", "200"])
            if "Stability" in title:
                cb.set_ticks(range(6))
                cb.set_ticklabels(list("ABCDEF"))
            cb.set_label(cbl, fontsize=13, color=INK2)
            cb.ax.tick_params(labelsize=12)
            cb.outline.set_visible(False)

        # thumbnails of the source-evidence map for extreme-radius points
        ev = st_tst[:, 0, :].numpy()
        order = np.argsort(radii)
        for rank, col in [(order[:3], BLUE), (order[-3:], ORANGE)]:
            for j in rank:
                m = ev[j].reshape(N_GRID, N_GRID)
                lo, hi = np.percentile(m, [2, 99])
                im = OffsetImage(np.clip(m, lo, hi), zoom=0.55,
                                 cmap="viridis", origin="lower")
                ab = AnnotationBbox(
                    im, (pr[j, 0], pr[j, 1]), frameon=True, pad=0.12,
                    bboxprops=dict(edgecolor=col, linewidth=2.2))
                axes[0].add_artist(ab)
        axes[0].annotate("thumbnails: source-evidence maps of the 3 "
                         "smallest (blue)\nand 3 largest (orange) regions",
                         xy=(0, -0.30), xycoords="axes fraction",
                         fontsize=12, color=MUTED, va="top",
                         linespacing=1.35)

        fig.suptitle("What the set network learned: embedding of all "
                     "2,000 test scenarios", fontsize=21,
                     fontweight="bold", x=0.05, ha="left", y=1.075)
        fig.text(0.05, 1.020,
                 f"{name.upper()} of the decoder-input embedding "
                 "(mean-pool ⊕ max-pool ⊕ context), physics-guided "
                 f"DeepSets (ensr, seed 1) — silhouette of radius split "
                 f"{sil:.2f};\nSpearman(axis 1): log radius {r1:+.2f}, "
                 f"masts {rm:+.2f}, log radius within a mast count "
                 f"{strat:+.2f} — the layout tracks network size (an "
                 "input), not difficulty per se",
                 fontsize=13, color=MUTED, ha="left", va="top",
                 linespacing=1.45)
        fig.subplots_adjust(wspace=0.09, top=0.90, bottom=0.13,
                            left=0.03, right=0.985)
        for ext in ("pdf", "png"):
            p = HERE / f"fig2_embedding_{name}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"wrote {p}")
        plt.close(fig)


# ------------------------------------------------------------- Figure 3
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

    titles = [("Source evidence", 0), ("Wind-error\nsensitivity", 1),
              ("Sensor visibility", 2), ("Fit residual", 3)]
    vmax = max(np.percentile(np.abs(ig[ch]), 99.5) for _, ch in titles)
    ext = [0, SITE, 0, SITE]
    fig, axes = plt.subplots(1, 4, figsize=(16.6, 5.6))
    for ax, (title, ch) in zip(axes, titles):
        im = ax.imshow(ig[ch], origin="lower", cmap="RdBu_r", vmin=-vmax,
                       vmax=vmax, extent=ext, interpolation="bilinear")
        ax.scatter(sx[:, 0], sx[:, 1], marker="^", s=64, c="white",
                   edgecolors=INK, linewidths=1.1, zorder=6)
        ax.plot(*tx, marker="*", color=GOLD, ms=16, mec=INK, mew=1.0,
                zorder=7)
        ax.plot(*ax_, marker="x", color=REDX, ms=11, mew=2.6, zorder=7)
        ax.set_title(f"{title}\n$\\Sigma$IG = {ig[ch].sum():+.2f}",
                     fontsize=16.5, fontweight="bold", linespacing=1.2,
                     pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
            sp.set_linewidth(1.2)
    cb = fig.colorbar(im, ax=axes, fraction=0.023, pad=0.015, shrink=0.86)
    cb.set_label("IG of log $\\hat{p}$(argmax cell)", fontsize=13.5,
                 color=INK2)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)

    fig.suptitle("Which physics map drove the decision? Integrated "
                 "gradients on the Figure-1 scenario", fontsize=21,
                 fontweight="bold", x=0.045, ha="left", y=1.10)
    fig.text(0.045, 1.025,
             f"scenario ch4t-test-{FIG1_SCENARIO:06d} ({ns} masts); red = "
             "pushed probability toward the predicted cell, blue = away; "
             f"zero-map baseline, {steps} midpoint steps — completeness "
             f"|error| {abs(ig.sum() - (lp_full - lp_zero)):.3f} nats; "
             "complements the Table II ablation",
             fontsize=13, color=MUTED, ha="left")
    for ext_ in ("pdf", "png"):
        p = HERE / f"fig3_integrated_gradients.{ext_}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
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
