"""Exact map-level SHAP analysis of the three physics feature maps (v3).

Each of the three `ens` input maps is treated as ONE feature: "present" =
the real map, "absent" = the zero baseline the conv head was
zero-initialized against.  With 3 features there are only 2^3 = 8
coalitions, so Shapley values are computed EXACTLY by enumerating every
coalition — no KernelSHAP sampling.  The value function is the model's
log-probability of the TRUE source cell, v_n(S) = log p_theta(c*_n | x_n,
maps in S), i.e. the training objective; a Shapley value is therefore the
map's average marginal NLL-reduction in nats.

Efficiency shortcut (mathematically exact, not an approximation): the model
is logits = base(x) + phys(M(x)), so the expensive set-network forward is
run ONCE per scenario and only the cheap conv head is re-run for each of
the 8 masked map stacks.

Models: DeepSets, GNN, and Set Transformer — the saved seed-1 checkpoints
of the super-emitter benchmark grid (csr-data-se48/checkpoints, written by
the benchmark training runs themselves with PW_SAVE_CKPT).

v3 figure change (user request): the Panel-B beeswarm's colour — the
vertical colourbar — now encodes the FEATURE VALUE (each map's spatial
mean for that scenario, rank-scaled to a percentile within its row, the
standard SHAP summary-plot gradient), not the mast count.

Outputs (this folder):
  shap_map_contributions.csv   per-scenario Shapley value + feature value
                               for each map
  shap_map_summary.{pdf,png,svg}   poster figure: mean |phi| bars + beeswarm

Sanity checks printed per model:
  - efficiency: max_n |sum_i phi_i - (v(all) - v(none))|  (~1e-15)
  - zero-baseline coalition vs base-network-only prediction (the trained
    head's phys(0) is a learned constant offset; its size is reported)

Usage (from repo root):
  python "Poster Graphics v3/shap/shap_maps.py"
"""
import csv
import itertools
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "Poster Graphics v3" / "xai graphs"))

from common import load_config, get_device, resolve            # noqa: E402
import paired_wind as pw                                       # noqa: E402
from methane_t import DeepSetsT, N_GRID, to_batch_t            # noqa: E402
from methane_t_uncertain import PhysHeadNet                    # noqa: E402
from xai_style import (CMAP_FEATVAL, MAP_ORDER, MUTED, INK2,   # noqa: E402
                       apply_style, header, save_all)

apply_style()

N_MAPS = len(MAP_ORDER)          # 3: the ens stack, residual map dropped
N_COAL = 1 << N_MAPS
# model palette shared with the Poster Graphics v3 result figures
MODEL_C = {"DeepSets": "#2a78d6", "GNN": "#c94040",
           "Set Transformer": "#1baf7a"}


def load_test_view(cfg, dd):
    d_clean = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    return pw.make_view(cfg, dd, ("test", d_clean), "noisy", "ens")


def build_model(name, cfg, dd, device):
    """(model, base_cls, ckpt_path) or None if the checkpoint is missing."""
    if name == "DeepSets":
        path = dd / "checkpoints" / "pw_noisy_ens_seed1.pt"
        base = DeepSetsT
        model = PhysHeadNet(cfg, N_MAPS)
    elif name == "GNN":
        from methane_t_gnn import GNNT
        path = dd / "checkpoints" / "pw_noisy_ens_gnn_seed1.pt"
        base = GNNT
        model = pw.phys_head_on(GNNT, N_MAPS)(cfg)
    else:
        from methane_t_settransformer import SetTransformerT
        path = dd / "checkpoints" / "pw_noisy_ens_st_seed1.pt"
        base = SetTransformerT
        model = pw.phys_head_on(SetTransformerT, N_MAPS)(cfg)
    if not path.exists():
        print(f"[{name}] checkpoint not found ({path}) — skipped")
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[{name}] checkpoint {path} (val_nll {ckpt['val_nll']:.4f})")
    return model, base, path


@torch.no_grad()
def coalition_values(model, base_cls, d, stats, device, chunk=256):
    """V (N, 8): log p(true cell) under every coalition of present maps,
    plus the base-network-only log p (head removed entirely)."""
    n = len(d["ids"])
    V = np.empty((n, N_COAL), np.float64)
    v_base_only = np.empty(n, np.float64)
    masks = torch.tensor([[float(m >> c & 1) for c in range(N_MAPS)]
                          for m in range(N_COAL)], device=device)
    for lo in range(0, n, chunk):
        idx = np.arange(lo, min(lo + chunk, n))
        b = to_batch_t(d, idx, device)
        st = stats[idx].to(device).view(len(idx), N_MAPS, N_GRID, N_GRID)
        base = base_cls.forward(model, b).float()
        tc = torch.tensor(d["true_cell"][idx], device=device)
        v_base_only[idx] = torch.log_softmax(base.double(), -1).gather(
            1, tc[:, None])[:, 0].cpu().numpy()
        for m in range(N_COAL):
            head = model.phys(st * masks[m][None, :, None, None])
            logits = base + head.flatten(1).float()
            V[idx, m] = torch.log_softmax(logits.double(), -1).gather(
                1, tc[:, None])[:, 0].cpu().numpy()
    return V, v_base_only


def exact_shapley(V):
    """Per-scenario exact Shapley values (N, 3) from the 8 coalition
    values, standard weights |S|!(n-|S|-1)!/n!."""
    n_sc = V.shape[0]
    phi = np.zeros((n_sc, N_MAPS))
    others = list(range(N_MAPS))
    for i in range(N_MAPS):
        rest = [c for c in others if c != i]
        for r in range(N_MAPS):
            for S in itertools.combinations(rest, r):
                m = sum(1 << c for c in S)
                w = (math.factorial(r) * math.factorial(N_MAPS - r - 1)
                     / math.factorial(N_MAPS))
                phi[:, i] += w * (V[:, m | (1 << i)] - V[:, m])
    return phi


def analyse(name, model, base_cls, ckpt_path, d, stats, device):
    V, v_base = coalition_values(model, base_cls, d, stats, device)
    phi = exact_shapley(V)

    resid = np.abs(phi.sum(1) - (V[:, N_COAL - 1] - V[:, 0]))
    print(f"[{name}] efficiency check: max |sum(phi) - (v(all)-v(none))| "
          f"= {resid.max():.2e}")
    d0 = np.abs(V[:, 0] - v_base)
    with torch.no_grad():
        p0 = model.phys(torch.zeros(1, N_MAPS, N_GRID, N_GRID,
                                    device=device)).flatten()
    print(f"[{name}] zero-map coalition vs base-network-only log p(true): "
          f"mean |diff| {d0.mean():.4f}, max {d0.max():.4f} nats — the "
          f"trained head's phys(0) is a learned constant offset map "
          f"(std {p0.std().item():.3f} logits), not exactly zero; at "
          f"initialization it was exactly zero")
    print(f"[{name}] v(all maps) mean {V[:, -1].mean():.4f}, "
          f"v(no maps) mean {V[:, 0].mean():.4f}  ->  total maps "
          f"contribution {(V[:, -1] - V[:, 0]).mean():.4f} nats/scenario")
    return phi, V, resid


def main():
    torch.manual_seed(0)
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    d, stats = load_test_view(cfg, dd)
    ns = d["n_sensors"].astype(int)
    ids = [str(s) for s in d["ids"]]
    print(f"test scenarios: {len(ids)} ({ids[0]} .. {ids[-1]})")

    # feature value per scenario and map: the channel's spatial mean —
    # the scalar the beeswarm colour encodes (rank-scaled per row below)
    featval = stats.numpy().reshape(len(ids), N_MAPS, -1).mean(2)

    results = {}
    for name in ("DeepSets", "GNN", "Set Transformer"):
        got = build_model(name, cfg, dd, device)
        if got is None:
            continue
        model, base_cls, path = got
        results[name] = analyse(name, model, base_cls, path, d, stats,
                                device)
        del model
        torch.cuda.empty_cache()

    # ---------------- CSV (canonical column order)
    csv_path = HERE / "shap_map_contributions.csv"
    cols = [t for t, _ in MAP_ORDER]

    def slug(c):
        return c.lower().replace(" ", "_").replace("-", "_")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scenario_id", "n_masts"]
                   + [f"shapley_{slug(c)}" for c in cols]
                   + [f"featval_{slug(c)}" for c in cols]
                   + ["v_all_maps", "v_no_maps", "efficiency_residual"])
        for name, (phi, V, resid) in results.items():
            for k in range(len(ids)):
                w.writerow([name, ids[k], ns[k]]
                           + [f"{phi[k, ch]:.6f}" for _, ch in MAP_ORDER]
                           + [f"{featval[k, ch]:.6f}"
                              for _, ch in MAP_ORDER]
                           + [f"{V[k, -1]:.6f}", f"{V[k, 0]:.6f}",
                              f"{resid[k]:.3e}"])
    print(f"wrote {csv_path}")

    # ---------------- printed summary table
    print("\nmean Shapley value per map (nats of NLL reduction; "
          "mean |phi| in parentheses):")
    hdr = "map".ljust(24) + "".join(n.ljust(26) for n in results)
    print(hdr)
    for title, ch in MAP_ORDER:
        row = title.ljust(24)
        for name, (phi, _, _) in results.items():
            row += (f"{phi[:, ch].mean():+.3f} "
                    f"({np.abs(phi[:, ch]).mean():.3f})").ljust(26)
        print(row)
    dphi = results["DeepSets"][0]
    order = [MAP_ORDER[k][0] for k in
             np.argsort([-np.abs(dphi[:, ch]).mean()
                         for _, ch in MAP_ORDER])]
    print(f"\nDeepSets ordering by mean |phi|: {' > '.join(order)}.")
    print("Table II's ladder adds the maps cumulatively and finds the "
          "evidence image the largest single contributor; the exact "
          "Shapley decomposition "
          f"{'AGREES — source evidence dominates' if order[0] == 'Source evidence' else 'DISAGREES with that ordering'} "
          "on the same seed-1 models and scenarios.")

    # ---------------- poster figure
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    names = list(results)
    nbar = len(names)
    ypos = np.arange(N_MAPS)[::-1]
    for j, name in enumerate(names):
        phi = results[name][0]
        vals = [np.abs(phi[:, ch]).mean() for _, ch in MAP_ORDER]
        off = (j - (nbar - 1) / 2) * (0.8 / nbar)
        ax.barh(ypos + off, vals, height=0.8 / nbar - 0.03,
                color=MODEL_C[name], label=name, zorder=3)
        for y, v in zip(ypos + off, vals):
            ax.annotate(f"{v:.2f}", xy=(v, y), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=12.5, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t for t, _ in MAP_ORDER], fontsize=15)
    ax.set_xlabel("mean |Shapley value| (nats)", fontsize=15)
    ax.set_title("A. Mean absolute contribution", loc="left", pad=9)
    ax.xaxis.grid(True, color="#e1e0d9", lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=13, loc="lower right")

    # Panel B — beeswarm coloured by FEATURE VALUE (per-row percentile of
    # the map's spatial mean): the classic SHAP summary-plot gradient.
    ax = axes[1]
    rng = np.random.default_rng(0)
    phi = results["DeepSets"][0]
    n_sc = phi.shape[0]
    for y, (title, ch) in zip(ypos, MAP_ORDER):
        x = phi[:, ch]
        # rank-scale this map's feature values to [0, 1] within the row
        pct = featval[:, ch].argsort().argsort() / (n_sc - 1)
        jit = rng.normal(0, 0.11, len(x)).clip(-0.32, 0.32)
        sc = ax.scatter(x, y + jit, c=pct, cmap=CMAP_FEATVAL, vmin=0,
                        vmax=1, s=7, linewidths=0, alpha=0.55,
                        rasterized=True)
        ax.plot(x.mean(), y, marker="D", color="#0b0b0b", ms=8, zorder=5)
    ax.axvline(0, color=INK2, lw=1.4, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([])
    ax.set_xlabel("per-scenario Shapley value (nats, log p of true cell)",
                  fontsize=15)
    ax.set_title("B. Per-scenario values — DeepSets", loc="left", pad=9)
    ax.xaxis.grid(True, color="#e1e0d9", lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=12)
    cb.set_label("map value for the scenario\n(percentile within row)",
                 fontsize=12.5, color=INK2, linespacing=1.3)
    cb.outline.set_visible(False)
    ax.annotate("black diamond = mean", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", fontsize=12,
                color=MUTED)

    header(fig, "Which physics map matters? Exact Shapley values over "
           "the three maps",
           "feature = one map, absent = zero baseline · value = log p(true "
           "cell) · all 8 coalitions enumerated per scenario · 2,000 "
           "seed-1 super-emitter test scenarios",
           ty=1.062, sy=1.019)
    save_all(fig, str(HERE / "shap_map_summary"))
    plt.close(fig)


if __name__ == "__main__":
    main()
