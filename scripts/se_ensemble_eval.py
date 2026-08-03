"""Seed-ensemble evaluation: average the 3 seeds' probability maps per
configuration, conformalize the ensemble on calib, audit on test.
Uses the saved checkpoints; no training.  Usage:
  python scripts/se_ensemble_eval.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "src")

from common import load_config, resolve, get_device
from conformal import regions, tail_scores, tail_threshold
from methane_t import DeepSetsT, N_GRID, to_batch_t
from methane_t_uncertain import PhysHeadNet
from paired_wind import make_view, phys_head_on

cfg = load_config()
dd = resolve(cfg, "data_dir")
alpha = cfg["conformal"]["alpha"]
device = get_device()
rad = lambda s: np.sqrt(np.asarray(s, float) / 4096 / np.pi) * 500
N_CELLS = N_GRID * N_GRID


@torch.no_grad()
def probs_for(model, d, stats):
    out = []
    for lo in range(0, len(d["ids"]), 512):
        idx = np.arange(lo, min(lo + 512, len(d["ids"])))
        b = to_batch_t(d, idx, device)
        if stats is not None:
            b["stats"] = stats[idx].to(device).view(len(idx), -1, N_GRID,
                                                    N_GRID)
        out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def build(arch, n_ch):
    if arch == "gnn":
        from methane_t_gnn import GNNT as base
        return phys_head_on(base, n_ch)(cfg) if n_ch else base(cfg)
    if arch == "st":
        from methane_t_settransformer import SetTransformerT as base
        return phys_head_on(base, n_ch)(cfg) if n_ch else base(cfg)
    return PhysHeadNet(cfg, n_ch) if n_ch else DeepSetsT(cfg)


clean = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))
         for s in ("calib", "test")}
views = {s: make_view(cfg, dd, (s, clean[s]), "noisy", "ens")
         for s in ("calib", "test")}

print("%-24s %-10s %-12s %-10s" % ("config (seed-ensemble)", "coverage",
                                   "radius (m)", "below 50m"))
for arch, at in [("deepsets", ""), ("gnn", "_gnn"), ("st", "_st")]:
    P = {"calib": 0.0, "test": 0.0}
    for seed in (1, 2, 3):
        ck = torch.load(dd / "checkpoints" / f"pw_noisy_ens{at}_seed{seed}.pt",
                        map_location=device, weights_only=False)
        model = build(arch, 3).to(device).eval()
        model.load_state_dict(ck["model_state"])
        for s in ("calib", "test"):
            d, stats = views[s]
            P[s] = P[s] + probs_for(model, d, stats) / 3.0
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(P["calib"], clean["calib"]["true_cell"],
                                    rng), alpha)
    reg = regions(P["test"], th, clean["test"]["true_cell"])
    r = rad(reg["sizes"])
    print("%-24s %-10.4f %-12.1f %-10s" % (f"{arch} ens x3", reg["covered"].mean(),
                                           float(np.median(r)),
                                           "%.1f%%" % (100 * (r < 50).mean())))
