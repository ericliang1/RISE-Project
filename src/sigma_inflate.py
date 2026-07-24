"""Robust plug-in inversion baseline (reviewer point 2): plug-in evidence at
the measured wind with an inflated noise scale sigma_eff = c*sigma absorbing
wind-model error.  c chosen on the calibration split (half calibrate / half
evaluate, min median conformal region); audited on test with the standard
harness.  Usage: python src/sigma_inflate.py"""
import json

import numpy as np
import torch

from common import cell_centers, get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from exact_posterior import marginal_log_evidence
from methane_pipeline import ch4_cfg
from methane_t import N_GRID, cell_responses_t, stab_of

N_CELLS = N_GRID * N_GRID


@torch.no_grad()
def posteriors_inflated(d, u_obs, cells, cfg_q, device, c):
    n = len(d["ids"])
    P = np.empty((n, N_CELLS), np.float64)
    for i in range(n):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g = cell_responses_t(cells, d["sensors"][i, :ns], u_obs[i],
                             stab_of(d, i), device)
        g = g[:, torch.tensor(keep, device=device)]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        a = g @ y
        b = (g * g).sum(1)
        yy = float(torch.dot(y, y))
        lp = marginal_log_evidence(a, b, yy, float(d["sigma"][i]) * c,
                                   int(y.shape[0]), cfg_q, device)
        P[i] = torch.exp(lp - torch.logsumexp(lp, 0)).cpu().numpy()
    return P


def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    alpha = cfg["conformal"]["alpha"]
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    d_c = dict(np.load(dd / "ch4t_calib.npz", allow_pickle=True))
    d_t = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    uo_c = np.load(dd / "ch4tu_calib_uobs.npy")
    uo_t = np.load(dd / "ch4tu_test_uobs.npy")

    # choose c on calib: first half calibrates, second half evaluates
    h = len(d_c["ids"]) // 2
    best = None
    for c in (1.5, 2.0, 3.0, 5.0):
        P = posteriors_inflated(d_c, uo_c, cells, cfg_q, device, c)
        rng = np.random.default_rng(cfg["conformal"]["score_seed"] + 5)
        th = tail_threshold(tail_scores(P[:h], d_c["true_cell"][:h], rng),
                            alpha)
        reg = regions(P[h:], th, d_c["true_cell"][h:])
        med = float(np.median(reg["sizes"]))
        cov = float(reg["covered"].mean())
        z0 = float((P[np.arange(len(d_c["ids"])), d_c["true_cell"]]
                    == 0).mean())
        print(f"  c={c}: calib-eval median {med:.0f} cells, cov {cov:.3f}, "
              f"p_true==0 frac {z0:.3f}", flush=True)
        if best is None or med < best[1]:
            best = (c, med, P)
    c_star, _, P_calib = best
    print(f"selected c* = {c_star}", flush=True)

    P_test = posteriors_inflated(d_t, uo_t, cells, cfg_q, device, c_star)
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(P_calib, d_c["true_cell"], rng), alpha)
    reg = regions(P_test, th, d_t["true_cell"])
    rad = float(np.sqrt(np.median(reg["sizes"] / N_CELLS) / np.pi) * 500)
    out = {"tag": "plugin_sigma_inflated", "c_star": c_star,
           "coverage": float(reg["covered"].mean()), "median_radius_m": rad,
           "frac_full_grid": float((reg["sizes"] == N_CELLS).mean())}
    res_path = rr / "ch4t_uncertain.json"
    all_res = json.load(open(res_path))
    all_res["plugin_sigma_inflated"] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
