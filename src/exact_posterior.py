"""Stage 5: exact grid posterior oracle (paper Sec. 4.3), float64 log-space.

For each candidate cell c (center of a 64x64 grid cell), with unit-rate
response vector g_c over the scenario's retained (sensor, time) readings:

  log p(y | c, q) = -M/2 log(2 pi sigma^2) - ||y - q g_c||^2 / (2 sigma^2)
  ||y - q g_c||^2 = ||y||^2 - 2 q <y, g_c> + q^2 ||g_c||^2

The rate is marginalized over LogUniform(0.5, 5) with 32-node Gauss-Legendre
quadrature in L = log q  (pi(q) dq = dL / log 10):

  log p(y | c) = logsumexp_m [log w_m + log p(y | c, e^{L_m})] - log(log 10)

and the posterior is normalized over all cells with logsumexp.  The expansion
of the quadratic form means g_c enters only through <y, g_c> and ||g_c||^2,
so the (cells x readings x q-nodes) tensor is never materialized.

Usage:  python src/exact_posterior.py --splits calib test [--config ...]
"""
import argparse
import time

import numpy as np
import torch

from common import cell_centers, get_device, load_config, resolve, update_json
from forward_model import cell_responses
from inference import load_split

_QGL_CACHE = {}


def _q_nodes(cfg, device):
    key = str(device)
    if key not in _QGL_CACHE:
        n = cfg["oracle"]["q_quad_nodes"]
        lo, hi = np.log(cfg["prior"]["q_log_low"]), np.log(cfg["prior"]["q_log_high"])
        x, w = np.polynomial.legendre.leggauss(n)
        L = 0.5 * (hi - lo) * (x + 1.0) + lo
        wt = 0.5 * (hi - lo) * w
        _QGL_CACHE[key] = (
            torch.tensor(np.exp(L), device=device, dtype=torch.float64),   # q_m
            torch.tensor(np.log(wt), device=device, dtype=torch.float64),  # log w_m
            float(hi - lo),                                                # log 10
        )
    return _QGL_CACHE[key]


@torch.no_grad()
def log_posterior_scenario(d, i, cfg, device, cells_t=None, grid_n=None):
    """Log posterior (n_cells,) float64 for scenario i of split dict d."""
    ph = cfg["physics"]
    oc = cfg["oracle"]
    n = grid_n or cfg["grid"]["n"]
    if cells_t is None:
        cells_t = torch.tensor(cell_centers(n), device=device, dtype=torch.float64)

    ns = int(d["n_sensors"][i])
    keep = d["keep"][i, :ns]                                  # (ns, T)
    sensors = torch.tensor(d["sensors"][i, :ns], device=device, dtype=torch.float64)
    times = torch.tensor(d["times"], device=device, dtype=torch.float64)
    u = torch.tensor(d["u"][i], device=device, dtype=torch.float64)
    D = torch.tensor(float(d["D"][i]), device=device, dtype=torch.float64)
    sigma = float(d["sigma"][i])

    g = cell_responses(cells_t, sensors, times, u, D, ph["sigma_s"],
                       ph["quad_nodes"], ph["tau_max"], oc["cell_chunk"])
    g = g / ph["c_ref"]                                       # (C, ns, T)
    keep_t = torch.tensor(keep, device=device)
    g = g[:, keep_t]                                          # (C, M) retained
    y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                     dtype=torch.float64)                     # (M,)

    M = y.shape[0]
    yy = torch.dot(y, y)
    a = g @ y                                                 # (C,)  <y, g_c>
    b = (g * g).sum(dim=1)                                    # (C,)  ||g_c||^2

    q_m, log_w, log_range = _q_nodes(cfg, device)
    # (C, Q): log p(y | c, q_m)
    quad = yy[None, None] - 2.0 * q_m[None, :] * a[:, None] \
        + (q_m ** 2)[None, :] * b[:, None]
    loglik = (-0.5 * M * np.log(2.0 * np.pi * sigma ** 2)
              - quad / (2.0 * sigma ** 2))
    log_ev = torch.logsumexp(log_w[None, :] + loglik, dim=1) - np.log(log_range)
    return log_ev - torch.logsumexp(log_ev, dim=0)


@torch.no_grad()
def posteriors_for_split(d, cfg, device, grid_n=None, verbose_every=500):
    """(S, n_cells) probs float32 + logp float32 + normalization check."""
    n = grid_n or cfg["grid"]["n"]
    cells_t = torch.tensor(cell_centers(n), device=device, dtype=torch.float64)
    S = len(d["ids"])
    probs = np.empty((S, n * n), dtype=np.float32)
    logp = np.empty((S, n * n), dtype=np.float32)
    norm_err = 0.0
    t0 = time.time()
    for i in range(S):
        lp = log_posterior_scenario(d, i, cfg, device, cells_t, n)
        p = torch.exp(lp)
        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
        probs[i] = p.float().cpu().numpy()
        logp[i] = lp.float().cpu().numpy()
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  {i+1}/{S}  ({(time.time()-t0)/(i+1)*1000:.0f} ms/scen)",
                  flush=True)
    return probs, logp, norm_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--grid-n", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")

    for split in args.splits:
        n = args.grid_n or cfg["grid"]["n"]
        tag = f"{split}_posterior" + (f"_{n}" if args.grid_n else "")
        print(f"exact posteriors: {split} (grid {n}x{n})", flush=True)
        d = load_split(data_dir, split)
        t0 = time.time()
        probs, logp, norm_err = posteriors_for_split(d, cfg, device, args.grid_n)
        wall = time.time() - t0
        np.savez_compressed(data_dir / f"{tag}.npz", ids=d["ids"], probs=probs,
                            logp=logp, grid_n=n)
        update_json(results_dir / "gates.json",
                    {"G4": {f"{tag}_max_norm_err": norm_err,
                            f"{tag}_wall_s": wall}})
        print(f"  done in {wall:.0f}s, max |1-sum(p)| = {norm_err:.2e}", flush=True)


if __name__ == "__main__":
    main()
