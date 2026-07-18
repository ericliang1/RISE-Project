"""Stage 5: exact grid posterior oracle (paper Sec. 4.3), float64 log-space.

For each candidate cell c (center of a 64x64 grid cell), with unit-rate
response vector g_c over the scenario's retained (sensor, time) readings:

  log p(y | c, q) = -M/2 log(2 pi sigma^2) - ||y - q g_c||^2 / (2 sigma^2)
  ||y - q g_c||^2 = ||y||^2 - 2 q <y, g_c> + q^2 ||g_c||^2

The rate is marginalized over the LogUniform(0.5, 5) prior.  NOTE — documented
deviation from the paper's "32-node Gauss-Legendre in log q": that fixed rule
is numerically inadequate here, because the likelihood as a function of q is a
Gaussian of width sigma_q = sigma / ||g_c|| centered at qhat_c = <y,g_c>/||g_c||^2,
which for sharp scenarios is ~1000x narrower than the 32-node spacing (the
pipeline's own brute-force gate fails by many nats).  We instead use a
peak-aware COMPOSITE Gauss-Legendre rule with the same budget class: the prior
interval [0.5, 5] is split per cell at qhat_c -/+ 8 sigma_q (clipped), with 32
nodes on the peak panel and 16 on each outer panel (64 total).  The outer
panels carry <= e^-32 relative mass, and the peak panel resolves the Gaussian
spectrally.  Validated against a 10x-node deep rule and a dense trapezoid in
sanity_oracle.py (gate < 1e-6).

The expansion ||y - q g_c||^2 = ||y||^2 - 2 q <y,g_c> + q^2 ||g_c||^2 means
g_c enters only through a_c = <y, g_c> and b_c = ||g_c||^2, so the
(cells x readings x q-nodes) tensor is never materialized.

Usage:  python src/exact_posterior.py --splits calib test [--config ...]
"""
import argparse
import time

import numpy as np
import torch

from common import cell_centers, get_device, load_config, resolve, update_json
from forward_model import cell_responses
from inference import load_split

_GL01_CACHE = {}


def _gl01(n, device):
    """Standard GL nodes/weights mapped to [0, 1]."""
    key = (n, str(device))
    if key not in _GL01_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL01_CACHE[key] = (
            torch.tensor(0.5 * (x + 1.0), device=device, dtype=torch.float64),
            torch.tensor(0.5 * w, device=device, dtype=torch.float64))
    return _GL01_CACHE[key]


def marginal_log_evidence(a, b, yy, sigma, M, cfg, device,
                          n_peak=None, n_outer=None, halfwidth=None):
    """log p(y|c) for all cells via peak-aware composite GL over q (see header).

      a (C,) = <y, g_c>,  b (C,) = ||g_c||^2,  yy scalar = ||y||^2.
    """
    oc = cfg["oracle"]
    qlo = float(cfg["prior"]["q_log_low"])
    qhi = float(cfg["prior"]["q_log_high"])
    n_peak = n_peak or oc["q_peak_nodes"]
    n_outer = n_outer or oc["q_outer_nodes"]
    halfwidth = halfwidth or oc["q_peak_halfwidth_sigmas"]
    log_range = np.log(qhi / qlo)

    bb = torch.clamp(b, min=1e-280)
    qpk = a / bb                                   # likelihood peak in q
    sq = sigma / torch.sqrt(bb)                    # peak width in q
    lo = torch.clamp(qpk - halfwidth * sq, min=qlo, max=qhi)   # (C,)
    hi = torch.clamp(qpk + halfwidth * sq, min=qlo, max=qhi)

    xs_o, ws_o = _gl01(n_outer, device)
    xs_p, ws_p = _gl01(n_peak, device)
    panels = [(torch.full_like(lo, qlo), lo, xs_o, ws_o),
              (lo, hi, xs_p, ws_p),
              (hi, torch.full_like(hi, qhi), xs_o, ws_o)]
    qs, ws = [], []
    for p0, p1, x01, w01 in panels:
        width = (p1 - p0)[:, None]                 # (C, 1), >= 0
        qs.append(p0[:, None] + width * x01[None, :])
        ws.append(width * w01[None, :])
    q = torch.cat(qs, dim=1)                       # (C, N)
    w = torch.cat(ws, dim=1)                       # (C, N)

    quad = yy - 2.0 * q * a[:, None] + (q ** 2) * b[:, None]
    loglik = -0.5 * M * np.log(2.0 * np.pi * sigma ** 2) - quad / (2.0 * sigma ** 2)
    # prior density 1/(q log_range); zero-width panels give w=0 -> -inf, fine
    logw = torch.where(w > 0, torch.log(torch.clamp(w, min=1e-300)),
                       torch.full_like(w, -np.inf))
    return torch.logsumexp(logw + loglik - torch.log(q) - np.log(log_range), dim=1)


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
    yy = float(torch.dot(y, y))
    a = g @ y                                                 # (C,)  <y, g_c>
    b = (g * g).sum(dim=1)                                    # (C,)  ||g_c||^2

    log_ev = marginal_log_evidence(a, b, yy, sigma, M, cfg, device)
    return log_ev - torch.logsumexp(log_ev, dim=0)


@torch.no_grad()
def posteriors_for_split(d, cfg, device, grid_n=None, verbose_every=500):
    """(S, n_cells) probs FLOAT64 + logp float32 + normalization check.

    probs must be float64: in float32, p(true cell) underflows to exactly 0 for
    ~15% of scenarios (sharp posteriors), which breaks the conformal machinery
    (score atoms at ~1.0 and qhat pinned at 1.0 -> severe undercoverage)."""
    n = grid_n or cfg["grid"]["n"]
    cells_t = torch.tensor(cell_centers(n), device=device, dtype=torch.float64)
    S = len(d["ids"])
    probs = np.empty((S, n * n), dtype=np.float64)
    logp = np.empty((S, n * n), dtype=np.float32)
    norm_err = 0.0
    t0 = time.time()
    for i in range(S):
        lp = log_posterior_scenario(d, i, cfg, device, cells_t, n)
        p = torch.exp(lp)
        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
        probs[i] = p.cpu().numpy()
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
