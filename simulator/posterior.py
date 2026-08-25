import numpy as np
import torch

from common import cell_centers
from scenarios import N_GRID, cell_responses_t, stab_of

_GL01_CACHE = {}


def _gl01(n, device):
    key = (n, str(device))
    if key not in _GL01_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL01_CACHE[key] = (
            torch.tensor(0.5 * (x + 1.0), device=device, dtype=torch.float64),
            torch.tensor(0.5 * w, device=device, dtype=torch.float64))
    return _GL01_CACHE[key]


def marginal_log_evidence(a, b, yy, sigma, M, cfg, device,
                          n_peak=None, n_outer=None, halfwidth=None):
    oc = cfg["oracle"]
    qlo = float(cfg["prior"]["q_log_low"])
    qhi = float(cfg["prior"]["q_log_high"])
    n_peak = n_peak or oc["q_peak_nodes"]
    n_outer = n_outer or oc["q_outer_nodes"]
    halfwidth = halfwidth or oc["q_peak_halfwidth_sigmas"]
    log_range = np.log(qhi / qlo)

    bb = torch.clamp(b, min=1e-280)
    qpk = a / bb
    sq = sigma / torch.sqrt(bb)
    lo = torch.clamp(qpk - halfwidth * sq, min=qlo, max=qhi)
    hi = torch.clamp(qpk + halfwidth * sq, min=qlo, max=qhi)

    xs_o, ws_o = _gl01(n_outer, device)
    xs_p, ws_p = _gl01(n_peak, device)
    panels = [(torch.full_like(lo, qlo), lo, xs_o, ws_o),
              (lo, hi, xs_p, ws_p),
              (hi, torch.full_like(hi, qhi), xs_o, ws_o)]
    qs, ws = [], []
    for p0, p1, x01, w01 in panels:
        width = (p1 - p0)[:, None]
        qs.append(p0[:, None] + width * x01[None, :])
        ws.append(width * w01[None, :])
    q = torch.cat(qs, dim=1)
    w = torch.cat(ws, dim=1)

    quad = yy - 2.0 * q * a[:, None] + (q ** 2) * b[:, None]
    loglik = -0.5 * M * np.log(2.0 * np.pi * sigma ** 2) - quad / (2.0 * sigma ** 2)
    logw = torch.where(w > 0, torch.log(torch.clamp(w, min=1e-300)),
                       torch.full_like(w, -np.inf))
    return torch.logsumexp(logw + loglik - torch.log(q) - np.log(log_range), dim=1)


@torch.no_grad()
def posteriors(d, cfg_q, device, verbose_every=2000):
    cells = torch.tensor(cell_centers(N_GRID), device=device, dtype=torch.float64)
    S = len(d["ids"])
    probs = np.empty((S, N_GRID * N_GRID), np.float64)
    norm_err = 0.0
    for i in range(S):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g = cell_responses_t(cells, d["sensors"][i, :ns], d["u_seq"][i],
                             stab_of(d, i), device)
        keep_t = torch.tensor(keep, device=device)
        g = g[:, keep_t]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        M = y.shape[0]
        a = g @ y
        b = (g * g).sum(1)
        yy = float(torch.dot(y, y))
        lp = marginal_log_evidence(a, b, yy, float(d["sigma"][i]), M,
                                   cfg_q, device)
        lp = lp - torch.logsumexp(lp, 0)
        p = torch.exp(lp)
        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
        probs[i] = p.cpu().numpy()
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  posterior {i+1}/{S}", flush=True)
    return probs, norm_err
