"""Analytic Green's-function forward model for the 2-D advection-diffusion PDE.

Unit-rate response (paper Eq. 2):

    A(x, t; x_s, u, D) = int_0^t  1/(pi w_s) * exp(-||x - x_s - u s||^2 / w_s) ds,
    w_s = 4 D s + 2 sigma_s^2 .

The age integral is evaluated with N-node Gauss-Legendre quadrature after the
log-age substitution s = t * exp(-tau), tau in [0, tau_max]:

    int_0^t f(s) ds = int_0^inf f(t e^-tau) t e^-tau dtau
                    ~ int_0^tau_max ... dtau   (tail < e^-tau_max, negligible)

which concentrates nodes near the near-singular small-s region.
A physical release of rate q produces field q * A.
"""
import numpy as np
import torch

_GL_CACHE = {}


def gl_nodes(n_nodes, tau_max, device, dtype):
    """Gauss-Legendre nodes/weights mapped from [-1,1] to [0, tau_max]."""
    key = (n_nodes, float(tau_max), str(device), dtype)
    if key not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n_nodes)      # on [-1, 1]
        tau = 0.5 * tau_max * (x + 1.0)
        wt = 0.5 * tau_max * w
        _GL_CACHE[key] = (torch.tensor(tau, device=device, dtype=dtype),
                          torch.tensor(wt, device=device, dtype=dtype))
    return _GL_CACHE[key]


def unit_response(x, t, xs, u, D, sigma_s, n_nodes=64, tau_max=12.0):
    """Vectorized unit-rate response A.

    All arguments are torch tensors that broadcast against each other:
      x  (..., 2)  sensor positions
      t  (...)     observation times, > 0
      xs (..., 2)  source location
      u  (..., 2)  wind vector
      D  (...)     diffusivity
    Returns A with the broadcast shape of the non-coordinate dims.
    A quadrature axis is appended internally (memory: broadcast_shape x n_nodes).
    """
    dtype = x.dtype
    device = x.device
    tau, wt = gl_nodes(n_nodes, tau_max, device, dtype)      # (Q,)

    t_ = t.unsqueeze(-1)                                     # (..., 1)
    s = t_ * torch.exp(-tau)                                 # (..., Q) ages
    w = 4.0 * D.unsqueeze(-1) * s + 2.0 * sigma_s ** 2       # (..., Q)

    # displacement d = x - xs - u * s, per quadrature node
    dx = (x[..., 0] - xs[..., 0]).unsqueeze(-1) - u[..., 0].unsqueeze(-1) * s
    dy = (x[..., 1] - xs[..., 1]).unsqueeze(-1) - u[..., 1].unsqueeze(-1) * s
    r2 = dx * dx + dy * dy

    integrand = torch.exp(-r2 / w) / (torch.pi * w)          # f(s)
    # ds = t e^-tau dtau  -> weight by s (= t e^-tau)
    return torch.sum(integrand * s * wt, dim=-1)


def sensor_responses(sensors, times, xs, u, D, sigma_s, n_nodes=64, tau_max=12.0):
    """Unit-rate responses for a batch of scenarios on a (sensor, time) lattice.

      sensors (B, S, 2), times (T,), xs (B, 2), u (B, 2), D (B,)
    Returns A of shape (B, S, T).
    """
    B, S, _ = sensors.shape
    T = times.shape[0]
    x = sensors[:, :, None, :].expand(B, S, T, 2)
    t = times[None, None, :].expand(B, S, T)
    xs_ = xs[:, None, None, :].expand(B, S, T, 2)
    u_ = u[:, None, None, :].expand(B, S, T, 2)
    D_ = D[:, None, None].expand(B, S, T)
    return unit_response(x, t, xs_, u_, D_, sigma_s, n_nodes, tau_max)


def cell_responses(cells, sensors, times, u, D, sigma_s, n_nodes=64, tau_max=12.0,
                   chunk=1024):
    """Unit-rate response g_c for every candidate source cell (oracle use).

      cells   (C, 2)  candidate source locations (cell centers)
      sensors (S, 2), times (T,), u (2,), D scalar tensor
    Returns g of shape (C, S, T), computed in `chunk`-cell blocks.
    """
    C = cells.shape[0]
    S = sensors.shape[0]
    T = times.shape[0]
    out = torch.empty((C, S, T), device=cells.device, dtype=cells.dtype)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        cc = cells[lo:hi]                                    # (c, 2)
        c = cc.shape[0]
        x = sensors[None, :, None, :].expand(c, S, T, 2)
        t = times[None, None, :].expand(c, S, T)
        xs_ = cc[:, None, None, :].expand(c, S, T, 2)
        u_ = u[None, None, None, :].expand(c, S, T, 2)
        D_ = D.reshape(1, 1, 1).expand(c, S, T)
        out[lo:hi] = unit_response(x, t, xs_, u_, D_, sigma_s, n_nodes, tau_max)
    return out


def zero_wind_exact(x, t, xs, D, sigma_s):
    """Closed-form zero-wind response via exponential integrals (float64, CPU).

    With u = 0:  A = 1/(4 pi D) * [E1(r^2 / w_t) - E1(r^2 / w_0)],
    w_0 = 2 sigma_s^2, w_t = 4 D t + 2 sigma_s^2.  (r > 0; E1 is decreasing
    and w_t > w_0, so the difference is positive.)
    """
    from scipy.special import exp1
    x = np.asarray(x, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    r2 = np.sum((x - xs) ** 2, axis=-1)
    w0 = 2.0 * sigma_s ** 2
    wt = 4.0 * D * np.asarray(t, dtype=np.float64) + w0
    return (1.0 / (4.0 * np.pi * D)) * (exp1(r2 / wt) - exp1(r2 / w0))
