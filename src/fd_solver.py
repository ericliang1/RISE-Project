"""Disposable 2nd-order finite-difference cross-check of the analytic model.

Solves  dC/dt + u.grad(C) = D lap(C) + S  on an enlarged box containing the
unit square, the source, and the advected plume (Dirichlet C=0 far boundary),
with 2nd-order central differences in space and Heun (RK2) in time, CFL-stable
dt.  The source is the same Gaussian blob of width sigma_s the analytic model
regularizes with:  S(x) = q * N(x; x_s, sigma_s^2 I).

Sensor readings are extracted by bilinear interpolation at times t_k and
compared against q * A(analytic) in relative L2 over all (sensor, time) pairs.
"""
import numpy as np
import torch

from forward_model import sensor_responses


def _box(xs, u, t_max, margin):
    """Axis-aligned box covering sensors [0,1]^2, source, and advected plume."""
    lo = np.minimum(0.0, np.minimum(xs, xs + u * t_max)) - margin
    hi = np.maximum(1.0, np.maximum(xs, xs + u * t_max)) + margin
    return lo, hi


@torch.no_grad()
def fd_sensor_readings(xs, u, D, q, sigma_s, sensors, times, h, cfl_safety=0.4,
                       margin=0.6, device="cuda", dtype=torch.float32,
                       progress=False):
    """Run the FD solver for one scenario; return (S, T) readings at sensors."""
    xs = np.asarray(xs, float)
    u = np.asarray(u, float)
    sensors = np.asarray(sensors, float)
    times = np.asarray(times, float)
    t_max = float(times.max())

    lo, hi = _box(xs, u, t_max, margin)
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    xg = torch.arange(nx, device=device, dtype=torch.float64) * h + lo[0]
    yg = torch.arange(ny, device=device, dtype=torch.float64) * h + lo[1]
    X, Y = torch.meshgrid(xg, yg, indexing="xy")             # (ny, nx)

    # Gaussian blob source, rate q, width sigma_s
    S = (q / (2.0 * np.pi * sigma_s ** 2) *
         torch.exp(-((X - xs[0]) ** 2 + (Y - xs[1]) ** 2) / (2.0 * sigma_s ** 2)))
    S = S.to(dtype)

    C = torch.zeros((ny, nx), device=device, dtype=dtype)

    dt_diff = h * h / (4.0 * D)
    dt_adv = h / (abs(u[0]) + abs(u[1]) + 1e-12)
    dt = cfl_safety * min(dt_diff, dt_adv)
    # land exactly on the reading times: use a uniform dt dividing 1/30
    per = times[0]
    n_sub = int(np.ceil(per / dt))
    dt = per / n_sub

    inv_h2 = 1.0 / (h * h)
    inv_2h = 1.0 / (2.0 * h)
    ux, uy = float(u[0]), float(u[1])
    Dc = float(D)

    def rhs(c):
        out = torch.zeros_like(c)
        lap = (c[1:-1, 2:] + c[1:-1, :-2] + c[2:, 1:-1] + c[:-2, 1:-1]
               - 4.0 * c[1:-1, 1:-1]) * inv_h2
        ddx = (c[1:-1, 2:] - c[1:-1, :-2]) * inv_2h
        ddy = (c[2:, 1:-1] - c[:-2, 1:-1]) * inv_2h
        out[1:-1, 1:-1] = Dc * lap - ux * ddx - uy * ddy + S[1:-1, 1:-1]
        return out

    # bilinear interpolation indices for the sensors
    sx = torch.tensor((sensors[:, 0] - lo[0]) / h, device=device)
    sy = torch.tensor((sensors[:, 1] - lo[1]) / h, device=device)
    ix0 = sx.floor().long().clamp(0, nx - 2)
    iy0 = sy.floor().long().clamp(0, ny - 2)
    fx = (sx - ix0).to(dtype)
    fy = (sy - iy0).to(dtype)

    def read(c):
        c00 = c[iy0, ix0]
        c01 = c[iy0, ix0 + 1]
        c10 = c[iy0 + 1, ix0]
        c11 = c[iy0 + 1, ix0 + 1]
        return ((1 - fy) * ((1 - fx) * c00 + fx * c01)
                + fy * ((1 - fx) * c10 + fx * c11))

    readings = torch.zeros((len(sensors), len(times)), device=device, dtype=dtype)
    t = 0.0
    for k, tk in enumerate(times):
        for _ in range(n_sub):
            k1 = rhs(C)
            k2 = rhs(C + dt * k1)
            C = C + 0.5 * dt * (k1 + k2)
        t += per
        readings[:, k] = read(C)
        if progress and (k + 1) % 10 == 0:
            print(f"  t = {t:.3f}  ({(k+1)}/{len(times)})", flush=True)
    return readings.double().cpu().numpy()


@torch.no_grad()
def analytic_sensor_readings(xs, u, D, q, sigma_s, sensors, times,
                             n_nodes=64, tau_max=12.0, device="cuda"):
    """Reference q*A at the same (sensor, time) lattice, float64."""
    dd = torch.device(device)
    A = sensor_responses(
        torch.tensor(sensors, device=dd, dtype=torch.float64)[None],
        torch.tensor(times, device=dd, dtype=torch.float64),
        torch.tensor(xs, device=dd, dtype=torch.float64)[None],
        torch.tensor(u, device=dd, dtype=torch.float64)[None],
        torch.tensor([D], device=dd, dtype=torch.float64),
        sigma_s, n_nodes, tau_max)[0]
    return (q * A).cpu().numpy()


def rel_l2(a, b):
    """Relative L2 of a vs reference b over all entries."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
