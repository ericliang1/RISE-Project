"""CH4-500: methane-configured forward model.

Steady-state 3-D Gaussian plume with ground reflection and Pasquill-Gifford
stability-dependent dispersion (Briggs open-country sigma curves) on a
500 m x 500 m site.  Closed form -- no quadrature -- so the exact-posterior
machinery transfers unchanged (the response is linear in the release rate Q).

Units: positions stored normalized in [0,1] (x500 m inside), wind stored
normalized (u_norm = u_phys / U_SCALE, direction preserved -- D4-rotation
safe), release rate Q in kg/h, concentrations/noise in ppm CH4 enhancement
above background.  Release height H = 3 m, sensor height Z_R = 2 m (fixed,
documented).  Readings are 30 x 10-s averages of the steady plume + iid noise.

Briggs (1973) open-country curves; 1 ppm CH4 = 6.55e-7 kg/m^3 at 25 C.
"""
import numpy as np
import torch

L_SITE = 500.0        # m
U_SCALE = 8.0         # m/s (wind normalization)
H_SRC = 3.0           # m release height
Z_R = 2.0             # m sensor height
SIGMA_Y0 = 2.0        # m source-size floor (crosswind)
SIGMA_Z0 = 1.0        # m source-size floor (vertical)
X_MIN = 1.0           # m downwind floor
PPM_KGM3 = 6.55e-7    # kg/m^3 per ppm CH4
KGH_KGS = 1.0 / 3600.0

# Briggs open-country: sigma_y = a x (1 + b x)^-1/2 ; sigma_z per-class form
_BR_Y = [(0.22, 1e-4), (0.16, 1e-4), (0.11, 1e-4),
         (0.08, 1e-4), (0.06, 1e-4), (0.04, 1e-4)]          # A..F
# sigma_z: (c, d, form): form 0: c x; 1: c x (1+d x)^-1/2; 2: c x (1+d x)^-1
_BR_Z = [(0.20, 0.0, 0), (0.12, 0.0, 0), (0.08, 2e-4, 1),
         (0.06, 1.5e-3, 1), (0.03, 3e-4, 2), (0.016, 3e-4, 2)]


def _sigmas(x, stab):
    """Briggs sigma_y, sigma_z (m) for downwind distance x (m), stability
    index tensor stab in {0..5}.  Vectorized over broadcast shapes."""
    ay = torch.tensor([p[0] for p in _BR_Y], dtype=x.dtype, device=x.device)[stab]
    by = torch.tensor([p[1] for p in _BR_Y], dtype=x.dtype, device=x.device)[stab]
    sy = ay * x / torch.sqrt(1.0 + by * x)
    cz = torch.tensor([p[0] for p in _BR_Z], dtype=x.dtype, device=x.device)[stab]
    dz = torch.tensor([p[1] for p in _BR_Z], dtype=x.dtype, device=x.device)[stab]
    fz = torch.tensor([p[2] for p in _BR_Z], dtype=torch.long, device=x.device)[stab]
    sz0 = cz * x
    sz1 = cz * x / torch.sqrt(1.0 + dz * x)
    sz2 = cz * x / (1.0 + dz * x)
    sz = torch.where(fz == 0, sz0, torch.where(fz == 1, sz1, sz2))
    return sy, sz


def plume_ppm_per_kgh(sensors_norm, xs_norm, u_norm, stab):
    """ppm enhancement per (kg/h) of release.

    sensors_norm (..., 2), xs_norm (..., 2) in [0,1]; u_norm (..., 2)
    normalized wind (m/s / U_SCALE); stab (...) long in {0..5}.
    Broadcast over leading dims; returns same leading shape.
    """
    d = (sensors_norm - xs_norm) * L_SITE                     # m
    u_phys = u_norm * U_SCALE
    U = torch.linalg.norm(u_phys, dim=-1).clamp(min=0.5)      # m/s
    ux = u_phys[..., 0] / U
    uy = u_phys[..., 1] / U
    x_dw = d[..., 0] * ux + d[..., 1] * uy                    # downwind (m)
    y_cw = -d[..., 0] * uy + d[..., 1] * ux                   # crosswind (m)
    x_eff = x_dw.clamp(min=X_MIN)
    sy, sz = _sigmas(x_eff, stab)
    sy = torch.sqrt(sy ** 2 + SIGMA_Y0 ** 2)
    sz = torch.sqrt(sz ** 2 + SIGMA_Z0 ** 2)
    c = (KGH_KGS / (2.0 * np.pi * U * sy * sz)
         * torch.exp(-0.5 * (y_cw / sy) ** 2)
         * (torch.exp(-0.5 * ((Z_R - H_SRC) / sz) ** 2)
            + torch.exp(-0.5 * ((Z_R + H_SRC) / sz) ** 2)))
    c = torch.where(x_dw > 0, c, torch.zeros_like(c))         # no upwind plume
    return c / PPM_KGM3                                       # ppm per kg/h


def plume_ppm_per_kgh_smeared(sensors_norm, xs_norm, u_norm, stab, sig_th):
    """Expected plume under N(0, sig_th^2) wind-direction error.

    First-order in the rotation: the crosswind offset shifts by -x*dth, so
    the crosswind Gaussian convolves to sigma_y_eff^2 = sigma_y^2 +
    (sig_th * x)^2 (classic meander-enhanced dispersion), and the hard
    upwind cutoff becomes the Gaussian gate P(x_dw > 0) with x_dw jitter
    std sig_th*|y_cw|.  Speed error is a near-constant scale factor and
    cancels in the matched-filter z statistic, so it is not smeared here."""
    d = (sensors_norm - xs_norm) * L_SITE
    u_phys = u_norm * U_SCALE
    U = torch.linalg.norm(u_phys, dim=-1).clamp(min=0.5)
    ux = u_phys[..., 0] / U
    uy = u_phys[..., 1] / U
    x_dw = d[..., 0] * ux + d[..., 1] * uy
    y_cw = -d[..., 0] * uy + d[..., 1] * ux
    x_eff = x_dw.clamp(min=X_MIN)
    sy, sz = _sigmas(x_eff, stab)
    sy = torch.sqrt(sy ** 2 + SIGMA_Y0 ** 2 + (sig_th * x_eff) ** 2)
    sz = torch.sqrt(sz ** 2 + SIGMA_Z0 ** 2)
    c = (KGH_KGS / (2.0 * np.pi * U * sy * sz)
         * torch.exp(-0.5 * (y_cw / sy) ** 2)
         * (torch.exp(-0.5 * ((Z_R - H_SRC) / sz) ** 2)
            + torch.exp(-0.5 * ((Z_R + H_SRC) / sz) ** 2)))
    gate = 0.5 * (1.0 + torch.erf(
        x_dw / (sig_th * y_cw.abs() + 1e-6) / np.sqrt(2.0)))
    return c * gate / PPM_KGM3


def ch4_cell_responses(cells, sensors, u_norm, stab, device, chunk=2048):
    """Unit-rate (1 kg/h) response g_c for all candidate cells.

      cells (C, 2) normalized centers; sensors (S, 2) normalized;
      u_norm (2,); stab scalar int.  Returns (C, S) float64.
    """
    C, S = cells.shape[0], sensors.shape[0]
    out = torch.empty((C, S), device=device, dtype=torch.float64)
    stab_t = torch.full((1, 1), int(stab), dtype=torch.long, device=device)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        cc = cells[lo:hi]
        out[lo:hi] = plume_ppm_per_kgh(
            sensors[None, :, :].expand(hi - lo, S, 2),
            cc[:, None, :].expand(hi - lo, S, 2),
            u_norm[None, None, :].expand(hi - lo, S, 2),
            stab_t.expand(hi - lo, S))
    return out
