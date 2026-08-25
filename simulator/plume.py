import numpy as np
import torch

L_SITE = 500.0
U_SCALE = 8.0
H_SRC = 3.0
Z_R = 2.0
SIGMA_Y0 = 2.0
SIGMA_Z0 = 1.0
X_MIN = 1.0
PPM_KGM3 = 6.55e-7
KGH_KGS = 1.0 / 3600.0

_BR_Y = [(0.22, 1e-4), (0.16, 1e-4), (0.11, 1e-4),
         (0.08, 1e-4), (0.06, 1e-4), (0.04, 1e-4)]
_BR_Z = [(0.20, 0.0, 0), (0.12, 0.0, 0), (0.08, 2e-4, 1),
         (0.06, 1.5e-3, 1), (0.03, 3e-4, 2), (0.016, 3e-4, 2)]


def _sigmas(x, stab):
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
    d = (sensors_norm - xs_norm) * L_SITE
    u_phys = u_norm * U_SCALE
    U = torch.linalg.norm(u_phys, dim=-1).clamp(min=0.5)
    ux = u_phys[..., 0] / U
    uy = u_phys[..., 1] / U
    x_dw = d[..., 0] * ux + d[..., 1] * uy
    y_cw = -d[..., 0] * uy + d[..., 1] * ux
    x_eff = x_dw.clamp(min=X_MIN)
    sy, sz = _sigmas(x_eff, stab)
    sy = torch.sqrt(sy ** 2 + SIGMA_Y0 ** 2)
    sz = torch.sqrt(sz ** 2 + SIGMA_Z0 ** 2)
    c = (KGH_KGS / (2.0 * np.pi * U * sy * sz)
         * torch.exp(-0.5 * (y_cw / sy) ** 2)
         * (torch.exp(-0.5 * ((Z_R - H_SRC) / sz) ** 2)
            + torch.exp(-0.5 * ((Z_R + H_SRC) / sz) ** 2)))
    c = torch.where(x_dw > 0, c, torch.zeros_like(c))
    return c / PPM_KGM3
