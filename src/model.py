"""Permutation-invariant DeepSets localizer (~5M params, paper Sec. 4.1).

Token per retained reading:
  [Fourier(x_i), Fourier(y_i), Fourier(t_k), asinh(y_ik / sigma)]  (25 dims)
Fourier(v) = [sin(pi f v), cos(pi f v)] for f in {1,2,4,8}  (8 dims / coord;
pi rather than 2*pi so features are injective on [0,1]).

Per-token MLP 25->256->256 (GELU + LayerNorm); pooling by BOTH masked mean and
masked max, concatenated (512); context MLP (u, v, logD, logsigma, N) 5->64->64;
decoder 576->1024->4096 logits over the 64x64 grid.
"""
import numpy as np
import torch
import torch.nn as nn


class DeepSetsLocalizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        td, th = m["token_dim"], m["token_hidden"]
        ch, dh = m["context_hidden"], m["decoder_hidden"]
        n_cells = cfg["grid"]["n"] ** 2

        self.token_mlp = nn.Sequential(
            nn.Linear(td, th), nn.GELU(), nn.LayerNorm(th),
            nn.Linear(th, th), nn.GELU(), nn.LayerNorm(th),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(5, ch), nn.GELU(), nn.LayerNorm(ch),
            nn.Linear(ch, ch), nn.GELU(), nn.LayerNorm(ch),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * th + ch, dh), nn.GELU(), nn.LayerNorm(dh),
            nn.Linear(dh, n_cells),
        )

    def fourier(self, v):
        """v (...,) -> (..., 8): sin/cos at frequencies {1,2,4,8} (pi-scaled)."""
        f = self.freqs.to(v.device, v.dtype)
        ang = np.pi * v.unsqueeze(-1) * f
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, batch):
        """batch dict of tensors:
          sensors (B,S,2), readings (B,S,T), keep (B,S,T) bool,
          times (T,), u (B,2), D (B,), sigma (B,), n_sensors (B,)
        Returns logits (B, n_cells).
        """
        sensors, readings = batch["sensors"], batch["readings"]
        keep, times = batch["keep"], batch["times"]
        B, S, T = readings.shape

        tok = torch.cat([
            self.fourier(sensors[:, :, None, 0].expand(B, S, T)),
            self.fourier(sensors[:, :, None, 1].expand(B, S, T)),
            self.fourier(times[None, None, :].expand(B, S, T)),
            torch.asinh(readings / batch["sigma"][:, None, None]).unsqueeze(-1),
        ], dim=-1)                                        # (B,S,T,25)
        emb = self.token_mlp(tok)                         # (B,S,T,256)

        mask = keep.unsqueeze(-1)                         # (B,S,T,1)
        emb = emb * mask
        cnt = mask.sum(dim=(1, 2)).clamp(min=1)           # (B,1)
        mean_pool = emb.sum(dim=(1, 2)) / cnt
        max_pool = emb.masked_fill(~mask, float("-inf")).amax(dim=(1, 2))

        ctx = self.context_mlp(torch.stack([
            batch["u"][:, 0], batch["u"][:, 1],
            torch.log(batch["D"]), torch.log(batch["sigma"]),
            batch["n_sensors"].to(readings.dtype),
        ], dim=-1))

        return self.decoder(torch.cat([mean_pool, max_pool, ctx], dim=-1))


def smoothed_targets(true_cells, n, std_cells, trunc_sigmas, device):
    """Normalized Gaussian bump (std in cells, truncated at trunc_sigmas)
    centered on the true source cell.  true_cells (B,) int64 -> (B, n*n)."""
    ix = (true_cells % n).to(device)
    iy = (true_cells // n).to(device)
    r = int(np.ceil(trunc_sigmas * std_cells))
    offs = torch.arange(-r, r + 1, device=device)
    dx, dy = torch.meshgrid(offs, offs, indexing="xy")
    d2 = (dx ** 2 + dy ** 2).float()
    w = torch.exp(-d2 / (2.0 * std_cells ** 2))
    w = w * (d2.sqrt() <= trunc_sigmas * std_cells)

    B = true_cells.shape[0]
    tgt = torch.zeros(B, n * n, device=device)
    gx = (ix[:, None, None] + dx[None]).clamp_(0, n - 1)  # clamp keeps mass in-grid
    gy = (iy[:, None, None] + dy[None]).clamp_(0, n - 1)
    flat = (gy * n + gx).reshape(B, -1)
    tgt.scatter_add_(1, flat, w.reshape(1, -1).expand(B, -1))
    return tgt / tgt.sum(dim=1, keepdim=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
