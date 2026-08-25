import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


class DeepSetsT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        td = m["token_dim"] + 2
        th, ch, dh = m["token_hidden"], m["context_hidden"], m["decoder_hidden"]
        self.token_mlp = nn.Sequential(
            nn.Linear(td, th), nn.GELU(), nn.LayerNorm(th),
            nn.Linear(th, th), nn.GELU(), nn.LayerNorm(th))
        self.context_mlp = nn.Sequential(
            nn.Linear(5, ch), nn.GELU(), nn.LayerNorm(ch),
            nn.Linear(ch, ch), nn.GELU(), nn.LayerNorm(ch))
        self.decoder = nn.Sequential(
            nn.Linear(2 * th + ch, dh), nn.GELU(), nn.LayerNorm(dh),
            nn.Linear(dh, cfg["grid"]["n"] ** 2))

    def fourier(self, v):
        f = self.freqs.to(v.device, v.dtype)
        ang = np.pi * v.unsqueeze(-1) * f
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, b):
        sensors, readings, keep, times = b["sensors"], b["readings"], b["keep"], b["times"]
        u_seq = b["u_seq"]
        B, S, Tt = readings.shape
        tok = torch.cat([
            self.fourier(sensors[:, :, None, 0].expand(B, S, Tt)),
            self.fourier(sensors[:, :, None, 1].expand(B, S, Tt)),
            self.fourier(times[None, None, :].expand(B, S, Tt)),
            torch.asinh(readings / b["sigma"][:, None, None]).unsqueeze(-1),
            u_seq[:, None, :, 0].expand(B, S, Tt).unsqueeze(-1),
            u_seq[:, None, :, 1].expand(B, S, Tt).unsqueeze(-1),
        ], dim=-1)
        emb = self.token_mlp(tok)
        mask = keep.unsqueeze(-1)
        emb = emb * mask
        cnt = mask.sum(dim=(1, 2)).clamp(min=1)
        mean_pool = emb.sum(dim=(1, 2)) / cnt
        max_pool = emb.masked_fill(~mask, float("-inf")).amax(dim=(1, 2))
        ctx = self.context_mlp(torch.stack([
            b["u_mean"][:, 0], b["u_mean"][:, 1], torch.log(b["D"]),
            torch.log(b["sigma"]), b["n_sensors"].to(readings.dtype)], dim=-1))
        return self.decoder(torch.cat([mean_pool, max_pool, ctx], dim=-1))


def _fourier(v, freqs):
    f = freqs.to(v.device, v.dtype)
    ang = np.pi * v.unsqueeze(-1) * f
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GNNT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        nf = 2 * len(m["fourier_freqs"])
        H = m["token_hidden"]
        ch = m["context_hidden"]
        dh = m["decoder_hidden"]
        H2, He, Hm = 128, 128, H

        tdim = 1 + nf + 2
        self.temporal_mlp = nn.Sequential(
            nn.Linear(tdim, H2), nn.GELU(), nn.LayerNorm(H2),
            nn.Linear(H2, H2), nn.GELU(), nn.LayerNorm(H2))
        self.node_init = nn.Sequential(
            nn.Linear(2 * nf + 2 * H2, H), nn.GELU(), nn.LayerNorm(H))
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, He), nn.GELU(), nn.LayerNorm(He))
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * H + He, Hm), nn.GELU(), nn.LayerNorm(Hm))
            for _ in range(2)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(H + 2 * Hm, H), nn.GELU()) for _ in range(2)])
        self.upd_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
        self.context_mlp = nn.Sequential(
            nn.Linear(5, ch), nn.GELU(), nn.LayerNorm(ch),
            nn.Linear(ch, ch), nn.GELU(), nn.LayerNorm(ch))
        self.decoder = nn.Sequential(
            nn.Linear(2 * H + ch, dh), nn.GELU(), nn.LayerNorm(dh),
            nn.Linear(dh, cfg["grid"]["n"] ** 2))

    def fourier(self, v):
        return _fourier(v, self.freqs)

    def forward(self, b):
        sensors, readings, keep = b["sensors"], b["readings"], b["keep"]
        times, u_seq = b["times"], b["u_seq"]
        B, N, Tt = readings.shape

        tok = torch.cat([
            torch.asinh(readings / b["sigma"][:, None, None]).unsqueeze(-1),
            self.fourier(times)[None, None, :, :].expand(B, N, Tt, -1),
            u_seq[:, None, :, 0].expand(B, N, Tt).unsqueeze(-1),
            u_seq[:, None, :, 1].expand(B, N, Tt).unsqueeze(-1),
        ], dim=-1)
        te = self.temporal_mlp(tok)
        tmask = keep.unsqueeze(-1)
        tcnt = tmask.sum(2).clamp(min=1)
        t_mean = (te * tmask).sum(2) / tcnt
        t_max = te.masked_fill(~tmask, float("-inf")).amax(2)
        t_max = torch.nan_to_num(t_max, neginf=0.0)

        node_static = torch.cat([self.fourier(sensors[..., 0]),
                                 self.fourier(sensors[..., 1])], dim=-1)
        h = self.node_init(torch.cat([node_static, t_mean, t_max], dim=-1))

        node_valid = keep.any(dim=2)

        P = sensors
        dvec = P[:, None, :, :] - P[:, :, None, :]
        dist = torch.linalg.norm(dvec, dim=-1, keepdim=True)
        uhat = dvec / dist.clamp(min=EPS)
        align = torch.einsum("btc,bijc->bijt", u_seq, uhat)
        edge_raw = torch.cat([dvec, dist, align.mean(-1, keepdim=True),
                              align.amax(-1, keepdim=True)], dim=-1)
        e = self.edge_mlp(edge_raw)

        vpair = (node_valid[:, :, None] & node_valid[:, None, :])
        vpair = vpair & ~torch.eye(N, dtype=torch.bool,
                                   device=vpair.device)[None]

        for msg, upd, norm in zip(self.msg, self.upd, self.upd_norm):
            hi = h[:, :, None, :].expand(B, N, N, -1)
            hj = h[:, None, :, :].expand(B, N, N, -1)
            m = msg(torch.cat([hi, hj, e], dim=-1))
            vp = vpair.unsqueeze(-1)
            cnt = vpair.sum(2, keepdim=True).clamp(min=1)
            a_mean = m.masked_fill(~vp, 0.0).sum(2) / cnt
            a_max = torch.nan_to_num(
                m.masked_fill(~vp, float("-inf")).amax(2), neginf=0.0)
            h = norm(h + upd(torch.cat([h, a_mean, a_max], dim=-1)))

        nv = node_valid.unsqueeze(-1)
        g_mean = (h * nv).sum(1) / node_valid.sum(1, keepdim=True).clamp(min=1)
        g_max = torch.nan_to_num(
            h.masked_fill(~nv, float("-inf")).amax(1), neginf=0.0)
        ctx = self.context_mlp(torch.stack([
            b["u_mean"][:, 0], b["u_mean"][:, 1], torch.log(b["D"]),
            torch.log(b["sigma"]), b["n_sensors"].to(readings.dtype)], dim=-1))
        return self.decoder(torch.cat([g_mean, g_max, ctx], dim=-1))


class MAB(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln0 = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, X, Y, key_pad=None):
        a, _ = self.mha(X, Y, Y, key_padding_mask=key_pad, need_weights=False)
        h = self.ln0(X + a)
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    def __init__(self, d, heads, m):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m, d))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(d, heads)
        self.mab1 = MAB(d, heads)

    def forward(self, X, key_pad=None):
        B = X.shape[0]
        H = self.mab0(self.I.expand(B, -1, -1), X, key_pad=key_pad)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, d, heads, k):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, k, d))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d, heads)

    def forward(self, Z, key_pad=None):
        B = Z.shape[0]
        return self.mab(self.S.expand(B, -1, -1), Z, key_pad=key_pad)


class SetTransformerT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        td = m["token_dim"] + 2
        d = m["token_hidden"]
        ch, dh = m["context_hidden"], m["decoder_hidden"]
        heads, m_ind = 4, 16
        self.embed = nn.Linear(td, d)
        self.enc = nn.ModuleList([ISAB(d, heads, m_ind), ISAB(d, heads, m_ind)])
        self.pma = PMA(d, heads, k=1)
        self.context_mlp = nn.Sequential(
            nn.Linear(5, ch), nn.GELU(), nn.LayerNorm(ch),
            nn.Linear(ch, ch), nn.GELU(), nn.LayerNorm(ch))
        self.decoder = nn.Sequential(
            nn.Linear(d + ch, dh), nn.GELU(), nn.LayerNorm(dh),
            nn.Linear(dh, cfg["grid"]["n"] ** 2))

    def fourier(self, v):
        f = self.freqs.to(v.device, v.dtype)
        ang = np.pi * v.unsqueeze(-1) * f
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, b):
        sensors, readings, keep = b["sensors"], b["readings"], b["keep"]
        times, u_seq = b["times"], b["u_seq"]
        B, S, Tt = readings.shape
        tok = torch.cat([
            self.fourier(sensors[:, :, None, 0].expand(B, S, Tt)),
            self.fourier(sensors[:, :, None, 1].expand(B, S, Tt)),
            self.fourier(times[None, None, :].expand(B, S, Tt)),
            torch.asinh(readings / b["sigma"][:, None, None]).unsqueeze(-1),
            u_seq[:, None, :, 0].expand(B, S, Tt).unsqueeze(-1),
            u_seq[:, None, :, 1].expand(B, S, Tt).unsqueeze(-1),
        ], dim=-1)
        X = self.embed(tok).reshape(B, S * Tt, -1)
        key_pad = ~keep.reshape(B, S * Tt)

        for isab in self.enc:
            X = isab(X, key_pad=key_pad)
        pooled = self.pma(X, key_pad=key_pad).squeeze(1)

        ctx = self.context_mlp(torch.stack([
            b["u_mean"][:, 0], b["u_mean"][:, 1], torch.log(b["D"]),
            torch.log(b["sigma"]), b["n_sensors"].to(readings.dtype)], dim=-1))
        return self.decoder(torch.cat([pooled, ctx], dim=-1))


class PhysHeadNet(DeepSetsT):
    def __init__(self, cfg, n_ch):
        super().__init__(cfg)
        self.phys = nn.Sequential(
            nn.Conv2d(n_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.phys[-1].weight)
        nn.init.zeros_(self.phys[-1].bias)

    def forward(self, b):
        base = super().forward(b)
        return base + self.phys(b["stats"]).flatten(1)


def phys_head_on(base_cls, n_ch):
    class WithPhys(base_cls):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.phys = nn.Sequential(
                nn.Conv2d(n_ch, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 1, 3, padding=1))
            nn.init.zeros_(self.phys[-1].weight)
            nn.init.zeros_(self.phys[-1].bias)

        def forward(self, b):
            return super().forward(b) + self.phys(b["stats"]).flatten(1)

    WithPhys.__name__ = f"{base_cls.__name__}Phys{n_ch}"
    return WithPhys
