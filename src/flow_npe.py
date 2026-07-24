"""Flow-based neural posterior estimation (NPE) baseline on CH4-T.

The 2026-SOTA question: is the amortization gap an artifact of the categorical
grid head?  Same DeepSetsT encoder, but the head is a conditional normalizing
flow -- a 2-D autoregressive rational-quadratic-spline CDF over (x, y) in
[0,1]^2 (Durkan et al. 2019 parameterization; the spline IS the CDF, so
densities and cell masses are exact).  Trained exactly as NPE: maximize
log-density of the true source location (dequantized uniformly within its
cell, since CH4-T uses cell-level attribution).  Audited on the byte-identical
conformal harness via exact per-cell masses.

Diagnosis prediction: the deficit is in the ENCODER (per-cell matched
filtering cannot pass a pooled set embedding), so this modern head should NOT
close the gap (~M0-level radius).  The run decides.

Usage:  CUDA_VISIBLE_DEVICES=1 python src/flow_npe.py [--seed 1]
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon

from common import get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from methane_t import DeepSetsT, N_GRID, d4_augment_t, to_batch_t

L = 500.0
N_CELLS = N_GRID * N_GRID
K = 16          # spline bins
D_MIN = 1e-4    # min derivative / bin size


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * L)


# ------------------------------------------------- rational-quadratic spline
def rqs_cdf(theta, x):
    """Monotone RQS CDF [0,1]->[0,1] and its density.
    theta: (..., 3K+1) raw params; x: (...) in [0,1].
    Returns (F(x), f(x)).  Gregory-Delbourgo rational quadratic per bin."""
    w = F.softmax(theta[..., :K], -1) * (1 - K * D_MIN) + D_MIN         # widths
    h = F.softmax(theta[..., K:2 * K], -1) * (1 - K * D_MIN) + D_MIN    # heights
    d = F.softplus(theta[..., 2 * K:]) + D_MIN                          # K+1 derivs
    cw = torch.cumsum(w, -1)
    cw = torch.cat([torch.zeros_like(cw[..., :1]), cw], -1)             # x knots
    ch = torch.cumsum(h, -1)
    ch = torch.cat([torch.zeros_like(ch[..., :1]), ch], -1)             # y knots
    xc = x.clamp(1e-6, 1 - 1e-6)
    k = (torch.searchsorted(cw[..., 1:-1].contiguous(),
                            xc.unsqueeze(-1)).squeeze(-1))              # bin idx
    ga = lambda t, idx: torch.gather(t, -1, idx.unsqueeze(-1)).squeeze(-1)
    wk, hk = ga(w, k), ga(h, k)
    xk, yk = ga(cw, k), ga(ch, k)
    dk, dk1 = ga(d, k), ga(d, k + 1)
    s = hk / wk
    xi = ((xc - xk) / wk).clamp(0.0, 1.0)
    om = xi * (1 - xi)
    den = s + (dk1 + dk - 2 * s) * om
    Fx = yk + hk * (s * xi ** 2 + dk * om) / den
    fx = s ** 2 * (dk1 * xi ** 2 + 2 * s * om + dk * (1 - xi) ** 2) / den ** 2
    return Fx, fx


def rqs_cell_masses(theta, edges):
    """Exact masses of the 64 intervals between edges (65,) under the CDF."""
    x = edges.expand(*theta.shape[:-1], 65)
    Fx, _ = rqs_cdf(theta.unsqueeze(-2).expand(*theta.shape[:-1], 65,
                                               theta.shape[-1]), x)
    return (Fx[..., 1:] - Fx[..., :-1]).clamp_min(0.0)


# ------------------------------------------------------------------- model
class NPEFlow(nn.Module):
    """DeepSetsT encoder (decoder stripped) + autoregressive 2-D RQS flow.
    fair=True removes the review-flagged handicaps: larger head + full-band
    Fourier x-conditioning (the conditional p(y|x) varies at 1/64 scale)."""

    def __init__(self, cfg, fair=False):
        super().__init__()
        self.enc = DeepSetsT(cfg)
        m = cfg["model"]
        feat = 2 * m["token_hidden"] + m["context_hidden"]
        self.enc.decoder = nn.Identity()
        hid = 512 if fair else 256
        self.xfreqs = [1, 2, 4, 8, 16, 32] if fair else [1, 2]
        n_x = 1 + 2 * len(self.xfreqs)
        self.cond_x = nn.Sequential(nn.Linear(feat, hid), nn.GELU(),
                                    nn.Linear(hid, 3 * K + 1))
        self.cond_y = nn.Sequential(nn.Linear(feat + n_x, hid), nn.GELU(),
                                    nn.Linear(hid, 3 * K + 1))

    def feats(self, b):
        return self.enc(b)

    def y_theta(self, f, x):
        xf = [x] + [t(2 * np.pi * fq * x) for fq in self.xfreqs
                    for t in (torch.sin, torch.cos)]
        return self.cond_y(torch.cat([f, torch.stack(xf, -1)], -1))

    def nll(self, b, xy):
        f = self.feats(b)
        Fx, fx = rqs_cdf(self.cond_x(f), xy[:, 0])
        Fy, fy = rqs_cdf(self.y_theta(f, xy[:, 0]), xy[:, 1])
        return -(torch.log(fx + 1e-12) + torch.log(fy + 1e-12)).mean()

    @torch.no_grad()
    def cell_probs(self, b, edges, centers):
        """Exact per-cell masses (B, 4096), cell = iy*64 + ix."""
        f = self.feats(b)
        B = f.shape[0]
        mx = rqs_cell_masses(self.cond_x(f), edges)                 # (B,64)
        fx_rep = f[:, None, :].expand(B, N_GRID, f.shape[-1]).reshape(
            B * N_GRID, -1)
        cx = centers[None, :].expand(B, N_GRID).reshape(-1)
        my = rqs_cell_masses(self.y_theta(fx_rep, cx), edges)       # (B*64,64)
        my = my.view(B, N_GRID, N_GRID)                             # (B,ix,iy)
        P = mx[:, :, None] * my                                     # (B,ix,iy)
        return P.permute(0, 2, 1).reshape(B, N_CELLS)               # iy*64+ix


# ---------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fair", action="store_true",
                    help="review-fair variant: 500 epochs, hid 512, "
                         "full-band x conditioning")
    ap.add_argument("--noisy-wind", action="store_true",
                    help="Track-2: model sees the OBSERVED wind "
                         "(ch4tu_*_uobs.npy) instead of the true wind")
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    n_epochs = 500 if args.fair else 200
    data = {n: dict(np.load(dd / f"ch4t_{n}.npz", allow_pickle=True))
            for n in ("train", "val", "calib", "test")}
    if args.noisy_wind:
        from methane_t_uncertain import with_obs_wind
        data = {n: with_obs_wind(d, np.load(dd / f"ch4tu_{n}_uobs.npy"))
                for n, d in data.items()}
    torch.manual_seed(6000 + args.seed + 700)
    rng = np.random.default_rng(args.seed)
    model = NPEFlow(cfg, fair=args.fair).to(device)
    print(f"NPEFlow params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=tr["lr_final"])
    d_train, d_val = data["train"], data["val"]
    n_train = len(d_train["ids"])
    cell_w = 1.0 / N_GRID
    best, best_state = np.inf, None

    def val_nll():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for lo in range(0, len(d_val["ids"]), 512):
                idx = np.arange(lo, min(lo + 512, len(d_val["ids"])))
                b = to_batch_t(d_val, idx, device)
                xy = torch.tensor(d_val["xs"][idx], dtype=torch.float32,
                                  device=device)
                tot += float(model.nll(b, xy)) * len(idx)
        return tot / len(d_val["ids"])

    for ep in range(n_epochs):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            batch = to_batch_t(d_train, idx, device)
            batch, xs, _ = d4_augment_t(batch, d_train["xs"][idx],
                                        batch["u_seq"], rng, device)
            # dequantize within the cell (sources are grid-snapped)
            jit = rng.uniform(-0.5, 0.5, xs.shape) * cell_w
            xy = torch.tensor(np.clip(xs + jit, 1e-5, 1 - 1e-5),
                              dtype=torch.float32, device=device)
            loss = model.nll(batch, xy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError("non-finite loss/grad on first epoch")
            opt.step()
        sched.step()
        crit = val_nll()
        if crit < best:
            best = crit
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep {ep} val_nll {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch improved val NLL")
    model.load_state_dict(best_state)
    model.eval()

    # ---- audit: exact cell masses -> byte-identical conformal harness ----
    edges = torch.linspace(0, 1, N_GRID + 1, device=device)
    centers = (torch.arange(N_GRID, device=device) + 0.5) / N_GRID

    def probs_split(d):
        out = []
        for lo in range(0, len(d["ids"]), 256):
            idx = np.arange(lo, min(lo + 256, len(d["ids"])))
            b = to_batch_t(d, idx, device)
            out.append(model.cell_probs(b, edges, centers).cpu().numpy())
        P = np.concatenate(out).astype(np.float64)
        drift = float(np.abs(P.sum(1) - 1).max())
        P /= P.sum(1, keepdims=True)
        return P, drift

    Pc, dr1 = probs_split(data["calib"])
    Pt, dr2 = probs_split(data["test"])
    print(f"cell-mass normalization drift: {max(dr1, dr2):.2e}", flush=True)

    rng_c = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(Pc, data["calib"]["true_cell"], rng_c),
                        cfg["conformal"]["alpha"])
    reg = regions(Pt, th, data["test"]["true_cell"])
    m1 = np.load(dd / "ch4t_audit_M1.npz")["sizes"]
    w = wilcoxon(np.log(m1.astype(float)),
                 np.log(reg["sizes"].astype(float)), alternative="greater")
    out = {"tag": (f"flow_npe{'_fair' if args.fair else ''}"
                   f"{'_noisywind' if args.noisy_wind else ''}"
                   f"_seed{args.seed}"),
           "coverage": float(reg["covered"].mean()),
           "median_radius_m": rad_m(reg["sizes"]),
           "wilcoxon_p_vs_M1": float(w.pvalue),
           "frac_sharper_than_M1": float((reg["sizes"] < m1).mean()),
           "norm_drift": max(dr1, dr2)}
    res_path = rr / "ch4t_flow_npe.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[out["tag"]] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    np.savez_compressed(dd / f"ch4t_audit_flow_npe{'_fair' if args.fair else ''}_seed{args.seed}.npz",
                        sizes=reg["sizes"])
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
