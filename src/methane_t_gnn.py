"""CH4-T, model M2: a relational (graph) localizer.

Motivation.  DeepSetsT (methane_t.py) pools (mast, time) reading tokens by a
masked mean+max -- permutation-invariant, but it never represents an inter-mast
RELATION.  Yet the signal the paper calls decisive is exactly relational: as the
wind veers, the plume sweeps ACROSS masts, so "mast A lit up before mast B, and
the wind was blowing A->B" localizes the source (time-of-arrival / bearing).
A graph net can carry that on its edges; a bag-of-tokens net cannot.

Design (mast-node, wind-conditioned edges).
  * Nodes = the <=12 masts.  A node's feature is its position (Fourier) plus a
    masked pool over its OWN time series of [asinh(y/sigma), Fourier(t), u(t)],
    so "when did I light up" lives on the node.
  * Edges = ordered mast pairs (i->j).  The edge feature carries geometry
    (displacement, distance) AND wind alignment over the window
    (mean/max of u(t) . unit(j-i)) -- i.e. "did the wind blow the plume i->j".
  * 2 rounds of masked message passing, then a global mean+max readout into the
    SAME decoder head shape as DeepSetsT, decoding a softmax over the 64x64 grid.

Everything downstream is byte-identical to DeepSetsT's pipeline: the tempered
exact-posterior teacher (blur 0.75), D4 augmentation, split conformal, and the
region-size audit.  Only the network changes, so M2 is comparable to M0/M1 on
the same 90%-coverage / equivalent-radius yardstick.

Capacity is matched to DeepSetsT (~4.9M params) -- the shared decoder head
(576 -> 1024 -> 4096) dominates -- so any audit difference reflects the
inductive bias, not parameter count.

Usage (frozen data + posteriors are reused; nothing is regenerated):
  python src/methane_t_gnn.py                 # train M2 distilled + one-hot, audit
  python src/methane_t_gnn.py --targets distill
  python src/methane_t_gnn.py --seed 2
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import (cell_centers, get_device, load_config, resolve,
                    update_json)
from conformal import regions, tail_scores, tail_threshold
from methane_model import L_SITE, U_SCALE
from methane_pipeline import ch4_cfg
from model import count_params
# reuse the DeepSetsT training loop, teacher tempering, and conformal reference
from methane_t import (N_GRID, T, SPLITS, blur_teacher, model_probs, train)

EPS = 1e-6


# ---------------------------------------------------------------- model
def _fourier(v, freqs):
    """[sin(pi f v), cos(pi f v)] over freqs -- identical to DeepSetsT."""
    f = freqs.to(v.device, v.dtype)
    ang = np.pi * v.unsqueeze(-1) * f
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GNNT(nn.Module):
    """Mast-node graph localizer with wind-conditioned edges (paper M2)."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        nf = 2 * len(m["fourier_freqs"])            # Fourier dims per coordinate
        H = m["token_hidden"]                        # node hidden (256), matched
        ch = m["context_hidden"]                     # 64
        dh = m["decoder_hidden"]                     # 1024
        H2, He, Hm = 128, 128, H                     # temporal / edge / message

        # per-(mast, time) token -> masked pool over t -> node temporal summary
        tdim = 1 + nf + 2                            # asinh(y/s), Fourier(t), u(t)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(tdim, H2), nn.GELU(), nn.LayerNorm(H2),
            nn.Linear(H2, H2), nn.GELU(), nn.LayerNorm(H2))
        # node init: [Fourier(x), Fourier(y), temporal mean+max] -> H
        self.node_init = nn.Sequential(
            nn.Linear(2 * nf + 2 * H2, H), nn.GELU(), nn.LayerNorm(H))
        # static edge encoder: [dx, dy, |d|, mean_align, max_align] -> He
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, He), nn.GELU(), nn.LayerNorm(He))
        # two message-passing rounds (message MLP + residual update MLP each)
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * H + He, Hm), nn.GELU(), nn.LayerNorm(Hm))
            for _ in range(2)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(H + 2 * Hm, H), nn.GELU()) for _ in range(2)])
        self.upd_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
        # scenario context (same 5 features as DeepSetsT)
        self.context_mlp = nn.Sequential(
            nn.Linear(5, ch), nn.GELU(), nn.LayerNorm(ch),
            nn.Linear(ch, ch), nn.GELU(), nn.LayerNorm(ch))
        # decoder head: SAME shape as DeepSetsT (2H + ch -> dh -> grid)
        self.decoder = nn.Sequential(
            nn.Linear(2 * H + ch, dh), nn.GELU(), nn.LayerNorm(dh),
            nn.Linear(dh, cfg["grid"]["n"] ** 2))

    def fourier(self, v):
        return _fourier(v, self.freqs)

    def forward(self, b):
        sensors, readings, keep = b["sensors"], b["readings"], b["keep"]
        times, u_seq = b["times"], b["u_seq"]        # (T,), (B,T,2)
        B, N, Tt = readings.shape

        # ---- node temporal summary (masked pool over time) ----
        tok = torch.cat([
            torch.asinh(readings / b["sigma"][:, None, None]).unsqueeze(-1),
            self.fourier(times)[None, None, :, :].expand(B, N, Tt, -1),
            u_seq[:, None, :, 0].expand(B, N, Tt).unsqueeze(-1),
            u_seq[:, None, :, 1].expand(B, N, Tt).unsqueeze(-1),
        ], dim=-1)                                   # (B,N,T,tdim)
        te = self.temporal_mlp(tok)
        tmask = keep.unsqueeze(-1)                    # (B,N,T,1)
        tcnt = tmask.sum(2).clamp(min=1)
        t_mean = (te * tmask).sum(2) / tcnt
        t_max = te.masked_fill(~tmask, float("-inf")).amax(2)
        t_max = torch.nan_to_num(t_max, neginf=0.0)  # masts with no readings

        node_static = torch.cat([self.fourier(sensors[..., 0]),
                                 self.fourier(sensors[..., 1])], dim=-1)
        h = self.node_init(torch.cat([node_static, t_mean, t_max], dim=-1))

        node_valid = keep.any(dim=2)                 # (B,N) bool

        # ---- static edge features (geometry + wind alignment) ----
        P = sensors                                  # (B,N,2)
        dvec = P[:, None, :, :] - P[:, :, None, :]   # (B,N,N,2): j - i  (edge i->j)
        dist = torch.linalg.norm(dvec, dim=-1, keepdim=True)          # (B,N,N,1)
        uhat = dvec / dist.clamp(min=EPS)
        align = torch.einsum("btc,bijc->bijt", u_seq, uhat)          # (B,N,N,T)
        edge_raw = torch.cat([dvec, dist, align.mean(-1, keepdim=True),
                              align.amax(-1, keepdim=True)], dim=-1)  # (B,N,N,5)
        e = self.edge_mlp(edge_raw)

        # valid ordered pairs, no self-loops
        vpair = (node_valid[:, :, None] & node_valid[:, None, :])
        vpair = vpair & ~torch.eye(N, dtype=torch.bool,
                                   device=vpair.device)[None]         # (B,N,N)

        # ---- message passing ----
        for msg, upd, norm in zip(self.msg, self.upd, self.upd_norm):
            hi = h[:, :, None, :].expand(B, N, N, -1)
            hj = h[:, None, :, :].expand(B, N, N, -1)
            m = msg(torch.cat([hi, hj, e], dim=-1))  # (B,N,N,Hm)
            vp = vpair.unsqueeze(-1)
            cnt = vpair.sum(2, keepdim=True).clamp(min=1)             # (B,N,1)
            a_mean = m.masked_fill(~vp, 0.0).sum(2) / cnt
            a_max = torch.nan_to_num(
                m.masked_fill(~vp, float("-inf")).amax(2), neginf=0.0)
            h = norm(h + upd(torch.cat([h, a_mean, a_max], dim=-1)))

        # ---- global readout over valid nodes ----
        nv = node_valid.unsqueeze(-1)
        g_mean = (h * nv).sum(1) / node_valid.sum(1, keepdim=True).clamp(min=1)
        g_max = torch.nan_to_num(
            h.masked_fill(~nv, float("-inf")).amax(1), neginf=0.0)
        ctx = self.context_mlp(torch.stack([
            b["u_mean"][:, 0], b["u_mean"][:, 1], torch.log(b["D"]),
            torch.log(b["sigma"]), b["n_sensors"].to(readings.dtype)], dim=-1))
        return self.decoder(torch.cat([g_mean, g_max, ctx], dim=-1))


# ---------------------------------------------------------------- audit
def _radius_m(sizes):
    """Region size (cells) -> equivalent-radius (m), as in the paper table."""
    area_m2 = (sizes / (N_GRID * N_GRID)) * (L_SITE ** 2)
    return np.sqrt(area_m2 / np.pi)


def load_split(data_dir, name):
    d = dict(np.load(data_dir / f"ch4t_{name}.npz", allow_pickle=True))
    P = np.load(data_dir / f"ch4t_{name}_posterior.npz")["probs"]
    return d, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=["distill", "onehot", "both"],
                    default="both", help="M2 supervision target(s) to train")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    alpha = cfg["conformal"]["alpha"]

    print("== load frozen CH4-T data + exact posteriors ==", flush=True)
    data, P_ex = {}, {}
    for name in SPLITS:
        data[name], P_ex[name] = load_split(data_dir, name)
        print(f"  {name}: {len(data[name]['ids'])}", flush=True)

    # exact-posterior conformal reference (same construction as methane_t.main)
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    cov_e = float(reg_e["covered"].mean())
    print(f"  conformal-on-exact coverage: {cov_e:.3f} (guard [0.85,0.97])",
          flush=True)
    assert 0.85 <= cov_e <= 0.97, (
        f"conformal-on-exact coverage {cov_e:.3f} out of range -- "
        "oracle/data mismatch, aborting.")

    # tempered teacher (identical to M1)
    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)),
                      N_GRID, 0.75, device=device)

    want = (["distill", "onehot"] if args.targets == "both" else [args.targets])
    res = {"exact_coverage": cov_e,
           "weak_frac": float((reg_e["sizes"] / 4096 >= 0.5).mean())}
    ratios = {}
    for tgt in want:
        distill = tgt == "distill"
        tag = "M2" if distill else "M2base"
        print(f"== train {tag} (GNNT, {tgt}) ==", flush=True)
        model = train(cfg, data["train"], data["val"], device, distill,
                      P_teacher=Pt if distill else None,
                      P_val_teacher=Pv if distill else None,
                      seed=args.seed, model_cls=GNNT)

        P_c = model_probs(model, data["calib"], device)
        P_t = model_probs(model, data["test"], device)
        r2 = np.random.default_rng(cfg["conformal"]["score_seed"])
        th = tail_threshold(tail_scores(P_c, data["calib"]["true_cell"], r2),
                            alpha)
        reg = regions(P_t, th, data["test"]["true_cell"])
        ratio = reg["sizes"] / np.maximum(reg_e["sizes"], 1)
        ratios[tag] = ratio
        ident = (reg_e["sizes"] / 4096) < 0.10
        res[tag] = {
            "coverage": float(reg["covered"].mean()),
            "median_radius_m": float(np.median(_radius_m(reg["sizes"]))),
            "ineff_median_all": float(np.median(ratio)),
            "ineff_median_identifiable": float(np.median(ratio[ident])),
        }
        np.savez_compressed(data_dir / f"ch4t_audit_{tag}_seed{args.seed}.npz",
                            sizes=reg["sizes"], exact_sizes=reg_e["sizes"])
        print(f"  {tag}: {res[tag]}", flush=True)

    # significance vs the distilled DeepSets baseline (M1), if its audit exists
    m1p = data_dir / "ch4t_audit_M1.npz"
    if "M2" in ratios and m1p.exists():
        from scipy.stats import wilcoxon
        a = np.load(m1p)
        r_m1 = a["sizes"] / np.maximum(a["exact_sizes"], 1)
        res["M1_median_radius_m"] = float(np.median(_radius_m(a["sizes"])))
        if r_m1.shape == ratios["M2"].shape:
            w = wilcoxon(np.log(r_m1), np.log(ratios["M2"]),
                         alternative="greater")
            res["wilcoxon_p_M2_sharper_than_M1"] = float(w.pvalue)
            print(f"  M1 median radius {res['M1_median_radius_m']:.1f} m  vs  "
                  f"M2 {res['M2']['median_radius_m']:.1f} m  "
                  f"(Wilcoxon p={w.pvalue:.2e})", flush=True)

    out = results_dir / f"ch4t_gnn_results_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    update_json(results_dir / "gates.json", {f"CH4T_GNN_seed{args.seed}": res})
    print(f"CH4T-GNN DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
