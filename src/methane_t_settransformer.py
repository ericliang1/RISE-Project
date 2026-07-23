"""CH4-T, model M3: a Set Transformer localizer.

Motivation.  M0/M1 (DeepSetsT) aggregate the (mast, time) reading tokens by a
masked mean+max pool -- permutation-invariant, but every token is summarized
independently; the set has no way to *attend* to its own most informative
elements.  A Set Transformer (Lee et al., 2019) keeps permutation-invariance
but replaces pooling with attention: tokens attend to each other (ISAB) and are
pooled by attention (PMA).  This lets the model weight "the two masts that lit
up as the wind veered" over the many quiet readings -- a soft, learned version
of the relation the GNN (M2) hard-codes on edges.

Design.  Same reading tokens as DeepSetsT (only the set-aggregation changes):
  token = [Fourier(x_i), Fourier(y_i), Fourier(t_k), asinh(y_ik/sigma), u(t_k)]
  -> linear embed -> 2x ISAB (m inducing points) -> PMA (1 seed)
  -> concat scenario context -> SAME decoder head shape -> softmax over 64x64.
Attention respects the keep mask (dropped readings / padded masts are excluded
from keys/values), so masked tokens never leak into the pooled representation.

Everything downstream is byte-identical to DeepSetsT: tempered exact-posterior
teacher (blur 0.75), D4 augmentation, split conformal, region-size audit.  Only
the network changes, so M3 is comparable to M0/M1 (and the GNN M2) on the same
90%-coverage / equivalent-radius yardstick.

Usage (frozen data + posteriors reused; nothing regenerated):
  python src/methane_t_settransformer.py                 # M3 distill + one-hot
  python src/methane_t_settransformer.py --seed 2
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import get_device, load_config, resolve, update_json
from conformal import regions, tail_scores, tail_threshold
from methane_model import L_SITE as L
from model import count_params
from methane_t import N_GRID, SPLITS, blur_teacher, model_probs, train
from methane_t_gnn import load_split, _radius_m


# ---------------------------------------------------------------- attention
class MAB(nn.Module):
    """Multihead Attention Block: MAB(X, Y) with key padding on Y."""
    def __init__(self, d, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln0 = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, X, Y, key_pad=None):
        # key_pad: (B, |Y|) bool, True = IGNORE (torch convention)
        a, _ = self.mha(X, Y, Y, key_padding_mask=key_pad, need_weights=False)
        h = self.ln0(X + a)
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: O(n*m) self-attention via m inducing pts."""
    def __init__(self, d, heads, m):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m, d))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(d, heads)   # inducing points attend to the (masked) set
        self.mab1 = MAB(d, heads)   # the set attends back to inducing points

    def forward(self, X, key_pad=None):
        B = X.shape[0]
        H = self.mab0(self.I.expand(B, -1, -1), X, key_pad=key_pad)  # (B,m,d)
        return self.mab1(X, H)                                        # keys valid


class PMA(nn.Module):
    """Pooling by Multihead Attention: k learnable seeds pool the (masked) set."""
    def __init__(self, d, heads, k):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, k, d))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d, heads)

    def forward(self, Z, key_pad=None):
        B = Z.shape[0]
        return self.mab(self.S.expand(B, -1, -1), Z, key_pad=key_pad)  # (B,k,d)


# ---------------------------------------------------------------- model
class SetTransformerT(nn.Module):
    """Set Transformer over (mast, time) reading tokens (paper M3)."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        td = m["token_dim"] + 2                       # + (u_x, u_y) at t_k
        d = m["token_hidden"]                          # d_model (256), matched
        ch, dh = m["context_hidden"], m["decoder_hidden"]
        heads, m_ind = 4, 16                           # attn heads, inducing pts
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
        ], dim=-1)                                     # (B,S,T,td)
        X = self.embed(tok).reshape(B, S * Tt, -1)     # (B,N,d), N=S*T
        key_pad = ~keep.reshape(B, S * Tt)             # True = ignore token

        for isab in self.enc:
            X = isab(X, key_pad=key_pad)
        pooled = self.pma(X, key_pad=key_pad).squeeze(1)   # (B,d)

        ctx = self.context_mlp(torch.stack([
            b["u_mean"][:, 0], b["u_mean"][:, 1], torch.log(b["D"]),
            torch.log(b["sigma"]), b["n_sensors"].to(readings.dtype)], dim=-1))
        return self.decoder(torch.cat([pooled, ctx], dim=-1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=["distill", "onehot", "both"],
                    default="both")
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

    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    cov_e = float(reg_e["covered"].mean())
    print(f"  conformal-on-exact coverage: {cov_e:.3f} (guard [0.85,0.97])",
          flush=True)
    assert 0.85 <= cov_e <= 0.97, f"exact coverage {cov_e:.3f} out of range"

    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)),
                      N_GRID, 0.75, device=device)

    want = ["distill", "onehot"] if args.targets == "both" else [args.targets]
    res = {"exact_coverage": cov_e,
           "weak_frac": float((reg_e["sizes"] / 4096 >= 0.5).mean())}
    ratios = {}
    for tgt in want:
        distill = tgt == "distill"
        tag = "M3" if distill else "M3base"
        print(f"== train {tag} (SetTransformerT, {tgt}) ==", flush=True)
        model = train(cfg, data["train"], data["val"], device, distill,
                      P_teacher=Pt if distill else None,
                      P_val_teacher=Pv if distill else None,
                      seed=args.seed, model_cls=SetTransformerT)

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

    # significance vs distilled DeepSets (M1) and vs the GNN (M2), if present
    if "M3" in ratios:
        from scipy.stats import wilcoxon
        for base, path in [("M1", data_dir / "ch4t_audit_M1.npz"),
                           ("M2", data_dir / "ch4t_audit_M2_seed1.npz")]:
            if not path.exists():
                continue
            a = np.load(path)
            r_b = a["sizes"] / np.maximum(a["exact_sizes"], 1)
            res[f"{base}_median_radius_m"] = float(np.median(_radius_m(a["sizes"])))
            if r_b.shape == ratios["M3"].shape:
                w = wilcoxon(np.log(r_b), np.log(ratios["M3"]),
                             alternative="greater")
                res[f"wilcoxon_p_M3_sharper_than_{base}"] = float(w.pvalue)
                print(f"  M3 vs {base}: {base}={res[f'{base}_median_radius_m']:.1f} m"
                      f"  M3={res['M3']['median_radius_m']:.1f} m  p={w.pvalue:.2e}",
                      flush=True)

    out = results_dir / f"ch4t_st_results_seed{args.seed}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    update_json(results_dir / "gates.json", {f"CH4T_ST_seed{args.seed}": res})
    print(f"CH4T-ST DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
