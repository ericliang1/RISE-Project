"""Physics-guided flow-NPE: the two winning levers unified.

Fair flow head (62.8-71.2 m, no physics) + the recipe's physics maps
(which took the categorical head to 44-49 m).  Physics enters through
ZERO-INITIALIZED additive spline-parameter heads, so step 0 is exactly the
fair NPEFlow; conditioning:
  x-spline   += head( column-pooled map features )            (B,2x2x64)
  y|x-spline += head( the map column at x )                   (B,2x64)
Optional --teacher: train on samples from the 0.75-blurred exact posterior
instead of the true location -- flow-native distillation.

Usage:  python src/flow_phys.py [--teacher] [--seed 1]
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

from common import get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from flow_npe import K, N_CELLS, NPEFlow, rad_m, rqs_cdf, rqs_cell_masses
from methane_t import N_GRID, d4_augment_t, to_batch_t
from methane_t_levers import stats_maps
from train import blur_teacher, d4_target_perms


class NPEFlowPhys(NPEFlow):
    def __init__(self, cfg):
        super().__init__(cfg, fair=True)
        hid = 256
        self.phys_x = nn.Sequential(nn.Linear(4 * N_GRID, hid), nn.GELU(),
                                    nn.Linear(hid, 3 * K + 1))
        self.phys_y = nn.Sequential(nn.Linear(2 * N_GRID, hid), nn.GELU(),
                                    nn.Linear(hid, 3 * K + 1))
        for h in (self.phys_x, self.phys_y):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    @staticmethod
    def colfeat(M):
        """(B,2,64,64)[ch,iy,ix] -> pooled-over-y features (B, 256)."""
        return torch.cat([M.amax(2), M.mean(2)], 1).flatten(1)

    @staticmethod
    def col_at(M, ix):
        """Map column at x-index: (B,2,64,64), (B,) -> (B, 128)."""
        cols = M.permute(0, 3, 1, 2)                       # (B, ix, ch, iy)
        return cols[torch.arange(len(ix), device=ix.device), ix].flatten(1)

    def nll(self, b, xy):
        f = self.feats(b)
        M = b["stats"]
        tx = self.cond_x(f) + self.phys_x(self.colfeat(M))
        ix = (xy[:, 0] * N_GRID).long().clamp(0, N_GRID - 1)
        ty = (self.y_theta(f, xy[:, 0])
              + self.phys_y(self.col_at(M, ix)))
        Fx, fx = rqs_cdf(tx, xy[:, 0])
        Fy, fy = rqs_cdf(ty, xy[:, 1])
        return -(torch.log(fx + 1e-12) + torch.log(fy + 1e-12)).mean()

    @torch.no_grad()
    def cell_probs(self, b, edges, centers):
        f = self.feats(b)
        M = b["stats"]
        B = f.shape[0]
        tx = self.cond_x(f) + self.phys_x(self.colfeat(M))
        mx = rqs_cell_masses(tx, edges)                    # (B,64)
        fx_rep = f[:, None, :].expand(B, N_GRID, f.shape[-1]).reshape(
            B * N_GRID, -1)
        cx = centers[None, :].expand(B, N_GRID).reshape(-1)
        ixs = (cx * N_GRID).long().clamp(0, N_GRID - 1)
        M_rep = M[:, None].expand(B, N_GRID, 2, N_GRID, N_GRID).reshape(
            B * N_GRID, 2, N_GRID, N_GRID)
        ty = (self.y_theta(fx_rep, cx)
              + self.phys_y(self.col_at(M_rep, ixs)))
        my = rqs_cell_masses(ty, edges).view(B, N_GRID, N_GRID)
        P = mx[:, :, None] * my                            # (B, ix, iy)
        return P.permute(0, 2, 1).reshape(B, N_CELLS)      # iy*64+ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--teacher", action="store_true",
                    help="train on samples from the tempered exact posterior")
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    n_epochs = 500
    data = {n: dict(np.load(dd / f"ch4t_{n}.npz", allow_pickle=True))
            for n in ("train", "val", "calib", "test")}
    stats = {n: stats_maps(dd, n, data[n])
             for n in ("train", "val", "calib", "test")}
    teacher = None
    if args.teacher:
        P_tr = torch.tensor(np.load(dd / "ch4t_train_posterior.npz")
                            ["probs"].astype(np.float32))
        teacher = blur_teacher(P_tr, N_GRID, 0.75, device=device).to(device)
    torch.manual_seed(6000 + args.seed + 900)
    rng = np.random.default_rng(args.seed)
    model = NPEFlowPhys(cfg).to(device)
    print(f"NPEFlowPhys params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    d_train, d_val = data["train"], data["val"]
    n_train = len(d_train["ids"])
    cell_w = 1.0 / N_GRID
    best, best_state = np.inf, None

    def attach(b, split, idx, codes=None):
        st = stats[split][idx].to(device)
        if codes is not None:
            st = st.reshape(len(idx), 2, N_CELLS).gather(
                2, inv[codes][:, None, :].expand(-1, 2, -1)).reshape(
                len(idx), 2, N_GRID, N_GRID)
        else:
            st = st.view(len(idx), 2, N_GRID, N_GRID)
        b["stats"] = st
        return b

    def val_nll():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for lo in range(0, len(d_val["ids"]), 512):
                idx = np.arange(lo, min(lo + 512, len(d_val["ids"])))
                b = attach(to_batch_t(d_val, idx, device), "val", idx)
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
            batch, xs, codes = d4_augment_t(batch, d_train["xs"][idx],
                                            batch["u_seq"], rng, device)
            batch = attach(batch, "train", idx, codes)
            if teacher is not None:
                tgt = teacher[idx].gather(1, inv[codes])
                cell = torch.multinomial(tgt.clamp_min(0), 1).squeeze(1)
                cx = (cell % N_GRID).float() + 0.5
                cy = (cell // N_GRID).float() + 0.5
                base = torch.stack([cx, cy], -1) / N_GRID
                jit = torch.tensor(
                    rng.uniform(-0.5, 0.5, (len(idx), 2)) * cell_w,
                    dtype=torch.float32, device=device)
                xy = (base + jit).clamp(1e-5, 1 - 1e-5)
            else:
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
        if ep % 25 == 0:
            print(f"  ep {ep} val_nll {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch improved val NLL")
    model.load_state_dict(best_state)
    model.eval()

    edges = torch.linspace(0, 1, N_GRID + 1, device=device)
    centers = (torch.arange(N_GRID, device=device) + 0.5) / N_GRID

    def probs_split(name):
        d = data[name]
        out = []
        for lo in range(0, len(d["ids"]), 256):
            idx = np.arange(lo, min(lo + 256, len(d["ids"])))
            b = attach(to_batch_t(d, idx, device), name, idx)
            out.append(model.cell_probs(b, edges, centers).cpu().numpy())
        P = np.concatenate(out).astype(np.float64)
        P /= P.sum(1, keepdims=True)
        return P

    Pc, Pt = probs_split("calib"), probs_split("test")
    rng_c = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(Pc, data["calib"]["true_cell"], rng_c),
                        cfg["conformal"]["alpha"])
    reg = regions(Pt, th, data["test"]["true_cell"])
    recipe = np.load(dd / "ch4t_audit_lever_suffstats.npz")["sizes"]
    w = wilcoxon(np.log(recipe.astype(float)),
                 np.log(reg["sizes"].astype(float)), alternative="greater")
    tag = f"flow_phys{'_teach' if args.teacher else ''}_seed{args.seed}"
    out = {"tag": tag, "coverage": float(reg["covered"].mean()),
           "median_radius_m": rad_m(reg["sizes"]),
           "wilcoxon_p_sharper_than_recipe": float(w.pvalue),
           "frac_sharper_than_recipe": float((reg["sizes"] < recipe).mean())}
    res_path = rr / "ch4t_flow_phys.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[tag] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    np.savez_compressed(dd / f"ch4t_audit_{tag}.npz", sizes=reg["sizes"])
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
