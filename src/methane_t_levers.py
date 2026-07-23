"""Three levers against the CH4-T amortization gap, one change each vs M1.

Diagnosis (results/ch4t_gap_decomposition.json): the tempered teacher itself
conformalizes at 11.7 m vs the students' 50-74 m, so the dominant term is the
student FAILING TO MATCH its teacher, not the tempering ceiling (7 -> 11.7 m).

  anneal    A: temper annealed 0.75 -> 0.15 cells (attacks the small term;
               cheap falsifiable control -- predicted gain <= ~5 m)
  sinkhorn  B: Sinkhorn-divergence loss to the RAW posterior (spatially aware
               loss; attacks trainability-at-sharpness AND fit failure)
  suffstats C: per-cell likelihood statistics (a, b) as conv input channels,
               residual logits (physics-guided amortization; attacks fit
               failure directly)

Harness is byte-identical to the campaign audit: same frozen splits, same
score_seed, same conformal code path, float64.  One lever at a time, seed 1.

Usage:
  python src/methane_t_levers.py --stage anneal|sinkhorn|gen_suffstats|suffstats
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon

from common import cell_centers, get_device, load_config, pos_to_cell, resolve
from conformal import regions, tail_scores, tail_threshold
from methane_t import (DeepSetsT, N_GRID, T, cell_responses_t, d4_augment_t,
                       stab_of, to_batch_t)
from train import blur_teacher, d4_target_perms

L = 500.0
N_CELLS = N_GRID * N_GRID


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * L)


def load_all(cfg):
    dd = resolve(cfg, "data_dir")
    data = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))
            for s in ("train", "val", "calib", "test")}
    P_ex = {s: np.load(dd / f"ch4t_{s}_posterior.npz")["probs"]
            for s in ("train", "val", "calib", "test")}
    return dd, data, P_ex


# ------------------------------------------------------------- audit (frozen)
def audit(probs_calib, probs_test, data, cfg, e_sizes, m1_sizes, tag, dd, rr):
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    alpha = cfg["conformal"]["alpha"]
    th = tail_threshold(tail_scores(probs_calib, data["calib"]["true_cell"],
                                    rng), alpha)
    reg = regions(probs_test, th, data["test"]["true_cell"])
    np.savez_compressed(dd / f"ch4t_audit_{tag}.npz", sizes=reg["sizes"],
                        exact_sizes=e_sizes)
    r1 = reg["sizes"] / np.maximum(e_sizes, 1)
    r0 = m1_sizes / np.maximum(e_sizes, 1)
    w = wilcoxon(np.log(r0), np.log(r1), alternative="greater")
    out = {"tag": tag, "coverage": float(reg["covered"].mean()),
           "median_radius_m": rad_m(reg["sizes"]),
           "wilcoxon_p_vs_M1": float(w.pvalue),
           "frac_sharper_than_M1": float((reg["sizes"] < m1_sizes).mean())}
    print(json.dumps(out), flush=True)
    res_path = rr / "ch4t_levers.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[tag] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    return out


# ---------------------------------------------------- val criterion (area)
@torch.no_grad()
def val_area_crit(model, d, device, stats=None, floor=0.85):
    """Median raw-HPD-0.90 area on val with a coverage floor: matches the
    audited quantity when the training objective is moving (anneal) or is not
    a CE (sinkhorn).  Lesson from the ADV-2D curriculum: teacher-CE selects
    blurry mid-anneal students."""
    P = probs_of(model, d, device, stats)
    order = np.argsort(-P, axis=1)
    ps = np.take_along_axis(P, order, 1)
    k = 1 + (np.cumsum(ps, 1) < 0.90).sum(1)
    areas = k / N_CELLS
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(N_CELLS)[None, :], 1)
    covered = ranks[np.arange(len(k)), d["true_cell"]] < k
    if covered.mean() < floor:
        return np.inf
    return float(np.median(areas))


@torch.no_grad()
def probs_of(model, d, device, stats=None):
    out = []
    n = len(d["ids"])
    for lo in range(0, n, 512):
        idx = np.arange(lo, min(lo + 512, n))
        b = to_batch_t(d, idx, device)
        if stats is not None:
            b["stats"] = stats[idx].to(device)
        out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ------------------------------------------------------------------ sinkhorn
def make_kernel(sigma_cells, device):
    r = int(np.ceil(4 * sigma_cells))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-x ** 2 / (2 * sigma_cells ** 2))
    return (k / k.sum()), r


def kconv(v, k1d, r):
    """Separable Gaussian K applied to (B,4096) maps."""
    B = v.shape[0]
    img = v.view(B, 1, N_GRID, N_GRID)
    img = F.conv2d(img, k1d.view(1, 1, 1, -1), padding=(0, r))
    img = F.conv2d(img, k1d.view(1, 1, -1, 1), padding=(r, 0))
    return img.reshape(B, N_CELLS)


def ot_eps(p, q, k1d, r, eps, iters=25):
    """Entropic OT dual value <p,f>+<q,g>, separable kernel, linear domain.

    The fixed-point iterations run under no_grad on detached inputs and the
    loss is the envelope-theorem form: gradients flow only LINEARLY through
    p and q against detached potentials f, g.  Backprop through the unrolled
    loop overflows float32 (1/denom^2 up to 1e50) and NaNs every gradient on
    the first batch -- caught by adversarial review before the run."""
    with torch.no_grad():
        pd, qd = p.detach(), q.detach()
        u = torch.ones_like(pd)
        v = torch.ones_like(qd)
        for _ in range(iters):
            u = (pd / kconv(v, k1d, r).clamp(min=1e-25)).clamp(1e-25, 1e25)
            v = (qd / kconv(u, k1d, r).clamp(min=1e-25)).clamp(1e-25, 1e25)
        f = eps * torch.log(u.clamp(min=1e-30))
        g = eps * torch.log(v.clamp(min=1e-30))
    return (torch.where(p > 0, p * f, torch.zeros_like(p)).sum(1)
            + (q * g).sum(1))


def sinkhorn_div(p, q, k1d, r, eps):
    """Debiased Sinkhorn divergence; p is the (detached) target."""
    return (ot_eps(p, q, k1d, r, eps)
            - 0.5 * ot_eps(q, q, k1d, r, eps)).mean()   # -0.5*OT(p,p) is const


# --------------------------------------------------------------- suffstats
@torch.no_grad()
def gen_suffstats(cfg, device):
    dd, data, _ = load_all(cfg)
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for split, d in data.items():
        path = dd / f"ch4t_{split}_suffstats.npz"
        if path.exists():
            print(f"  {split}: exists, skip", flush=True)
            continue
        n = len(d["ids"])
        A = np.empty((n, N_CELLS), np.float32)
        B = np.empty((n, N_CELLS), np.float32)
        for i in range(n):
            ns = int(d["n_sensors"][i])
            keep = d["keep"][i, :ns]
            g = cell_responses_t(cells, d["sensors"][i, :ns], d["u_seq"][i],
                                 stab_of(d, i), device)
            g = g[:, torch.tensor(keep, device=device)]
            y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                             dtype=torch.float64)
            A[i] = (g @ y).float().cpu().numpy()
            B[i] = (g * g).sum(1).float().cpu().numpy()
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i+1}/{n}", flush=True)
        np.savez_compressed(path, a=A, b=B)
        print(f"  {split}: saved", flush=True)


def stats_maps(dd, split, d):
    """(N,2,4096) float32 features: matched-filter z and log-b."""
    z = np.load(dd / f"ch4t_{split}_suffstats.npz")
    a, b = z["a"].astype(np.float64), z["b"].astype(np.float64)
    sig = d["sigma"][:, None]
    zmap = np.clip(a / (sig * np.sqrt(b) + 1e-30), -60, 60) / 10.0
    logb = (np.log10(b + 1e-30) + 8.0) / 6.0
    return torch.tensor(np.stack([zmap, logb], 1), dtype=torch.float32)


class DeepSetsTPhys(DeepSetsT):
    """DeepSetsT + residual conv head over per-cell likelihood statistics.
    Final conv zero-initialized: training starts exactly at the M1 baseline
    and opens the physics channel as it helps."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.phys = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.phys[-1].weight)
        nn.init.zeros_(self.phys[-1].bias)

    def forward(self, b):
        base = super().forward(b)
        s = b["stats"].view(-1, 2, N_GRID, N_GRID)
        return base + self.phys(s).flatten(1)


# ------------------------------------------------------------------ trainer
def train_lever(cfg, data, P_raw_train, P_raw_val, device, mode, stats=None,
                seed=1):
    """mode: 'anneal' (CE, blur 0.75->0.15), 'sinkhorn' (raw target),
    'suffstats' (CE, fixed 0.75 blur, stats input)."""
    tr = cfg["training"]
    torch.manual_seed(6000 + seed + 300)
    rng = np.random.default_rng(seed)
    n_epochs = int(os.environ.get("LEVER_EPOCHS", "200"))   # smoke-test hook
    model = (DeepSetsTPhys(cfg) if mode == "suffstats"
             else DeepSetsT(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    d_train, d_val = data["train"], data["val"]
    n_train = len(d_train["ids"])
    Pt_raw = P_raw_train.to(device)
    if mode == "suffstats":
        teacher = blur_teacher(Pt_raw, N_GRID, 0.75, device=device)
        val_teacher = blur_teacher(P_raw_val.to(device), N_GRID, 0.75,
                                   device=device)
    best, best_state = np.inf, None
    bad_crits = 0
    anneal_ep = max(1, int(n_epochs * 0.6))
    for ep in range(n_epochs):
        if mode == "sinkhorn":
            # eps-scaling: coarse kernel early so far-off mass still gets a
            # gradient, ending at 1.0 cells (reach 4 cells ~ 31 m) -- a
            # 0.5-cell end kernel is truncated at 2 cells, below the residual
            # misalignment it must correct (review finding).
            s_k = 1.0 + (4.0 - 1.0) * 0.5 * (
                1 + np.cos(np.pi * min(ep, anneal_ep) / anneal_ep))
            k1d, r = make_kernel(float(s_k), device)
            eps = 2 * float(s_k) ** 2
        if mode == "anneal":
            s_now = 0.15 + (0.75 - 0.15) * 0.5 * (
                1 + np.cos(np.pi * min(ep, anneal_ep) / anneal_ep))
            # cos(0)=1 -> start 0.75, end 0.15
            teacher = blur_teacher(Pt_raw, N_GRID, float(s_now), device=device)
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            batch = to_batch_t(d_train, idx, device)
            batch, xs, codes = d4_augment_t(batch, d_train["xs"][idx],
                                            batch["u_seq"], rng, device)
            gather_idx = inv[codes]
            if mode == "sinkhorn":
                tgt = Pt_raw[idx].gather(1, gather_idx)
            else:
                tgt = teacher[idx].gather(1, gather_idx)
            if mode == "suffstats":
                st = stats["train"][idx].to(device)
                batch["stats"] = st.reshape(len(idx), 2, N_CELLS).gather(
                    2, gather_idx[:, None, :].expand(-1, 2, -1)).reshape(
                    len(idx), 2, N_GRID, N_GRID)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            logits = logits.float()
            if mode == "sinkhorn":
                q = torch.softmax(logits, -1)
                loss = sinkhorn_div(tgt.detach(), q, k1d, r, eps)
            else:
                logp = torch.log_softmax(logits, -1)
                loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError(
                    f"{mode}: non-finite loss/grad on first epoch "
                    f"(loss={float(loss)}, gnorm={float(gn)}) -- abort now, "
                    f"not after the full run")
            opt.step()
        sched.step()
        if mode == "suffstats":                     # matched to M1 convention
            crit = teacher_ce_val(model, d_val, val_teacher, device,
                                  stats["val"])
        else:                                       # objective moves / not CE
            crit = val_area_crit(model, d_val, device,
                                 stats["val"] if stats else None)
        if np.isfinite(crit):
            bad_crits = 0
        else:
            bad_crits += 1
            if bad_crits >= 15:
                raise RuntimeError(
                    f"{mode}: criterion non-finite for {bad_crits} consecutive "
                    f"epochs (coverage floor never met) -- aborting")
        if crit < best:
            best = crit
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep < 5:
            print(f"  {mode} ep {ep} crit {crit:.5f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"{mode}: no epoch passed the selection criterion "
                           f"(best={best})")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def teacher_ce_val(model, d, P_val_teacher, device, stats=None):
    n, tot = len(d["ids"]), 0.0
    for lo in range(0, n, 512):
        idx = np.arange(lo, min(lo + 512, n))
        b = to_batch_t(d, idx, device)
        if stats is not None:
            b["stats"] = stats[idx].to(device)
        logp = torch.log_softmax(model(b).float(), -1)
        tot += float(-(P_val_teacher[idx] * logp).sum())
    return tot / n


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["anneal", "sinkhorn", "gen_suffstats",
                             "suffstats"])
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    if args.stage == "gen_suffstats":
        gen_suffstats(cfg, device)
        return
    dd, data, P_ex = load_all(cfg)
    rr = resolve(cfg, "results_dir")
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng),
                          cfg["conformal"]["alpha"])
    e_sizes = regions(P_ex["test"], th_e, data["test"]["true_cell"])["sizes"]
    m1z = np.load(dd / "ch4t_audit_M1.npz")
    m1_sizes = m1z["sizes"]
    if "exact_sizes" in m1z.files:      # staleness guard: same frozen test set
        assert np.array_equal(m1z["exact_sizes"], e_sizes), \
            "stored M1 audit was made against different exact regions"
    Pt_raw = torch.tensor(P_ex["train"].astype(np.float32))
    Pv_raw = torch.tensor(P_ex["val"].astype(np.float32))
    stats = None
    if args.stage == "suffstats":
        stats = {s: stats_maps(dd, s, data[s])
                 for s in ("train", "val", "calib", "test")}
    model = train_lever(cfg, data, Pt_raw, Pv_raw, device, args.stage,
                        stats=stats)
    Pc = probs_of(model, data["calib"], device,
                  stats["calib"] if stats else None)
    Pt = probs_of(model, data["test"], device,
                  stats["test"] if stats else None)
    audit(Pc, Pt, data, cfg, e_sizes, m1_sizes, f"lever_{args.stage}", dd, rr)


if __name__ == "__main__":
    main()
