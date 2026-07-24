"""CH4-T-U2: uncertainty-NATIVE oracle and student.

The wind error is part of the simulator: readings come from the true wind,
the model sees only u_hat = true + anemometer error (10 deg / 10% per step,
iid).  The Bayes-correct evidence marginalizes the error:

  p(y | c, q, u_hat) = E_eps[ prod_{k,i} N(y_ik ; q g_ik(u_hat_k o eps_k)) ]
                     = prod_k E_{eps_k}[ prod_i N(...) ]          (*)

(*) holds because both the per-step errors and the per-step likelihood
factors are independent across k given (c, q).  Each E_{eps_k} is a 2-D
integral (angle, log-speed) -> 3x3 Gauss-Hermite nodes, essentially exact.
This fixes the K=16 joint-MC teacher null of CH4T-U: that estimator sampled
a 60-D space; this one does thirty 2-D integrals.

Rate marginalization: per-cell peak-aware composite Gauss-Legendre panels in
q (peak located on the GH-mean responses), evaluated explicitly since the
wind-marginal likelihood is no longer Gaussian in q.

Stages:
  validate  factorized-GH evidence vs brute-force joint-MC on a few scenarios
  gen       U2 oracle posteriors (all splits) + marginalized maps (3ch)
  oracles   audit U2 oracle (BLOCKING self-cal gate ~0.90)
  train     PhysHeadNet student: marginalized maps + tempered U2 teacher

Usage:  python src/methane_t_u2.py --stage validate|gen|oracles|train [--seed 1]
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import wilcoxon

from common import cell_centers, get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from methane_model import plume_ppm_per_kgh
from methane_t import N_GRID, SPLITS, T, stab_of, to_batch_t
from methane_t_uncertain import (SIG_S, SIG_TH, audit, fp_inputs, perturb_wind,
                                 with_obs_wind)
from train import blur_teacher

L = 500.0
N_CELLS = N_GRID * N_GRID
GH_N = 3                       # Gauss-Hermite nodes per error dim (3x3 = 9)
Q_LO, Q_HI = 10.0, 500.0
NQ_PEAK, NQ_OUT = 24, 12       # q-quadrature: peak panel + outer panels


def gh_nodes():
    """3x3 tensor Gauss-Hermite for the 2-D per-step wind error."""
    x, w = np.polynomial.hermite_e.hermegauss(GH_N)   # weight e^{-t^2/2}
    w = w / w.sum()
    dth = np.repeat(x * SIG_TH, GH_N)
    dls = np.tile(x * SIG_S, GH_N)
    ww = np.repeat(w, GH_N) * np.tile(w, GH_N)
    return dth, dls, ww / ww.sum()


def wind_variants(u_hat):
    """(J, T, 2) deterministic GH wind variants of an observed sequence."""
    dth, dls, ww = gh_nodes()
    th = np.arctan2(u_hat[:, 1], u_hat[:, 0])[None, :] + dth[:, None]
    sp = np.linalg.norm(u_hat, axis=1)[None, :] * np.exp(dls[:, None])
    return np.stack([sp * np.cos(th), sp * np.sin(th)], -1), ww


@torch.no_grad()
def responses_variants(cells, sensors, u_var, stab, device, chunk=256):
    """(J, C, S, T) unit-rate responses for all GH wind variants."""
    J = u_var.shape[0]
    C, S = cells.shape[0], sensors.shape[0]
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)
    uu = torch.as_tensor(u_var, dtype=torch.float64, device=device)
    out = torch.empty((J, C, S, T), device=device, dtype=torch.float64)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        c = hi - lo
        cc = cells[lo:hi][None, :, None, None, :].expand(J, c, S, T, 2)
        s2 = ss[None, None, :, None, :].expand(J, c, S, T, 2)
        u2 = uu[:, None, None, :, :].expand(J, c, S, T, 2)
        st = torch.full((J, c, S, T), int(stab), dtype=torch.long,
                        device=device)
        out[:, lo:hi] = plume_ppm_per_kgh(s2, cc, u2, st)
    return out


@torch.no_grad()
def u2_evidence(d, i, u_hat, cells, device):
    """Wind-marginalized log evidence per cell (C,), via (*) + q quadrature."""
    ns = int(d["n_sensors"][i])
    keep = torch.tensor(d["keep"][i, :ns], device=device)          # (S,T)
    y = torch.tensor(d["readings"][i, :ns], device=device,
                     dtype=torch.float64)                          # (S,T)
    sig = float(d["sigma"][i])
    u_var, ww = wind_variants(u_hat)
    g = responses_variants(cells, d["sensors"][i, :ns], u_var,
                           stab_of(d, i), device)                  # (J,C,S,T)
    g = g * keep[None, None]                                       # masked
    yk = y * keep
    lw = torch.tensor(np.log(ww), device=device)                   # (J,)

    # peak-aware q nodes from GH-mean responses (per-cell least squares)
    gm = (g * torch.exp(lw)[:, None, None, None]).sum(0)           # (C,S,T)
    a = (gm * yk[None]).sum((1, 2))
    b = (gm * gm).sum((1, 2)).clamp_min(1e-300)
    qhat = (a / b).clamp(Q_LO, Q_HI)                               # (C,)
    qsd = (sig / torch.sqrt(b)).clamp_min(1e-6)
    lo = torch.maximum(torch.full_like(qhat, Q_LO), qhat - 8 * qsd)
    hi = torch.minimum(torch.full_like(qhat, Q_HI), qhat + 8 * qsd)
    xs_gl, w_gl = np.polynomial.legendre.leggauss(NQ_PEAK)
    xo_gl, wo_gl = np.polynomial.legendre.leggauss(NQ_OUT)

    def panel(a0, b0, x, w):
        # nodes (C,n) and log weights incl. prior 1/(q log(hi/lo))
        mid, half = (a0 + b0) / 2, (b0 - a0) / 2
        q = mid[:, None] + half[:, None] * torch.tensor(x, device=device)
        lw_ = (torch.log(half.clamp_min(1e-300))[:, None]
               + torch.tensor(np.log(w), device=device)[None, :]
               - torch.log(q.clamp_min(1e-300))
               - np.log(np.log(Q_HI / Q_LO)))
        return q, lw_

    panels = [panel(lo, hi, xs_gl, w_gl),
              panel(torch.full_like(qhat, Q_LO), lo, xo_gl, wo_gl),
              panel(hi, torch.full_like(qhat, Q_HI), xo_gl, wo_gl)]
    yy = float((yk * yk).sum())
    M = float(keep.sum())
    const = -0.5 * M * np.log(2 * np.pi * sig ** 2)
    out = []
    for q, lwq in panels:
        n_nodes = q.shape[1]
        # log m_k for each (node, cell, step): logsumexp over GH variants
        # ll_jk = -(||y_k||^2 - 2 q a_jk + q^2 b_jk) / 2 sig^2 per step
        a_j = (g * yk[None, None]).sum(2)                          # (J,C,T)
        b_j = (g * g).sum(2)                                       # (J,C,T)
        yy_k = (yk * yk).sum(0)                                    # (T,)
        res = []
        for n in range(n_nodes):
            qn = q[:, n][None, :, None]                            # (1,C,1)
            ll = (-(yy_k[None, None] - 2 * qn * a_j + qn ** 2 * b_j)
                  / (2 * sig ** 2)) + lw[:, None, None]
            lm = torch.logsumexp(ll, 0).sum(1)                     # (C,)
            res.append(lm + lwq[:, n])
        out.append(torch.stack(res, 1))
    lo_all = torch.cat(out, 1)                                     # (C, nodes)
    return torch.logsumexp(lo_all, 1) + const


@torch.no_grad()
def u2_posteriors(d, u_obs, cells, device, tag=""):
    n = len(d["ids"])
    P = np.empty((n, N_CELLS), np.float64)
    for i in range(n):
        lp = u2_evidence(d, i, u_obs[i], cells, device)
        P[i] = torch.exp(lp - torch.logsumexp(lp, 0)).cpu().numpy()
        if (i + 1) % 500 == 0:
            print(f"  {tag} u2 posterior {i+1}/{n}", flush=True)
    return P


# ---------------------------------------------------------------- validate
@torch.no_grad()
def stage_validate(cfg, device):
    """Factorized-GH evidence must match brute-force joint-MC."""
    dd = resolve(cfg, "data_dir")
    root = cfg["seeds"]["root_entropy"]
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    u_obs = np.load(dd / "ch4tu_test_uobs.npy")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    from methane_t import cell_responses_t
    for i in (3, 17):
        lp_f = u2_evidence(d, i, u_obs[i], cells, device)
        lp_f = lp_f - torch.logsumexp(lp_f, 0)
        # brute force: joint MC with many full-sequence draws
        K = 4000
        ns = int(d["n_sensors"][i])
        keep = torch.tensor(d["keep"][i, :ns], device=device)
        y = torch.tensor(d["readings"][i, :ns], device=device,
                         dtype=torch.float64) * keep
        sig = float(d["sigma"][i])
        qs = np.exp(np.random.default_rng(0).uniform(
            np.log(Q_LO), np.log(Q_HI), 64))
        acc = torch.full((N_CELLS, 64), -np.inf, device=device,
                         dtype=torch.float64)
        rng = np.random.default_rng([root, 999, i])
        CH = 200
        for j in range(K):
            uj = perturb_wind(u_obs[i], rng)
            g = cell_responses_t(cells, d["sensors"][i, :ns], uj,
                                 stab_of(d, i), device) * keep[None]
            a = (g * y[None]).sum((1, 2))
            b = (g * g).sum((1, 2))
            for lo in range(0, 64, CH):
                q = torch.tensor(qs[lo:lo+CH], device=device)[None, :]
                ll = (-(float((y*y).sum()) - 2*q*a[:, None] + q**2*b[:, None])
                      / (2*sig**2))
                acc[:, lo:lo+CH] = torch.logaddexp(acc[:, lo:lo+CH], ll)
        # MC posterior on the same q samples (importance = prior draws)
        lp_b = torch.logsumexp(acc, 1)
        lp_b = lp_b - torch.logsumexp(lp_b, 0)
        # compare where mass lives
        m = torch.exp(lp_f) > 1e-6
        diff = (lp_f[m] - lp_b[m]).abs().max()
        tv = 0.5 * (torch.exp(lp_f) - torch.exp(lp_b)).abs().sum()
        print(f"  scenario {i}: max|dlogp| on support {float(diff):.3f}, "
              f"TV {float(tv):.4f}", flush=True)


# --------------------------------------------------------------------- gen
@torch.no_grad()
def stage_gen(cfg, device):
    dd = resolve(cfg, "data_dir")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        u_obs = np.load(dd / f"ch4tu_{name}_uobs.npy")
        path = dd / f"ch4tu2_{name}_oracle.npz"
        if not path.exists():
            P = u2_posteriors(d, u_obs, cells, device, tag=name)
            np.savez_compressed(path, probs=P)
            print(f"  {name}: u2 oracle saved", flush=True)
        # marginalized maps: matched filter on GH-mean responses + spread
        mpath = dd / f"ch4tu2_{name}_maps.npz"
        if not mpath.exists():
            n = len(d["ids"])
            A = np.empty((n, N_CELLS), np.float32)
            B = np.empty((n, N_CELLS), np.float32)
            SP = np.empty((n, N_CELLS), np.float32)
            for i in range(n):
                ns = int(d["n_sensors"][i])
                keep = torch.tensor(d["keep"][i, :ns], device=device)
                y = torch.tensor(d["readings"][i, :ns], device=device,
                                 dtype=torch.float64) * keep
                u_var, ww = wind_variants(u_obs[i])
                g = responses_variants(cells, d["sensors"][i, :ns], u_var,
                                       stab_of(d, i), device) * keep[None, None]
                w_t = torch.tensor(ww, device=device)[:, None]
                zs = ((g * y[None, None]).sum((2, 3))
                      / (float(d["sigma"][i])
                         * (g * g).sum((2, 3)).clamp_min(1e-300).sqrt()))
                A[i] = (zs * w_t).sum(0).float().cpu().numpy()      # mean z
                SP[i] = ((zs - (zs * w_t).sum(0)[None]) ** 2 * w_t)\
                    .sum(0).sqrt().float().cpu().numpy()            # spread z
                gm = (g * torch.tensor(ww, device=device)
                      [:, None, None, None]).sum(0)      # GH-mean response
                B[i] = (gm * gm).sum((1, 2)).float().cpu().numpy()
                if (i + 1) % 2000 == 0:
                    print(f"  {name} maps {i+1}/{n}", flush=True)
            zmap = np.clip(A, -60, 60) / 10.0
            spmap = np.clip(SP, 0, 60) / 10.0
            logb = (np.log10(B + 1e-30) + 8.0) / 6.0
            np.savez_compressed(mpath, maps=np.stack(
                [zmap, spmap, logb], 1).astype(np.float32))
            print(f"  {name}: u2 maps saved", flush=True)


# ---------------------------------------------------------------- oracles
def stage_oracles(cfg):
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    data = {}
    for name in ("calib", "test"):
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        data[name] = with_obs_wind(d, np.load(dd / f"ch4tu_{name}_uobs.npy"))
    Pc = np.load(dd / "ch4tu2_calib_oracle.npz")["probs"]
    Pt = np.load(dd / "ch4tu2_test_oracle.npz")["probs"]
    tc = data["test"]["true_cell"]
    pt = Pt[np.arange(len(tc)), tc]
    print(f"  frac p_true==0: {(pt == 0).mean():.3f} "
          f"(det oracle had 0.302)", flush=True)
    sizes = audit(Pc, Pt, data, cfg, "u2_oracle", rr)
    res = json.load(open(rr / "ch4t_uncertain.json"))
    cov = res["u2_oracle"]["coverage"]
    assert 0.85 <= cov <= 0.97, (
        f"U2 oracle conformal coverage {cov:.3f} outside [0.85,0.97] -- "
        "marginalization is wrong, do not train on this teacher")
    print(f"  U2 oracle radius: {res['u2_oracle']['median_radius_m']:.1f} m "
          f"= Bayes limit under wind uncertainty", flush=True)
    return sizes


# ------------------------------------------------------------------ train
def stage_train(cfg, device, seed):
    from methane_t_uncertain import PhysHeadNet, probs_of, train_u
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    data = {}
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        data[name] = with_obs_wind(d, np.load(dd / f"ch4tu_{name}_uobs.npy"))
    stats = {n: torch.tensor(np.load(dd / f"ch4tu2_{n}_maps.npz")["maps"])
             for n in SPLITS}
    P_tr = torch.tensor(np.load(dd / "ch4tu2_train_oracle.npz")
                        ["probs"].astype(np.float32))
    P_val = torch.tensor(np.load(dd / "ch4tu2_val_oracle.npz")
                         ["probs"].astype(np.float32))
    teacher_tr = blur_teacher(P_tr, N_GRID, 0.75, device=device)
    teacher_val = blur_teacher(P_val, N_GRID, 0.75, device=device)
    model = train_u(cfg, data, stats, teacher_tr, teacher_val, device, seed)
    Pc = probs_of(model, data["calib"], device, stats["calib"])
    Pt = probs_of(model, data["test"], device, stats["test"])
    ref_path = dd / "ch4tu_audit_s_ens_det.npz"       # best CH4T-U student
    ref = np.load(ref_path)["sizes"] if ref_path.exists() else None
    tag = "s_u2" + ("" if seed == 1 else f"_seed{seed}")
    audit(Pc, Pt, data, cfg, tag, rr, ref_sizes=ref,
          npz_path=dd / f"ch4tu2_audit_{tag}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["validate", "gen", "oracles", "train"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    if args.stage == "validate":
        stage_validate(cfg, device)
    elif args.stage == "gen":
        stage_gen(cfg, device)
    elif args.stage == "oracles":
        stage_oracles(cfg)
    else:
        stage_train(cfg, device, args.seed)


if __name__ == "__main__":
    main()
