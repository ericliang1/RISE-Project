"""Wind-robust physics-guided localization under measured wind.

Noisy wind is part of the simulator's output: readings are generated at the
true wind, the dataset records only u_obs (10-deg/10% per-step error), and
models never see the truth.  Physics enters ONLY through input maps; ordinary
CE to the source cell; no auxiliary losses; deployment = one forward pass.

Map kinds (--maps), all closed-form linear algebra at preprocessing:
  det    2ch  z, log b at the reported wind
  smear  2ch  analytically direction-smeared z, log b (zero MC noise)
  ens    3ch  mean z, spread z, mean log b over 8 error-model draws
  ensr   4ch  ens + wind-marginalized residual (goodness-of-fit) channel
  ensp/full   ens + MC-marginal-posterior channel (+ smear/resid variants)
  off         plain DeepSetsT, no physics channels

Dual-condition audit with condition-matched conformal calibration (noisy-only
map kinds audit under measured wind only).

Stages:
  gen / gen-marg / gen-smear         map & residual-channel generation
  verify / verify-marg / verify-smear   gates: closed form vs direct / MC
  train    --views clean|noisy|paired  --maps <kind>  [--seed 1]
Usage examples:
  python src/paired_wind.py --stage gen-marg
  python src/paired_wind.py --stage train --views noisy --maps ensr --seed 2
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from common import cell_centers, get_device, load_config, pos_to_cell, resolve
from conformal import region_mask, regions, tail_scores, tail_threshold
from methane_t import DeepSetsT, N_GRID, SPLITS, cell_responses_t, \
    d4_augment_t, stab_of, to_batch_t
from methane_model import plume_ppm_per_kgh_smeared
from methane_t import T as T_STEPS
from methane_t_levers import stats_maps
from methane_t_uncertain import PhysHeadNet, SIG_TH, ab_maps, logbmap, \
    perturb_wind, split_tag, with_obs_wind, zmap
from model import smoothed_targets
from train import d4_target_perms

N_CELLS = N_GRID * N_GRID
R_MAX = 10.0
K_MARG = 8       # wind draws for the marginalized residual (stream 93)
# map kinds that only exist under the measured-wind condition
NOISY_ONLY_MAPS = ("ens", "ensr", "smear", "ensp", "full", "rand",
                   "zdet")


def phys_head_on(base_cls, n_ch):
    """Zero-init conv head over n_ch physics images, on any base
    localizer whose forward takes the batch dict (mirrors PhysHeadNet)."""

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


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * 500)


def yy_nobs(d):
    """Per-scenario sum of squared kept readings and kept count."""
    y = d["readings"] * d["keep"]
    return (y ** 2).sum((1, 2)), d["keep"].sum((1, 2)).astype(np.float64)


# ------------------------------------------------------------------ stage gen
@torch.no_grad()
def stage_gen(cfg, device):
    dd = resolve(cfg, "data_dir")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        u_obs = np.load(dd / f"ch4tu_{name}_uobs.npy")
        path = dd / f"ch4tu_{name}_suffstats.npz"
        if not path.exists():
            n = len(d["ids"])
            A = np.empty((n, N_CELLS), np.float32)
            B = np.empty((n, N_CELLS), np.float32)
            for i in range(n):
                ns = int(d["n_sensors"][i])
                keep = d["keep"][i, :ns]
                g = cell_responses_t(cells, d["sensors"][i, :ns], u_obs[i],
                                     stab_of(d, i), device)
                g = g[:, torch.tensor(keep, device=device)]
                y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                                 dtype=torch.float64)
                A[i] = (g @ y).float().cpu().numpy()
                B[i] = (g * g).sum(1).float().cpu().numpy()
                if (i + 1) % 2000 == 0:
                    print(f"  {name} noisy-ss {i+1}/{n}", flush=True)
            np.savez_compressed(path, a=A, b=B)
            print(f"  {name}: noisy suffstats saved", flush=True)
        # residual maps for both views
        rpath = dd / f"pw_{name}_resid.npz"
        if not rpath.exists():
            yy, nob = yy_nobs(d)
            out = {}
            for view, ssfile in (("clean", f"ch4t_{name}_suffstats.npz"),
                                 ("noisy", f"ch4tu_{name}_suffstats.npz")):
                z = np.load(dd / ssfile)
                a = z["a"].astype(np.float64)
                b = z["b"].astype(np.float64)
                qh = np.maximum(0.0, a / (b + 1e-30))
                R = yy[:, None] - 2 * qh * a + qh ** 2 * b
                r = R / np.maximum(nob, 1)[:, None]
                r = r - r.min(1, keepdims=True)
                med = np.median(r, axis=1, keepdims=True) + 1e-12
                out[view] = np.minimum(r / med, R_MAX).astype(np.float16)
            np.savez_compressed(rpath, clean=out["clean"],
                                noisy=out["noisy"])
            print(f"  {name}: residual maps saved", flush=True)


# ------------------------------------------------------------ stage gen-smear
@torch.no_grad()
def cell_responses_t_smeared(cells, sensors, u_seq, stab, device, chunk=512):
    """cell_responses_t with the direction-error-smeared plume."""
    C, S = cells.shape[0], sensors.shape[0]
    out = torch.empty((C, S, T_STEPS), device=device, dtype=torch.float64)
    uu = torch.as_tensor(u_seq, dtype=torch.float64, device=device)
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        c = hi - lo
        cc = cells[lo:hi][:, None, None, :].expand(c, S, T_STEPS, 2)
        s2 = ss[None, :, None, :].expand(c, S, T_STEPS, 2)
        u2 = uu[None, None, :, :].expand(c, S, T_STEPS, 2)
        st = torch.full((c, S, T_STEPS), int(stab), dtype=torch.long,
                        device=device)
        out[lo:hi] = plume_ppm_per_kgh_smeared(s2, cc, u2, st, SIG_TH)
    return out


@torch.no_grad()
def stage_gen_smear(cfg, device):
    """Analytically wind-smeared maps at u_obs: the matched filter of the
    EXPECTED fingerprint under the per-step direction-error model.  Same
    marginal the ens mean channel estimates with 8 MC draws, but in closed
    form -- zero MC noise, det-map cost."""
    dd = resolve(cfg, "data_dir")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        path = dd / f"ch4tu_{name}_maps_smear.npz"
        if path.exists():
            continue
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        u_obs = np.load(dd / f"ch4tu_{name}_uobs.npy")
        n = len(d["ids"])
        A = np.empty((n, N_CELLS), np.float64)
        B = np.empty((n, N_CELLS), np.float64)
        for i in range(n):
            ns = int(d["n_sensors"][i])
            keep = d["keep"][i, :ns]
            g = cell_responses_t_smeared(cells, d["sensors"][i, :ns],
                                         u_obs[i], stab_of(d, i), device)
            g = g[:, torch.tensor(keep, device=device)]
            y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                             dtype=torch.float64)
            A[i] = (g @ y).cpu().numpy()
            B[i] = (g * g).sum(1).cpu().numpy()
            if (i + 1) % 2000 == 0:
                print(f"  {name}: smear {i+1}/{n}", flush=True)
        maps = np.stack([zmap(A, B, d["sigma"]), logbmap(B)], 1)
        np.savez_compressed(path, maps=maps.astype(np.float32))
        print(f"  {name}: smear maps saved", flush=True)


# --------------------------------------------------------- stage verify-smear
@torch.no_grad()
def stage_verify_smear(cfg, device, n_draws=256):
    """MC check of the analytic smearing: per-cell a-statistic of the
    smeared fingerprint vs the mean over n_draws direction-only draws of the
    sharp plume (direction only: speed error cancels in z and is not
    smeared).  MC noise at 256 draws is ~1-2%, so tolerances are loose."""
    dd = resolve(cfg, "data_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    u_obs = np.load(dd / "ch4tu_test_uobs.npy")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    rng = np.random.default_rng(2)
    for i in rng.integers(0, len(d["ids"]), 2):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        g_sm = cell_responses_t_smeared(cells, d["sensors"][i, :ns],
                                        u_obs[i], stab_of(d, i), device)
        a_sm = (g_sm[:, torch.tensor(keep, device=device)] @ y).cpu().numpy()
        acc = np.zeros(N_CELLS)
        for _ in range(n_draws):
            th = rng.normal(0, SIG_TH, u_obs[i].shape[0])
            ck, sk = np.cos(th), np.sin(th)
            uk = np.stack([u_obs[i][:, 0] * ck - u_obs[i][:, 1] * sk,
                           u_obs[i][:, 0] * sk + u_obs[i][:, 1] * ck], 1)
            g = cell_responses_t(cells, d["sensors"][i, :ns], uk,
                                 stab_of(d, i), device)
            acc += (g[:, torch.tensor(keep, device=device)] @ y).cpu().numpy()
        a_mc = acc / n_draws
        top = a_mc >= np.quantile(a_mc, 0.9)
        rel = np.abs(a_sm[top] - a_mc[top]) / (np.abs(a_mc[top]) + 1e-12)
        corr = float(np.corrcoef(a_sm, a_mc)[0, 1])
        print(f"  scen {i}: corr {corr:.4f}, top-decile median rel err "
              f"{float(np.median(rel)):.3f}", flush=True)
        assert corr > 0.99 and np.median(rel) < 0.05, "smear mismatch"


# ------------------------------------------------------------- stage gen-marg
@torch.no_grad()
def stage_gen_marg(cfg, device, k_draws=K_MARG):
    """Wind-marginalized residual for the noisy view: MEAN closed-form
    residual over k_draws winds drawn from the error model centered at u_obs
    (streams [root, 93, split_tag, i, k]; independent of the ens-maps 92
    stream).  The plain residual scores fit at the single misreported wind;
    this one scores expected fit over the winds the error model says are
    plausible, so the true cell is no longer penalized for the wind error."""
    dd = resolve(cfg, "data_dir")
    root = cfg["seeds"]["root_entropy"]
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        path = dd / f"pw_{name}_resid_marg.npz"
        if path.exists():
            continue
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        u_obs = np.load(dd / f"ch4tu_{name}_uobs.npy")
        n = len(d["ids"])
        tag = split_tag(name)
        yy, nob = yy_nobs(d)
        Rsum = np.zeros((n, N_CELLS), np.float64)
        for k in range(k_draws):
            uk = np.stack([perturb_wind(u_obs[i],
                                        np.random.default_rng(
                                            [root, 93, tag, i, k]))
                           for i in range(n)])
            A, B = ab_maps(d, uk, cells, device)
            qh = np.maximum(0.0, A / (B + 1e-30))
            Rsum += yy[:, None] - 2 * qh * A + qh ** 2 * B
            print(f"  {name}: marg draw {k+1}/{k_draws}", flush=True)
        r = (Rsum / k_draws) / np.maximum(nob, 1)[:, None]
        r = r - r.min(1, keepdims=True)
        med = np.median(r, axis=1, keepdims=True) + 1e-12
        np.savez_compressed(path, noisy=np.minimum(r / med, R_MAX)
                            .astype(np.float16))
        print(f"  {name}: marginalized residual saved", flush=True)


# ---------------------------------------------------------- stage verify-marg
@torch.no_grad()
def stage_verify_marg(cfg, device, k_draws=K_MARG):
    """End-to-end recompute of the marginalized residual for sampled test
    scenarios with the same streams, vs the saved fp16 rows; plus closed-form
    vs direct ||y - qhat g||^2 on sampled cells of every draw."""
    dd = resolve(cfg, "data_dir")
    root = cfg["seeds"]["root_entropy"]
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    u_obs = np.load(dd / "ch4tu_test_uobs.npy")
    tag = split_tag("test")
    saved = np.load(dd / "pw_test_resid_marg.npz")["noisy"]
    yy, nob = yy_nobs(d)
    rng = np.random.default_rng(1)
    worst_cell, worst_row = 0.0, 0.0
    for i in rng.integers(0, len(d["ids"]), 3):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        Rbar = np.zeros(N_CELLS)
        for k in range(k_draws):
            uk = perturb_wind(u_obs[i],
                              np.random.default_rng([root, 93, tag, i, k]))
            g = cell_responses_t(cells, d["sensors"][i, :ns], uk,
                                 stab_of(d, i), device)
            g = g[:, torch.tensor(keep, device=device)]
            a = (g @ y).cpu().numpy()
            b = (g * g).sum(1).cpu().numpy()
            qh = np.maximum(0.0, a / (b + 1e-30))
            R = yy[i] - 2 * qh * a + qh ** 2 * b
            for c in rng.integers(0, N_CELLS, 5):
                direct = float(((y - float(qh[c]) * g[c]) ** 2).sum())
                worst_cell = max(worst_cell,
                                 abs(R[c] - direct) / max(direct, 1e-9))
            Rbar += R
        r = (Rbar / k_draws) / max(float(nob[i]), 1.0)
        r = r - r.min()
        r = np.minimum(r / (np.median(r) + 1e-12), R_MAX)
        worst_row = max(worst_row,
                        float(np.abs(r - saved[i].astype(np.float64)).max()))
    print(f"marg verify: closed-vs-direct worst rel {worst_cell:.2e}, "
          f"saved-row worst abs {worst_row:.2e}", flush=True)
    assert worst_cell < 1e-4 and worst_row < 2e-2, "marg residual mismatch"


# --------------------------------------------------------------- stage verify
@torch.no_grad()
def stage_verify(cfg, device):
    """Closed-form R_c vs direct ||y - qhat g_c||^2 on sampled cells."""
    dd = resolve(cfg, "data_dir")
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    z = np.load(dd / "ch4t_test_suffstats.npz")
    yy, nob = yy_nobs(d)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in rng.integers(0, len(d["ids"]), 5):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g_all = cell_responses_t(cells, d["sensors"][i, :ns], d["u_seq"][i],
                                 stab_of(d, i), device)
        g_all = g_all[:, torch.tensor(keep, device=device)]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        a = z["a"][i].astype(np.float64)
        b = z["b"][i].astype(np.float64)
        for c in rng.integers(0, N_CELLS, 20):
            qh = max(0.0, a[c] / (b[c] + 1e-30))
            R_closed = yy[i] - 2 * qh * a[c] + qh ** 2 * b[c]
            R_direct = float(((y - qh * g_all[c]) ** 2).sum())
            worst = max(worst, abs(R_closed - R_direct)
                        / max(abs(R_direct), 1e-9))
    print(f"residual closed-form vs direct: worst rel err {worst:.2e}",
          flush=True)
    assert worst < 1e-4, "closed-form residual mismatch"


# ---------------------------------------------------------------- stage train
def make_view(cfg, dd, name_data, view, maps="det"):
    """(data dict, input maps tensor or None).
    maps: "det" (2ch at u_obs), "ens" (3ch over wind draws), "ensr" (ens +
    marginalized-residual channel), or None (no physics-map channels)."""
    split, d_clean = name_data
    if view == "clean":
        d = d_clean
        assert maps in ("det", None), "ens/ensr maps are noisy-view only"
        stats = stats_maps(dd, split, d_clean) if maps else None
    else:
        d = with_obs_wind(d_clean,
                          np.load(dd / f"ch4tu_{split}_uobs.npy"))
        if maps == "det":
            stats = torch.tensor(
                np.load(dd / f"ch4tu_{split}_maps_det.npz")["maps"])
        elif maps == "zdet":
            # evidence image alone (first det channel)
            stats = torch.tensor(
                np.load(dd / f"ch4tu_{split}_maps_det.npz")["maps"][:, :1])
        elif maps == "rand":
            # negative control: 4 random channels, same shape/scale class
            # as the physics images, deterministic per split+scenario
            n = len(d_clean["ids"])
            rng_r = np.random.default_rng([4242, split_tag(split)])
            stats = torch.tensor(rng_r.normal(0.5, 0.3, (n, 4, N_CELLS))
                                 .astype(np.float32))
        elif maps == "smear":
            stats = torch.tensor(
                np.load(dd / f"ch4tu_{split}_maps_smear.npz")["maps"])
        elif maps in ("ens", "ensr", "ensp", "full"):
            ens = torch.tensor(
                np.load(dd / f"ch4tu_{split}_maps_ens.npz")["maps"])
            if maps == "ens":
                stats = ens
            elif maps == "ensr":
                r = np.load(dd / f"pw_{split}_resid_marg.npz")["noisy"]
                r = torch.tensor(r.astype(np.float32) / R_MAX)[:, None, :]
                stats = torch.cat([ens, r], 1)
            else:
                P = np.load(dd / f"ch4tu_{split}_oracle_marg.npz")["probs"]
                lp = torch.tensor(((np.log10(P + 1e-30) + 30.0) / 30.0)
                                  .astype(np.float32))[:, None, :]
                if maps == "ensp":
                    stats = torch.cat([ens, lp], 1)
                else:   # full: smeared evidence + spread + resid + marg post
                    sm = torch.tensor(np.load(
                        dd / f"ch4tu_{split}_maps_smear.npz")["maps"])
                    r = np.load(dd / f"pw_{split}_resid_marg.npz")["noisy"]
                    r = torch.tensor(r.astype(np.float32) / R_MAX)[:, None, :]
                    stats = torch.cat([sm, ens[:, 1:2], r, lp], 1)
        else:
            stats = None
    return d, stats


def stage_train(cfg, device, views, seed, maps="det", arch="deepsets"):
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    m = cfg["model"]
    alpha = cfg["conformal"]["alpha"]
    clean = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))
             for s in SPLITS}
    view_list = (["clean", "noisy"] if views == "paired" else [views])
    if maps in NOISY_ONLY_MAPS:
        assert views == "noisy", f"{maps} configs are noisy-view only"
    V = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps)
             for v in view_list} for s in SPLITS}
    # audit conditions; these map kinds only exist under measured wind
    conds = (("noisy",) if maps in NOISY_ONLY_MAPS
             else ("clean", "noisy"))
    A = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps)
             for v in conds} for s in ("calib", "test")}
    torch.manual_seed(6000 + seed + 1300)
    rng = np.random.default_rng(seed)
    n_ch = V["train"][view_list[0]][1].shape[1] if maps else 0
    if arch == "gnn":
        from methane_t_gnn import GNNT as base_cls
    elif arch == "st":
        from methane_t_settransformer import SetTransformerT as base_cls
    else:
        base_cls = DeepSetsT
    if not maps:
        model = base_cls(cfg).to(device)
    elif arch == "deepsets":
        model = PhysHeadNet(cfg, n_ch).to(device)
    else:
        model = phys_head_on(base_cls, n_ch)(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    n_train = len(clean["train"]["ids"])
    best, best_state = np.inf, None

    def fwd_loss(v, idx):
        d, stats = V["train"][v]
        batch = to_batch_t(d, idx, device)
        batch, xs, codes = d4_augment_t(batch, d["xs"][idx],
                                        batch["u_seq"], rng, device)
        gi = inv[codes]
        if maps:
            st = stats[idx].to(device)
            batch["stats"] = st.reshape(len(idx), n_ch, N_CELLS).gather(
                2, gi[:, None, :].expand(-1, n_ch, -1)).reshape(
                len(idx), n_ch, N_GRID, N_GRID)
        tc = torch.tensor(pos_to_cell(xs, N_GRID))
        tgt = smoothed_targets(tc, N_GRID, m["target_smooth_std_cells"],
                               m["target_trunc_sigmas"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        logp = torch.log_softmax(logits.float(), -1)
        return -(tgt * logp).sum(1).mean()

    @torch.no_grad()
    def probs(split_views, split, v):
        d, stats = split_views[split][v]
        out = []
        for lo in range(0, len(d["ids"]), 512):
            idx = np.arange(lo, min(lo + 512, len(d["ids"])))
            b = to_batch_t(d, idx, device)
            if stats is not None:
                b["stats"] = stats[idx].to(device).view(len(idx), -1, N_GRID,
                                                        N_GRID)
            out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    for ep in range(200):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            loss = sum(fwd_loss(v, idx) for v in view_list) / len(view_list)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError("non-finite loss/grad on first epoch")
            opt.step()
        sched.step()
        model.eval()
        tot = 0.0
        for v in view_list:
            d, stats = V["val"][v]
            with torch.no_grad():
                for lo in range(0, len(d["ids"]), 512):
                    idx = np.arange(lo, min(lo + 512, len(d["ids"])))
                    b = to_batch_t(d, idx, device)
                    if stats is not None:
                        b["stats"] = stats[idx].to(device).view(
                            len(idx), -1, N_GRID, N_GRID)
                    logp = torch.log_softmax(model(b).float(), -1)
                    tc = torch.tensor(d["true_cell"][idx], device=device)
                    tot += float(-logp.gather(1, tc[:, None]).sum())
        crit = tot / (len(V["val"][view_list[0]][0]["ids"]) * len(view_list))
        if crit < best:
            best = crit
            best_state = {k: v_.detach().clone()
                          for k, v_ in model.state_dict().items()}
        if ep % 25 == 0:
            print(f"  ep {ep} val_nll {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch improved")
    model.load_state_dict(best_state)
    model.eval()

    # dual-condition audit, condition-matched calibration
    mt = {"det": "", "ens": "_ens", "ensr": "_ensr", "smear": "_smear",
          "ensp": "_ensp", "full": "_full", "rand": "_rand",
          "zdet": "_zdet",
          None: "_nomaps"}[maps]
    at = "" if arch == "deepsets" else f"_{arch}"
    tag = f"pw_{views}{mt}{at}_seed{seed}"
    out = {"tag": tag, "maps": maps or "off", "arch": arch,
           "val_nll": float(best)}
    if os.environ.get("PW_SAVE_CKPT"):
        ck = dd / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save({"model_state": model.state_dict(), "maps": maps or "off",
                    "arch": arch, "seed": seed, "val_nll": float(best)},
                   ck / f"{tag}.pt")
        print(f"saved checkpoint {ck / f'{tag}.pt'}", flush=True)
    for cond in conds:
        Pc = probs(A, "calib", cond)
        Pt = probs(A, "test", cond)
        rngc = np.random.default_rng(cfg["conformal"]["score_seed"])
        d_cal = A["calib"][cond][0]
        d_tst = A["test"][cond][0]
        th = tail_threshold(tail_scores(Pc, d_cal["true_cell"], rngc), alpha)
        reg = regions(Pt, th, d_tst["true_cell"])
        radii = np.sqrt(reg["sizes"] / N_CELLS / np.pi) * 500
        out[cond] = {"coverage": float(reg["covered"].mean()),
                     "median_radius_m": rad_m(reg["sizes"]),
                     "frac_below_50m": float((radii < 50).mean())}
        masks = np.stack([region_mask(Pt[i], th) for i in range(len(Pt))])
        np.savez_compressed(dd / f"pw_audit_{tag}_{cond}.npz",
                            sizes=reg["sizes"], covered=reg["covered"],
                            masks=np.packbits(masks, axis=1))
    res_path = rr / "paired_wind.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[tag] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gen", "gen-marg", "gen-smear", "verify",
                             "verify-marg", "verify-smear", "train"])
    ap.add_argument("--views", choices=["clean", "noisy", "paired"],
                    default="paired")
    ap.add_argument("--maps", choices=["on", "det", "smear", "ens", "ensr",
                                       "ensp", "full", "rand", "zdet",
                                       "off"],
                    default="on",
                    help="det (=on): 2ch at u_obs; smear: 2ch analytically "
                         "wind-smeared; ens: 3ch over wind draws; ensr: ens "
                         "+ marg-residual; ensp: ens + marg-posterior; "
                         "full: smear + spread + marg-posterior; off: plain "
                         "DeepSetsT")
    ap.add_argument("--arch", choices=["deepsets", "gnn", "st"],
                    default="deepsets")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    if args.stage == "gen":
        stage_gen(cfg, device)
    elif args.stage == "gen-marg":
        stage_gen_marg(cfg, device)
    elif args.stage == "gen-smear":
        stage_gen_smear(cfg, device)
    elif args.stage == "verify":
        stage_verify(cfg, device)
    elif args.stage == "verify-marg":
        stage_verify_marg(cfg, device)
    elif args.stage == "verify-smear":
        stage_verify_smear(cfg, device)
    else:
        stage_train(cfg, device, args.views, args.seed,
                    maps={"on": "det", "off": None}.get(args.maps,
                                                        args.maps),
                    arch=args.arch)


if __name__ == "__main__":
    main()
