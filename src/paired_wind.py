"""Paired Wind-Robust Physics-Guided Localization.

One shared model, two training views per scenario (clean wind w_true and the
fixed 10-deg/10% observed wind w_obs already used by the robustness track),
ordinary CE to the source cell on both views, plus an optional
expected-physics-residual loss  L_phys = sum_c p(c) rbar_c  where rbar is the
per-example normalized, clipped residual of the best nonnegative rate fit
(closed form: R_c = yy - 2 qhat_c a_c + qhat_c^2 b_c, qhat = max(0, a/b)).
No inference-time ensembling; deployment = one forward pass on measured wind.

Dual-condition audit: each model is evaluated under BOTH conditions with
condition-matched conformal calibration (clean calib for clean eval, noisy
calib for noisy eval).

Also the 2x2 factorial under measured wind (--views noisy):
{--maps on|off} x {--lam 0|>0} = original model / +maps / +physics loss /
+maps+loss, all trained AND evaluated with the observed wind (noisy wind is
part of the simulator's output; models never see the true wind).

Stages:
  gen      noisy-view raw suffstats (a,b) at u_obs + residual maps (both views)
  verify   closed-form residual vs direct reconstruction on sampled cells
  train    --views clean|noisy|paired  --maps on|off  --lam <float>  [--seed 1]
Usage examples:
  python src/paired_wind.py --stage gen
  python src/paired_wind.py --stage train --views noisy --maps off --lam 0.05
"""
import argparse
import json

import numpy as np
import torch

from common import cell_centers, get_device, load_config, pos_to_cell, resolve
from conformal import regions, tail_scores, tail_threshold
from methane_t import DeepSetsT, N_GRID, SPLITS, cell_responses_t, \
    d4_augment_t, stab_of, to_batch_t
from methane_t_levers import stats_maps
from methane_t_uncertain import PhysHeadNet, ab_maps, perturb_wind, \
    split_tag, with_obs_wind
from model import smoothed_targets
from train import d4_target_perms

N_CELLS = N_GRID * N_GRID
R_MAX = 10.0
K_MARG = 8       # wind draws for the marginalized residual (stream 93)


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
def make_view(cfg, dd, name_data, view, maps="det", resid_kind="plain"):
    """(data dict, input maps tensor or None, residual tensor).
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
        elif maps in ("ens", "ensr"):
            stats = torch.tensor(
                np.load(dd / f"ch4tu_{split}_maps_ens.npz")["maps"])
            if maps == "ensr":
                r = np.load(dd / f"pw_{split}_resid_marg.npz")["noisy"]
                r = torch.tensor(r.astype(np.float32) / R_MAX)[:, None, :]
                stats = torch.cat([stats, r], 1)
        else:
            stats = None
    if resid_kind == "marg":
        assert view == "noisy", "marginalized residual is noisy-view only"
        resid = torch.tensor(np.load(dd / f"pw_{split}_resid_marg.npz")
                             ["noisy"].astype(np.float32))
    else:
        resid = torch.tensor(
            np.load(dd / f"pw_{split}_resid.npz")[view].astype(np.float32))
    return d, stats, resid


def stage_train(cfg, device, views, lam, seed, maps="det",
                resid_kind="plain"):
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    m = cfg["model"]
    alpha = cfg["conformal"]["alpha"]
    clean = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))
             for s in SPLITS}
    view_list = (["clean", "noisy"] if views == "paired" else [views])
    if maps in ("ens", "ensr"):
        assert views == "noisy", "ens/ensr configs are noisy-view only"
    V = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps, resid_kind)
             for v in view_list} for s in SPLITS}
    # audit conditions; ens/ensr maps only exist for the noisy condition
    conds = ("noisy",) if maps in ("ens", "ensr") else ("clean", "noisy")
    A = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps)
             for v in conds} for s in ("calib", "test")}
    torch.manual_seed(6000 + seed + 1300)
    rng = np.random.default_rng(seed)
    n_ch = V["train"][view_list[0]][1].shape[1] if maps else 0
    model = (PhysHeadNet(cfg, n_ch) if maps else DeepSetsT(cfg)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    n_train = len(clean["train"]["ids"])
    best, best_state = np.inf, None

    def fwd_loss(v, idx):
        d, stats, resid = V["train"][v]
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
        ce = -(tgt * logp).sum(1).mean()
        if lam > 0:
            rb = resid[idx].to(device).gather(1, gi)
            phys = (torch.softmax(logits.float(), -1) * rb).sum(1).mean()
            return ce + lam * phys
        return ce

    @torch.no_grad()
    def probs(split_views, split, v):
        d, stats, _ = split_views[split][v]
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
            d, stats, _ = V["val"][v]
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
    mt = {"det": "", "ens": "_ens", "ensr": "_ensr", None: "_nomaps"}[maps]
    tag = (f"pw_{views}{mt}{'_marg' if resid_kind == 'marg' else ''}"
           f"_lam{lam}_seed{seed}")
    out = {"tag": tag, "maps": maps or "off", "lam": lam,
           "resid": resid_kind, "val_nll": float(best)}
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
        np.savez_compressed(dd / f"pw_audit_{tag}_{cond}.npz",
                            sizes=reg["sizes"])
    res_path = rr / "paired_wind.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[tag] = out
    with open(res_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gen", "gen-marg", "verify", "verify-marg",
                             "train"])
    ap.add_argument("--views", choices=["clean", "noisy", "paired"],
                    default="paired")
    ap.add_argument("--maps", choices=["on", "det", "ens", "ensr", "off"],
                    default="on",
                    help="det (=on): 2ch at u_obs; ens: 3ch over wind "
                         "draws; ensr: ens + marg-residual channel; "
                         "off: plain DeepSetsT")
    ap.add_argument("--resid", choices=["plain", "marg"], default="plain",
                    help="marg = wind-marginalized residual in the loss")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    if args.stage == "gen":
        stage_gen(cfg, device)
    elif args.stage == "gen-marg":
        stage_gen_marg(cfg, device)
    elif args.stage == "verify":
        stage_verify(cfg, device)
    elif args.stage == "verify-marg":
        stage_verify_marg(cfg, device)
    else:
        stage_train(cfg, device, args.views, args.lam, args.seed,
                    maps={"on": "det", "off": None}.get(args.maps,
                                                        args.maps),
                    resid_kind=args.resid)


if __name__ == "__main__":
    main()
