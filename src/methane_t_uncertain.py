"""CH4-T-U: measured-wind uncertainty extension.

The base testbed assumes the wind record is exact.  Here the model (and the
physics maps) only see an OBSERVED wind u_hat = true wind + anemometer/
representativeness error (10 deg direction, 10% speed, iid per step);
readings remain generated from the true wind.

Theory: p(y|c) at u_hat is a misspecified likelihood; the Bayes-correct
evidence marginalizes the wind error, p(y|c) = E_{u~p(u|u_hat)} p(y|c,u).
Conformal converts correctness into sharpness, so the marginalized oracle
should be sharper than the deterministic one at the same verified coverage,
and the same ordering should transfer to distilled students.  That is the
falsifiable prediction; the run decides.

Grid (all same DeepSets+phys-head architecture, CE loss, byte-identical
conformal audit):
  oracle_det   exact posterior at u_hat (misspecified)
  oracle_marg  K=8 Monte-Carlo wind-marginalized posterior
  s_<maps>_<teacher>  2x2: maps in {det (z,logb @ u_hat), ens (mean_z,
                      std_z, mean_logb over K=8 draws)} x teacher in
                      {tempered oracle_det, tempered oracle_marg}

Usage:
  python src/methane_t_uncertain.py --stage gen        # winds, maps, oracles
  python src/methane_t_uncertain.py --stage oracles    # audit both oracles
  python src/methane_t_uncertain.py --stage train --maps ens --teacher marg [--seed 1]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

from common import cell_centers, get_device, load_config, resolve, \
    savez_atomic
from conformal import regions, tail_scores, tail_threshold
from exact_posterior import marginal_log_evidence
from methane_pipeline import ch4_cfg
from methane_t import (DeepSetsT, N_GRID, SPLITS, T, cell_responses_t,
                       d4_augment_t, stab_of, to_batch_t)
from train import blur_teacher, d4_target_perms

L = 500.0
N_CELLS = N_GRID * N_GRID
SIG_TH = np.deg2rad(10.0)      # direction error, per step
SIG_S = 0.10                   # log-speed error, per step
K_MAPS = 8                     # wind draws for ensemble input maps
K_TEACHER = 8                  # wind draws for marginalized oracle: same K
                               # budget as the input maps, so the physics-only
                               # baseline and the method get identical
                               # marginalization effort


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * L)


def split_tag(name):
    return 60 + list(SPLITS).index(name)


def perturb_wind(u_seq, rng):
    """One draw of wind error applied to a (T,2) sequence."""
    th = np.arctan2(u_seq[:, 1], u_seq[:, 0]) + rng.normal(0, SIG_TH, T)
    sp = np.linalg.norm(u_seq, axis=1) * np.exp(rng.normal(0, SIG_S, T))
    return np.stack([sp * np.cos(th), sp * np.sin(th)], 1)


def obs_wind_split(d, name, root):
    """Deterministic observed wind u_hat for every scenario of a split."""
    out = np.empty_like(d["u_seq"])
    tag = split_tag(name)
    for i in range(len(d["ids"])):
        rng = np.random.default_rng([root, 90, tag, i])
        out[i] = perturb_wind(d["u_seq"][i], rng)
    return out


def with_obs_wind(d, u_obs):
    """Split dict whose model-visible wind is the observed one."""
    d2 = dict(d)
    d2["u_seq"] = u_obs
    d2["u_mean"] = u_obs.mean(1)
    return d2


# ------------------------------------------------------------ physics maps
@torch.no_grad()
def ab_maps(d, u_seqs, cells, device):
    """Per-cell (a,b) for scenario dict d under a given wind per scenario.
    u_seqs: (N,T,2).  Returns float64 arrays (N,C) a and b."""
    n = len(d["ids"])
    A = np.empty((n, N_CELLS), np.float64)
    B = np.empty((n, N_CELLS), np.float64)
    for i in range(n):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g = cell_responses_t(cells, d["sensors"][i, :ns], u_seqs[i],
                             stab_of(d, i), device)
        g = g[:, torch.tensor(keep, device=device)]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        A[i] = (g @ y).cpu().numpy()
        B[i] = (g * g).sum(1).cpu().numpy()
    return A, B


def zmap(a, b, sig):
    """Evidence channel: asinh replaces the former hard clip(+-60)/10.
    Identical to z/10 in the small-signal regime, logarithmic (never flat)
    for strong sources -- on the super-emitter population the clip
    saturated ~30% of near-source cells (train-split measurement), handing
    the head a plateau exactly where the peak belongs.  Same
    variance-stabilizing transform the raw reading tokens use."""
    z = a / (sig[:, None] * np.sqrt(b) + 1e-30)
    return np.arcsinh(z / 10.0)


def logbmap(b):
    return (np.log10(b + 1e-30) + 8.0) / 6.0


# ------------------------------------------------------------------ stage gen
@torch.no_grad()
def stage_gen(cfg, cfg_q, device):
    dd = resolve(cfg, "data_dir")
    root = cfg["seeds"]["root_entropy"]
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        n = len(d["ids"])
        tag = split_tag(name)
        u_obs = obs_wind_split(d, name, root)
        np.save(dd / f"ch4tu_{name}_uobs.npy", u_obs)

        if (dd / f"ch4tu_{name}_maps_ens.npz").exists():   # resume after wall clock
            print(f"  {name}: maps exist, skipping", flush=True)
        else:
            _gen_maps(d, u_obs, cells, device, dd, name, tag, root, n)
        if name not in ("calib", "test"):
            continue        # oracles feed only the gate + physics baseline
        if (dd / f"ch4tu_{name}_oracle_marg.npz").exists():
            print(f"  {name}: oracles exist, skipping", flush=True)
            continue
        _gen_oracles(d, u_obs, cells, device, dd, name, tag, root, n, cfg_q)
        print(f"  {name}: oracles done", flush=True)


def _gen_maps(d, u_obs, cells, device, dd, name, tag, root, n):
        # deterministic maps + evidence at u_hat
        A, B = ab_maps(d, u_obs, cells, device)
        det = np.stack([zmap(A, B, d["sigma"]), logbmap(B)], 1)
        savez_atomic(dd / f"ch4tu_{name}_maps_det.npz",
                            maps=det.astype(np.float32))

        # ensemble maps + marginalized evidence over wind draws
        zs = np.empty((K_MAPS, n, N_CELLS), np.float32)
        lb = np.empty((K_MAPS, n, N_CELLS), np.float32)
        for j in range(K_MAPS):
            uj = np.stack([perturb_wind(u_obs[i],
                                        np.random.default_rng(
                                            [root, 92, tag, i, j]))
                           for i in range(n)])
            Aj, Bj = ab_maps(d, uj, cells, device)
            zs[j] = zmap(Aj, Bj, d["sigma"])
            lb[j] = logbmap(Bj)
        ens = np.stack([zs.mean(0), zs.std(0) * 3.0, lb.mean(0)], 1)
        savez_atomic(dd / f"ch4tu_{name}_maps_ens.npz",
                            maps=ens.astype(np.float32))
        print(f"  {name}: maps done", flush=True)


def _gen_oracles(d, u_obs, cells, device, dd, name, tag, root, n, cfg_q):
        # oracle_det: evidence at u_hat; oracle_marg: log-mean-exp over draws
        P_det = np.empty((n, N_CELLS), np.float64)
        P_marg = np.empty((n, N_CELLS), np.float64)
        for i in range(n):
            ns = int(d["n_sensors"][i])
            keep = d["keep"][i, :ns]
            y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                             dtype=torch.float64)
            yy = float(torch.dot(y, y))
            M = y.shape[0]
            sig = float(d["sigma"][i])

            def evidence(u_seq):
                g = cell_responses_t(cells, d["sensors"][i, :ns], u_seq,
                                     stab_of(d, i), device)
                g = g[:, torch.tensor(keep, device=device)]
                a = g @ y
                b = (g * g).sum(1)
                return marginal_log_evidence(a, b, yy, sig, M, cfg_q, device)

            lp = evidence(u_obs[i])
            P_det[i] = torch.exp(lp - torch.logsumexp(lp, 0)).cpu().numpy()
            lps = torch.stack([
                evidence(perturb_wind(u_obs[i],
                                      np.random.default_rng(
                                          [root, 91, tag, i, j])))
                for j in range(K_TEACHER)])
            lm = torch.logsumexp(lps, 0) - np.log(K_TEACHER)
            P_marg[i] = torch.exp(lm - torch.logsumexp(lm, 0)).cpu().numpy()
            if (i + 1) % 1000 == 0:
                print(f"  {name} oracle {i+1}/{n}", flush=True)
        savez_atomic(dd / f"ch4tu_{name}_oracle_det.npz", probs=P_det)
        savez_atomic(dd / f"ch4tu_{name}_oracle_marg.npz", probs=P_marg)


# ------------------------------------------------------------------ audits
def fp_inputs(dd, maps_kind, teacher_kind):
    """Provenance fingerprint of the artifacts that determine an audit, so a
    partial rerun can never silently pair against stale references."""
    out = {}
    for f in (f"ch4tu_test_maps_{maps_kind}.npz",
              f"ch4tu_calib_maps_{maps_kind}.npz",
              f"ch4tu_test_oracle_{teacher_kind}.npz",
              f"ch4tu_calib_oracle_{teacher_kind}.npz",
              "ch4tu_test_uobs.npy"):
        out[f] = hashlib.sha256((dd / f).read_bytes()).hexdigest()[:16]
    return out


def audit(probs_calib, probs_test, data, cfg, tag, rr, ref_sizes=None,
          npz_path=None, fp=None):
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(probs_calib,
                                    data["calib"]["true_cell"], rng),
                        cfg["conformal"]["alpha"])
    reg = regions(probs_test, th, data["test"]["true_cell"])
    out = {"tag": tag, "coverage": float(reg["covered"].mean()),
           "median_radius_m": rad_m(reg["sizes"])}
    if ref_sizes is not None:
        w = wilcoxon(np.log(ref_sizes.astype(float)),
                     np.log(reg["sizes"].astype(float)),
                     alternative="greater")
        out["wilcoxon_p_vs_ref"] = float(w.pvalue)
        out["frac_sharper_than_ref"] = float((reg["sizes"]
                                              < ref_sizes).mean())
    if npz_path is not None:                 # npz BEFORE json: no state where
        np.savez_compressed(npz_path,        # json claims an absent audit
                            sizes=reg["sizes"], fp=json.dumps(fp or {}))
    res_path = rr / "ch4t_uncertain.json"
    all_res = json.load(open(res_path)) if res_path.exists() else {}
    all_res[tag] = out
    tmp = res_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(all_res, f, indent=2)
    os.replace(tmp, res_path)
    print(json.dumps(out), flush=True)
    return reg["sizes"]


# ------------------------------------------------------------------ model
class PhysHeadNet(DeepSetsT):
    """DeepSetsT + zero-init conv residual head over n_ch physics maps."""

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


@torch.no_grad()
def probs_of(model, d, device, stats):
    out = []
    n = len(d["ids"])
    for lo in range(0, n, 512):
        idx = np.arange(lo, min(lo + 512, n))
        b = to_batch_t(d, idx, device)
        b["stats"] = stats[idx].to(device).view(len(idx), -1, N_GRID, N_GRID)
        out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_u(cfg, data, stats, teacher_tr, teacher_val, device, seed):
    tr = cfg["training"]
    torch.manual_seed(6000 + seed + 500)
    rng = np.random.default_rng(seed)
    n_ch = stats["train"].shape[1]
    model = PhysHeadNet(cfg, n_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    d_train, d_val = data["train"], data["val"]
    n_train = len(d_train["ids"])
    best, best_state = np.inf, None
    for ep in range(200):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            batch = to_batch_t(d_train, idx, device)
            batch, xs, codes = d4_augment_t(batch, d_train["xs"][idx],
                                            batch["u_seq"], rng, device)
            gather_idx = inv[codes]
            tgt = teacher_tr[idx].to(device).gather(1, gather_idx)
            st = stats["train"][idx].to(device)
            batch["stats"] = st.reshape(len(idx), n_ch, N_CELLS).gather(
                2, gather_idx[:, None, :].expand(-1, n_ch, -1)
            ).reshape(len(idx), n_ch, N_GRID, N_GRID)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            logp = torch.log_softmax(logits.float(), -1)
            loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError("non-finite loss/grad on first epoch")
            opt.step()
        sched.step()
        # val teacher-CE (matched to M1/suffstats convention)
        tot = 0.0
        with torch.no_grad():
            for lo in range(0, len(d_val["ids"]), 512):
                idx = np.arange(lo, min(lo + 512, len(d_val["ids"])))
                b = to_batch_t(d_val, idx, device)
                b["stats"] = stats["val"][idx].to(device).view(
                    len(idx), n_ch, N_GRID, N_GRID)
                logp = torch.log_softmax(model(b).float(), -1)
                tot += float(-(teacher_val[idx].to(device) * logp).sum())
        crit = tot / len(d_val["ids"])
        if crit < best:
            best = crit
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep {ep} crit {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch passed the selection criterion")
    model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["gen", "oracles", "train"])
    ap.add_argument("--maps", choices=["det", "ens"], default="ens")
    ap.add_argument("--teacher", choices=["det", "marg"], default="marg")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    root = cfg["seeds"]["root_entropy"]

    if args.stage == "gen":
        stage_gen(cfg, cfg_q, device)
        return

    data = {}
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        u_obs = np.load(dd / f"ch4tu_{name}_uobs.npy")
        data[name] = with_obs_wind(d, u_obs)

    if args.stage == "oracles":
        s_det = audit(np.load(dd / "ch4tu_calib_oracle_det.npz")["probs"],
                      np.load(dd / "ch4tu_test_oracle_det.npz")["probs"],
                      data, cfg, "oracle_det", rr)
        audit(np.load(dd / "ch4tu_calib_oracle_marg.npz")["probs"],
              np.load(dd / "ch4tu_test_oracle_marg.npz")["probs"],
              data, cfg, "oracle_marg", rr, ref_sizes=s_det)
        return

    stats = {n: torch.tensor(
        np.load(dd / f"ch4tu_{n}_maps_{args.maps}.npz")["maps"])
        for n in SPLITS}
    P_tr = torch.tensor(np.load(
        dd / f"ch4tu_train_oracle_{args.teacher}.npz")["probs"].astype(
        np.float32))
    P_val = torch.tensor(np.load(
        dd / f"ch4tu_val_oracle_{args.teacher}.npz")["probs"].astype(
        np.float32))
    teacher_tr = blur_teacher(P_tr, N_GRID, 0.75, device=device)
    teacher_val = blur_teacher(P_val, N_GRID, 0.75, device=device)
    model = train_u(cfg, data, stats, teacher_tr, teacher_val, device,
                    args.seed)
    Pc = probs_of(model, data["calib"], device, stats["calib"])
    Pt = probs_of(model, data["test"], device, stats["test"])
    # paired reference: the det/det student, with loud staleness checks
    tag = f"s_{args.maps}_{args.teacher}" + (
        "" if args.seed == 1 else f"_seed{args.seed}")
    ref = None
    if tag != "s_det_det":
        ref_path = dd / "ch4tu_audit_s_det_det.npz"
        if not ref_path.exists():
            raise RuntimeError("paired reference s_det_det missing -- "
                               "run --maps det --teacher det first")
        z = np.load(ref_path)
        if json.loads(str(z["fp"])) != fp_inputs(dd, "det", "det"):
            raise RuntimeError("stale s_det_det reference -- artifacts were "
                               "regenerated; rerun det/det first")
        ref = z["sizes"]
    audit(Pc, Pt, data, cfg, tag, rr, ref_sizes=ref,
          npz_path=dd / f"ch4tu_audit_{tag}.npz",
          fp=fp_inputs(dd, args.maps, args.teacher))


if __name__ == "__main__":
    main()
