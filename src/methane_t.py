"""CH4-T: time-varying-wind Gaussian plume testbed.

Same Pasquill-Gifford plume physics as methane_model, but each scenario has a
KNOWN fluctuating wind sequence u(t_1..t_T) (AR(1) meander in direction and
speed) instead of a constant wind.  The response is evaluated per timestep, so
it stays linear in Q and closed-form -> the exact-posterior machinery transfers
with the response folded to (cell, sensor, time).  Restores time-of-arrival /
bearing information (as the wind veers, the plume sweeps across masts), which
should sharply reduce the un-localizable fraction that the steady model had.

The localizer sees the instantaneous wind per reading token (token 25 -> 27).

Usage:
  python src/methane_t.py --probe 400      # identifiability check only
  python src/methane_t.py                   # full data -> posteriors -> M0/M1 -> audit
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import (cell_centers, get_device, load_config, pos_to_cell,
                    resolve, update_json)
from conformal import region_mask, regions, tail_scores, tail_threshold
from exact_posterior import marginal_log_evidence
from methane_model import U_SCALE, plume_ppm_per_kgh
from methane_pipeline import ch4_cfg
from model import count_params, smoothed_targets
from train import blur_teacher, d4_target_perms, teacher_ce, val_nll

N_GRID = 64
T = 30
SPLITS = {"train": 10000, "val": 1000, "calib": 2000, "test": 2000}
TAG_BASE = 50


# ---------------------------------------------------------------- wind
def wind_sequence(rng):
    """Known fluctuating wind: mean (U0, theta0) + AR(1) meander. Returns
    (T,2) normalized by U_SCALE, plus the mean vector (normalized)."""
    U0 = np.exp(rng.uniform(np.log(1.0), np.log(8.0)))
    th0 = rng.uniform(0, 2 * np.pi)
    rho = 0.7
    amp_dir = np.deg2rad(rng.uniform(20.0, 40.0))
    amp_spd = rng.uniform(0.10, 0.30)
    ed = es = 0.0
    seq = np.zeros((T, 2))
    for k in range(T):
        ed = rho * ed + np.sqrt(1 - rho ** 2) * rng.normal()
        es = rho * es + np.sqrt(1 - rho ** 2) * rng.normal()
        th = th0 + amp_dir * ed
        U = U0 * np.exp(amp_spd * es)
        seq[k] = [U * np.cos(th), U * np.sin(th)]
    mean_u = seq.mean(0) / U_SCALE
    return seq / U_SCALE, mean_u


# ---------------------------------------------------------------- physics
def sensor_responses_t(sensors, xs, u_seq, stab, device):
    """(S,T) unit-rate ppm response with per-timestep wind."""
    S = sensors.shape[0]
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)[:, None, :].expand(S, T, 2)
    xx = torch.as_tensor(xs, dtype=torch.float64, device=device)[None, None, :].expand(S, T, 2)
    uu = torch.as_tensor(u_seq, dtype=torch.float64, device=device)[None, :, :].expand(S, T, 2)
    st = torch.full((S, T), int(stab), dtype=torch.long, device=device)
    return plume_ppm_per_kgh(ss, xx, uu, st)


def cell_responses_t(cells, sensors, u_seq, stab, device, chunk=512):
    """(C,S,T) unit-rate response for all candidate cells."""
    C, S = cells.shape[0], sensors.shape[0]
    out = torch.empty((C, S, T), device=device, dtype=torch.float64)
    uu = torch.as_tensor(u_seq, dtype=torch.float64, device=device)
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        c = hi - lo
        cc = cells[lo:hi][:, None, None, :].expand(c, S, T, 2)
        s2 = ss[None, :, None, :].expand(c, S, T, 2)
        u2 = uu[None, None, :, :].expand(c, S, T, 2)
        st = torch.full((c, S, T), int(stab), dtype=torch.long, device=device)
        out[lo:hi] = plume_ppm_per_kgh(s2, cc, u2, st)  # (sensor, source): source=cells
    return out


# ---------------------------------------------------------------- data
def gen_split(name, count, cfg, root):
    tag = TAG_BASE + list(SPLITS).index(name)
    rows = {k: [] for k in ["ids", "xs", "q", "u_seq", "u_mean", "D", "sigma",
                            "n_sensors", "sensors", "readings", "keep"]}
    times = np.arange(1, T + 1) / T
    for i in range(count):
        rng = np.random.default_rng([root, tag, i])
        xs = rng.uniform(0.1, 0.9, 2)
        # cell-level attribution: sources lie on the 64x64 grid.  With
        # time-varying wind the posterior is often sharper than a cell, so a
        # sub-cell source offset would leave the true cell nearly empty and
        # force conformal to inflate every region; snapping removes that
        # discretization tax and keeps the exact-posterior audit meaningful.
        cell = pos_to_cell(xs[None], N_GRID)[0]
        xs = np.array([(cell % N_GRID + 0.5) / N_GRID,
                       (cell // N_GRID + 0.5) / N_GRID])
        q = np.exp(rng.uniform(np.log(10.0), np.log(500.0)))
        u_seq, u_mean = wind_sequence(rng)
        stab = int(rng.integers(0, 6))
        sig = np.exp(rng.uniform(np.log(0.2), np.log(3.0)))
        pd = rng.uniform(0.0, 0.2)
        n = int(rng.integers(4, 13))
        sensors = rng.uniform(0, 1, (n, 2))
        pad = np.zeros((12, 2)); pad[:n] = sensors
        g = sensor_responses_t(sensors, xs, u_seq, stab, "cpu").numpy()  # (n,T)
        readings = np.zeros((12, T), np.float32)
        keep = np.zeros((12, T), bool)
        readings[:n] = (q * g + rng.normal(0, sig, (n, T))).astype(np.float32)
        keep[:n] = rng.uniform(size=(n, T)) >= pd
        if not keep.any():
            keep[0, 0] = True
        rows["ids"].append(f"ch4t-{name}-{i:06d}")
        rows["xs"].append(xs); rows["q"].append(q)
        rows["u_seq"].append(u_seq); rows["u_mean"].append(u_mean)
        rows["D"].append(np.exp(stab / 5.0)); rows["sigma"].append(sig)
        rows["n_sensors"].append(n); rows["sensors"].append(pad)
        rows["readings"].append(readings); rows["keep"].append(keep)
    d = {k: np.array(v) for k, v in rows.items()}
    d["times"] = times
    d["true_cell"] = pos_to_cell(d["xs"], N_GRID)
    return d


def stab_of(d, i):
    return int(round(np.log(d["D"][i]) * 5.0))


# ---------------------------------------------------------------- oracle
@torch.no_grad()
def posteriors(d, cfg_q, device, verbose_every=2000):
    cells = torch.tensor(cell_centers(N_GRID), device=device, dtype=torch.float64)
    S = len(d["ids"])
    probs = np.empty((S, N_GRID * N_GRID), np.float64)
    norm_err = 0.0
    for i in range(S):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g = cell_responses_t(cells, d["sensors"][i, :ns], d["u_seq"][i],
                             stab_of(d, i), device)          # (C,ns,T)
        keep_t = torch.tensor(keep, device=device)
        g = g[:, keep_t]                                     # (C, M)
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        M = y.shape[0]
        a = g @ y
        b = (g * g).sum(1)
        yy = float(torch.dot(y, y))
        lp = marginal_log_evidence(a, b, yy, float(d["sigma"][i]), M,
                                   cfg_q, device)
        lp = lp - torch.logsumexp(lp, 0)
        p = torch.exp(lp)
        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
        probs[i] = p.cpu().numpy()
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  posterior {i+1}/{S}", flush=True)
    return probs, norm_err


def hpd_area(p, mass=0.90):
    s = np.sort(p)[::-1]
    return (1 + (np.cumsum(s) < mass).sum()) / p.size


# ---------------------------------------------------------------- model
class DeepSetsT(nn.Module):
    """DeepSets with instantaneous wind appended to each reading token."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.freqs = torch.tensor([float(f) for f in m["fourier_freqs"]])
        td = m["token_dim"] + 2                       # + (u_x, u_y) at t_k
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
        u_seq = b["u_seq"]                                # (B,T,2)
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


def to_batch_t(d, idx, device):
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return {"sensors": t(d["sensors"][idx], torch.float32),
            "readings": t(d["readings"][idx], torch.float32),
            "keep": t(d["keep"][idx], torch.bool),
            "times": t(d["times"], torch.float32),
            "u_seq": t(d["u_seq"][idx], torch.float32),
            "u_mean": t(d["u_mean"][idx], torch.float32),
            "D": t(d["D"][idx], torch.float32),
            "sigma": t(d["sigma"][idx], torch.float32),
            "n_sensors": t(d["n_sensors"][idx], torch.float32)}


def d4_augment_t(batch, xs, u_seq, rng, device):
    """D4 symmetry on sensors, source, and the whole wind sequence jointly."""
    B = batch["sensors"].shape[0]
    k = torch.tensor(rng.integers(0, 4, B), device=device)
    r = torch.tensor(rng.integers(0, 2, B), device=device)
    c = 0.5

    def rot(px, py, kk):
        ox = torch.where(kk == 0, px, torch.where(kk == 1, -py,
                         torch.where(kk == 2, -px, py)))
        oy = torch.where(kk == 0, py, torch.where(kk == 1, px,
                         torch.where(kk == 2, -py, -px)))
        return ox, oy

    sx = batch["sensors"][..., 0] - c
    sy = batch["sensors"][..., 1] - c
    sy = torch.where(r[:, None] == 1, -sy, sy)
    sx, sy = rot(sx, sy, k[:, None])
    out = dict(batch)
    out["sensors"] = torch.stack([sx + c, sy + c], dim=-1)
    # wind sequence (B,T,2): reflect y then rotate
    ux, uy = u_seq[..., 0], u_seq[..., 1]
    uy = torch.where(r[:, None] == 1, -uy, uy)
    ux, uy = rot(ux, uy, k[:, None])
    out["u_seq"] = torch.stack([ux, uy], dim=-1)
    out["u_mean"] = out["u_seq"].mean(1)
    xsx = torch.tensor(xs[:, 0], device=device) - c
    xsy = torch.tensor(xs[:, 1], device=device) - c
    xsy = torch.where(r == 1, -xsy, xsy)
    xsx, xsy = rot(xsx, xsy, k)
    xs_new = torch.stack([xsx + c, xsy + c], dim=-1).double().cpu().numpy()
    return out, xs_new, (r * 4 + k)


def train(cfg, d_train, d_val, device, distill, P_teacher=None,
          P_val_teacher=None, seed=1, model_cls=DeepSetsT):
    tr = cfg["training"]
    torch.manual_seed(6000 + seed + (100 if distill else 0))
    rng = np.random.default_rng(seed)
    model = model_cls(cfg).to(device)
    if seed == 1 and not distill:
        print(f"  {model_cls.__name__} params: "
              f"{count_params(model)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200,
                                                       eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device) if distill else None
    m = cfg["model"]
    n_train = len(d_train["ids"])
    best, best_state, since = np.inf, None, 0
    for ep in range(200):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            batch = to_batch_t(d_train, idx, device)
            batch, xs, codes = d4_augment_t(batch, d_train["xs"][idx],
                                            batch["u_seq"], rng, device)
            if distill:
                tgt = P_teacher[idx].to(device).gather(1, inv[codes])
            else:
                tc = torch.tensor(pos_to_cell(xs, N_GRID))
                tgt = smoothed_targets(tc, N_GRID, m["target_smooth_std_cells"],
                                       m["target_trunc_sigmas"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            logp = torch.log_softmax(logits.float(), -1)
            loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
        sched.step()
        crit = _val_crit(model, d_val, device, distill, P_val_teacher)
        if crit < best:
            best, since = crit, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 20 == 0:
            print(f"  {'M1' if distill else 'M0'} ep {ep} crit {crit:.4f}",
                  flush=True)
        if since >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _val_crit(model, d, device, distill, P_val_teacher):
    model.eval()
    n = len(d["ids"])
    tot = 0.0
    with torch.no_grad():
        for lo in range(0, n, 512):
            idx = np.arange(lo, min(lo + 512, n))
            logp = torch.log_softmax(model(to_batch_t(d, idx, device)).float(), -1)
            if distill:
                tot += float(-(P_val_teacher[idx].to(device) * logp).sum(1).sum())
            else:
                tc = torch.tensor(d["true_cell"][idx], device=device)
                tot += float(-logp.gather(1, tc[:, None]).sum())
    return tot / n


@torch.no_grad()
def model_probs(model, d, device):
    out = []
    for lo in range(0, len(d["ids"]), 512):
        idx = np.arange(lo, min(lo + 512, len(d["ids"])))
        out.append(torch.softmax(model(to_batch_t(d, idx, device)).float(),
                                 -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    root = cfg["seeds"]["root_entropy"]
    alpha = cfg["conformal"]["alpha"]

    if args.probe:
        print(f"== PROBE: {args.probe} test scenarios ==", flush=True)
        d = gen_split("test", args.probe, cfg, root)
        P, ne = posteriors(d, cfg_q, device, verbose_every=100)
        areas = np.array([hpd_area(P[i]) for i in range(len(P))])
        weak = float((areas >= 0.5).mean())
        ident = float((areas < 0.10).mean())
        # physics sanity: for informative scenarios the MAP cell should be at
        # or adjacent to the true cell (a swapped/mis-specified oracle fails this)
        tc = d["true_cell"]
        mapc = P.argmax(1)
        cheb = np.maximum(np.abs(mapc % 64 - tc % 64), np.abs(mapc // 64 - tc // 64))
        sharp = areas < 0.02
        print(f"  norm_err {ne:.2e}", flush=True)
        print(f"  MAP within 1 cell of true (sharp scenarios): "
              f"{(cheb[sharp] <= 1).mean():.2f} of {int(sharp.sum())}", flush=True)
        print(f"  weakly identifiable (oracle>=50% site): {weak:.2f} "
              f"(steady model was 0.57)", flush=True)
        print(f"  identifiable (oracle<10% site): {ident:.2f}", flush=True)
        print(f"  oracle area quartiles: "
              f"{np.quantile(areas, [0.25, 0.5, 0.75]).round(3)}", flush=True)
        update_json(results_dir / "gates.json",
                    {"CH4T_probe": {"weak_frac": weak, "ident_frac": ident,
                                    "n": args.probe}})
        return

    print("== data ==", flush=True)
    data = {}
    for name, count in SPLITS.items():
        p = data_dir / f"ch4t_{name}.npz"
        if p.exists():
            data[name] = dict(np.load(p, allow_pickle=True))
        else:
            data[name] = gen_split(name, count, cfg, root)
            np.savez_compressed(p, **data[name])
        print(f"  {name}: {len(data[name]['ids'])}", flush=True)

    print("== exact posteriors ==", flush=True)
    P_ex = {}
    for name in SPLITS:
        p = data_dir / f"ch4t_{name}_posterior.npz"
        if p.exists():
            P_ex[name] = np.load(p)["probs"]
        else:
            P_ex[name], ne = posteriors(data[name], cfg_q, device)
            np.savez_compressed(p, ids=data[name]["ids"], probs=P_ex[name])
            print(f"  {name}: norm_err {ne:.2e}", flush=True)

    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    e_area = reg_e["sizes"] / 4096
    cov_e = float(reg_e["covered"].mean())
    print(f"  conformal-on-exact coverage: {cov_e:.3f} (must be ~0.90)",
          flush=True)
    print(f"  test weakly-identifiable frac: {(e_area >= 0.5).mean():.2f}",
          flush=True)
    # GUARD: the exact posterior must self-calibrate; if not, the oracle
    # forward model is misspecified (e.g. a source/sensor swap) and every
    # downstream number is meaningless.
    assert 0.85 <= cov_e <= 0.97, (
        f"conformal-on-exact coverage {cov_e:.3f} outside [0.85, 0.97] -- "
        "oracle forward model is broken, aborting.")  # upper 0.97 allows the
    # benign over-coverage near-atomic (>~2-cell) posteriors produce on a grid

    print("== train M0/M1 ==", flush=True)
    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    models = {"M0": train(cfg, data["train"], data["val"], device, False),
              "M1": train(cfg, data["train"], data["val"], device, True, Pt, Pv)}

    print("== audit ==", flush=True)
    res = {"exact_coverage": float(reg_e["covered"].mean()),
           "weak_frac": float((e_area >= 0.5).mean())}
    from scipy.stats import wilcoxon
    ratios = {}
    for name, model in models.items():
        P_c = model_probs(model, data["calib"], device)
        P_t = model_probs(model, data["test"], device)
        r2 = np.random.default_rng(cfg["conformal"]["score_seed"])
        th = tail_threshold(tail_scores(P_c, data["calib"]["true_cell"], r2), alpha)
        reg = regions(P_t, th, data["test"]["true_cell"])
        ratio = reg["sizes"] / np.maximum(reg_e["sizes"], 1)
        ratios[name] = ratio
        ident = e_area < 0.10
        res[name] = {
            "coverage": float(reg["covered"].mean()),
            "ineff_median_all": float(np.median(ratio)),
            "ineff_median_identifiable": float(np.median(ratio[ident])),
        }
        np.savez_compressed(data_dir / f"ch4t_audit_{name}.npz",
                            sizes=reg["sizes"], exact_sizes=reg_e["sizes"])
        print(f"  {name}: {res[name]}", flush=True)
    w = wilcoxon(np.log(ratios["M0"]), np.log(ratios["M1"]),
                 alternative="greater")
    res["wilcoxon_p_all"] = float(w.pvalue)
    with open(results_dir / "ch4t_results.json", "w") as f:
        json.dump(res, f, indent=2)
    update_json(results_dir / "gates.json", {"CH4T": res})
    print("CH4T DONE", flush=True)


if __name__ == "__main__":
    main()
