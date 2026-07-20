"""CH4-500 end-to-end pipeline: gates -> data -> posteriors -> M0/M1 -> audit.

A methane-configured second testbed (see methane_model.py) run through the
IDENTICAL machinery: exact posteriors as audit reference and distillation
teacher, D4 augmentation, tail-space conformal, matched-coverage inefficiency.

Outputs: data/ch4_*.npz, results/ch4_results.json, figures/fig15_methane.pdf.
Usage:  python src/methane_pipeline.py
"""
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from scipy.stats import wilcoxon

from common import (cell_centers, get_device, load_config, pos_to_cell,
                    resolve, scenario_rng, update_json)
from conformal import regions, tail_scores, tail_threshold
from exact_posterior import marginal_log_evidence
from inference import to_batch
from methane_model import (L_SITE, U_SCALE, ch4_cell_responses,
                           plume_ppm_per_kgh)
from model import DeepSetsLocalizer, smoothed_targets
from train import blur_teacher, d4_augment, d4_target_perms, teacher_ce, val_nll

SPLITS = {"train": 10000, "val": 1000, "calib": 2000, "test": 2000}
TAG_BASE = 40                     # SeedSequence split tags 40..43
Q_LO, Q_HI = 10.0, 500.0          # kg/h
SIG_LO, SIG_HI = 0.2, 3.0         # ppm
N_GRID = 64
T = 30

C_BASE, C_IMPR, C_EXACT = "#2a78d6", "#e87ba4", "#008300"
C_MUT, C_GRID_ = "#52514e", "#e3e2df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": C_MUT, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight"})


def ch4_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["prior"]["q_log_low"] = Q_LO
    c["prior"]["q_log_high"] = Q_HI
    return c


# ------------------------------------------------------------------ gates
def gates(device):
    out = {}
    t = lambda a: torch.tensor(a, device=device, dtype=torch.float64)
    # crosswind mirror symmetry
    rng = np.random.default_rng(0)
    xs = t(rng.uniform(0.2, 0.8, (200, 2)))
    u = t(np.stack([np.full(200, 0.5), np.zeros(200)], -1))   # wind +x
    dy = rng.uniform(0.01, 0.4, 200)
    dx = rng.uniform(0.05, 0.5, 200)
    sA = xs + t(np.stack([dx, dy], -1))
    sB = xs + t(np.stack([dx, -dy], -1))
    stab = torch.tensor(rng.integers(0, 6, 200), device=device)
    cA = plume_ppm_per_kgh(sA, xs, u, stab)
    cB = plume_ppm_per_kgh(sB, xs, u, stab)
    out["symmetry_max_rel"] = float(((cA - cB).abs() /
                                     cA.abs().clamp(min=1e-30)).max())
    # mass-flux conservation: int int U*C dz dy = Q at several x
    devs = []
    for stab_i in range(6):
        for x in [50.0, 150.0, 400.0]:
            y = torch.linspace(-400, 400, 4001, device=device,
                               dtype=torch.float64)
            z = torch.linspace(0, 400, 2001, device=device,
                               dtype=torch.float64)
            from methane_model import _sigmas, SIGMA_Y0, SIGMA_Z0, H_SRC
            sy, sz = _sigmas(t([x]), torch.tensor([stab_i], device=device))
            sy = torch.sqrt(sy ** 2 + SIGMA_Y0 ** 2)
            sz = torch.sqrt(sz ** 2 + SIGMA_Z0 ** 2)
            U = 3.0
            q_kgs = 1.0 / 3600.0
            c = (q_kgs / (2 * np.pi * U * sy * sz)
                 * torch.exp(-0.5 * (y[:, None] / sy) ** 2)
                 * (torch.exp(-0.5 * ((z[None, :] - H_SRC) / sz) ** 2)
                    + torch.exp(-0.5 * ((z[None, :] + H_SRC) / sz) ** 2)))
            flux = float(torch.trapz(torch.trapz(c, z, dim=1), y) * U)
            devs.append(abs(flux / q_kgs - 1.0))
    out["mass_flux_max_rel_dev"] = float(max(devs))
    # magnitude sanity: 100 kg/h, U=3 m/s, D stability, 100 m downwind
    c100 = plume_ppm_per_kgh(t([[0.7, 0.5]]), t([[0.5, 0.5]]),
                             t([[3.0 / U_SCALE, 0.0]]),
                             torch.tensor([3], device=device))
    out["ppm_100kgh_100m_D"] = float(c100[0] * 100.0)
    out["gates_pass"] = bool(out["symmetry_max_rel"] < 1e-10
                             and out["mass_flux_max_rel_dev"] < 0.01
                             and 5.0 < out["ppm_100kgh_100m_D"] < 500.0)
    return out


# ------------------------------------------------------------------ data
def gen_split(name, count, cfg, root):
    tag = TAG_BASE + list(SPLITS).index(name)
    times = np.arange(1, T + 1) / T
    rows = {k: [] for k in ["ids", "xs", "q", "u", "D", "sigma", "p_drop",
                            "n_sensors", "sensors", "readings", "keep"]}
    for i in range(count):
        rng = scenario_rng(root, tag, i)
        xs = rng.uniform(0.1, 0.9, 2)
        q = np.exp(rng.uniform(np.log(Q_LO), np.log(Q_HI)))
        U = np.exp(rng.uniform(np.log(1.0), np.log(8.0)))
        th = rng.uniform(0, 2 * np.pi)
        u_norm = np.array([U * np.cos(th), U * np.sin(th)]) / U_SCALE
        stab = int(rng.integers(0, 6))
        sig = np.exp(rng.uniform(np.log(SIG_LO), np.log(SIG_HI)))
        pd = rng.uniform(0.0, 0.2)
        n = int(rng.integers(4, 13))
        sensors = rng.uniform(0, 1, (n, 2))
        pad = np.full((12, 2), 0.0)
        pad[:n] = sensors
        g = plume_ppm_per_kgh(torch.tensor(sensors), torch.tensor(xs)[None],
                              torch.tensor(u_norm)[None],
                              torch.tensor([stab])).numpy()      # (n,)
        mean = q * g
        readings = np.zeros((12, T), np.float32)
        keep = np.zeros((12, T), bool)
        readings[:n] = (mean[:, None]
                        + rng.normal(0, sig, (n, T))).astype(np.float32)
        keep[:n] = rng.uniform(size=(n, T)) >= pd
        if not keep.any():
            keep[0, 0] = True
        rows["ids"].append(f"ch4-{name}-{i:06d}")
        rows["xs"].append(xs); rows["q"].append(q)
        rows["u"].append(u_norm); rows["D"].append(np.exp(stab / 5.0))
        rows["sigma"].append(sig); rows["p_drop"].append(pd)
        rows["n_sensors"].append(n); rows["sensors"].append(pad)
        rows["readings"].append(readings); rows["keep"].append(keep)
    d = {k: np.array(v) for k, v in rows.items()}
    d["times"] = times
    d["true_cell"] = pos_to_cell(d["xs"], N_GRID)
    return d


def stab_of(d, i):
    return int(round(np.log(d["D"][i]) * 5.0))


# ------------------------------------------------------------------ oracle
@torch.no_grad()
def ch4_posteriors(d, cfg_q, device, verbose_every=2000):
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    S = len(d["ids"])
    probs = np.empty((S, N_GRID * N_GRID), np.float64)
    norm_err = 0.0
    for i in range(S):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]                              # (ns, T)
        sensors = torch.tensor(d["sensors"][i, :ns], device=device,
                               dtype=torch.float64)
        u = torch.tensor(d["u"][i], device=device, dtype=torch.float64)
        g = ch4_cell_responses(cells, sensors, u, stab_of(d, i), device)
        cnt = torch.tensor(keep.sum(1), device=device, dtype=torch.float64)
        ysum = torch.tensor((d["readings"][i, :ns] * keep).sum(1),
                            device=device, dtype=torch.float64)
        yy = float((torch.tensor(d["readings"][i, :ns][keep],
                                 device=device, dtype=torch.float64) ** 2).sum())
        a = g @ ysum                                          # (C,)
        b = (g ** 2) @ cnt                                    # (C,)
        M = int(keep.sum())
        lp = marginal_log_evidence(a, b, yy, float(d["sigma"][i]), M,
                                   cfg_q, device)
        lp = lp - torch.logsumexp(lp, 0)
        p = torch.exp(lp)
        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
        probs[i] = p.cpu().numpy()
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  posterior {i+1}/{S}", flush=True)
    return probs, norm_err


# ------------------------------------------------------------------ train
def train_ch4(cfg, d_train, d_val, device, distill, P_teacher=None,
              P_val_teacher=None, seed=1):
    tr = cfg["training"]
    torch.manual_seed(3000 + seed + (100 if distill else 0))
    rng = np.random.default_rng(seed)
    model = DeepSetsLocalizer(cfg).to(device)
    max_ep = 200
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep,
                                                       eta_min=tr["lr_final"])
    inv_perms = d4_target_perms(N_GRID, device) if distill else None
    m = cfg["model"]
    n_train = len(d_train["ids"])
    bs = tr["batch_size"]
    best, best_state, since = np.inf, None, 0
    for ep in range(max_ep):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, bs):
            idx = perm[lo: lo + bs]
            batch = to_batch(d_train, idx, device)
            xs = d_train["xs"][idx]
            batch, xs, codes = d4_augment(batch, xs, rng, device)
            if distill:
                tgt = P_teacher[idx].to(device).gather(1, inv_perms[codes])
            else:
                tc = torch.tensor(pos_to_cell(xs, N_GRID))
                tgt = smoothed_targets(tc, N_GRID,
                                       m["target_smooth_std_cells"],
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
        crit = (teacher_ce(model, d_val, P_val_teacher, device) if distill
                else val_nll(model, d_val, device))
        if crit < best:
            best, since = crit, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 20 == 0:
            print(f"  {'M1' if distill else 'M0'} ep {ep} crit {crit:.4f} "
                  f"best {best:.4f}", flush=True)
        if since >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def model_probs(model, d, device):
    S = len(d["ids"])
    out = []
    for lo in range(0, S, 512):
        idx = np.arange(lo, min(lo + 512, S))
        out.append(torch.softmax(model(to_batch(d, idx, device)).float(),
                                 -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    fig_dir = resolve(cfg, "figures_dir")
    root = cfg["seeds"]["root_entropy"]
    alpha = cfg["conformal"]["alpha"]

    print("== gates ==", flush=True)
    g = gates(device)
    print(g, flush=True)
    assert g["gates_pass"], g

    print("== data ==", flush=True)
    data = {}
    for name, count in SPLITS.items():
        p = data_dir / f"ch4_{name}.npz"
        if p.exists():
            data[name] = dict(np.load(p, allow_pickle=True))
        else:
            data[name] = gen_split(name, count, cfg, root)
            np.savez_compressed(p, **data[name])
        print(f"  {name}: {len(data[name]['ids'])}", flush=True)
    ids_all = np.concatenate([data[n]["ids"] for n in SPLITS])
    assert len(set(ids_all.tolist())) == len(ids_all)

    print("== exact posteriors ==", flush=True)
    P_ex = {}
    for name in SPLITS:
        p = data_dir / f"ch4_{name}_posterior.npz"
        if p.exists():
            P_ex[name] = np.load(p)["probs"]
        else:
            P_ex[name], ne = ch4_posteriors(data[name], cfg_q, device)
            np.savez_compressed(p, ids=data[name]["ids"], probs=P_ex[name])
            print(f"  {name}: norm_err {ne:.2e}", flush=True)

    # oracle sanity: mode recovery with tiny noise, cell-centered sources
    print("== oracle sanity ==", flush=True)
    hits = 0
    rng = np.random.default_rng(9)
    for _ in range(24):
        i = int(rng.integers(0, len(data["val"]["ids"])))
        d1 = {k: (v[i:i + 1] if k != "times" else v)
              for k, v in data["val"].items()}
        cell = d1["true_cell"][0]
        d1["xs"] = np.array([[(cell % N_GRID + 0.5) / N_GRID,
                              (cell // N_GRID + 0.5) / N_GRID]])
        ns = int(d1["n_sensors"][0])
        gg = plume_ppm_per_kgh(torch.tensor(d1["sensors"][0, :ns]),
                               torch.tensor(d1["xs"][0])[None],
                               torch.tensor(d1["u"][0])[None],
                               torch.tensor([stab_of(d1, 0)])).numpy()
        if (d1["q"][0] * gg).max() < 1.0:      # identifiability floor: 1 ppm
            continue
        d1["sigma"] = np.array([1e-4])
        d1["readings"] = np.zeros_like(d1["readings"])
        d1["readings"][0, :ns] = (d1["q"][0] * gg[:, None]
                                  + rng.normal(0, 1e-4, (ns, T))).astype(np.float32)
        pp, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
        hits += int(pp[0].argmax() == cell)
    print(f"  mode recovery (identifiable, cell-centered): {hits}/~24 tried",
          flush=True)

    print("== conformal on exact ==", flush=True)
    rngc = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rngc), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    cov_e = float(reg_e["covered"].mean())
    print(f"  exact coverage {cov_e:.4f}", flush=True)

    print("== train M0 / M1 ==", flush=True)
    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    models = {"M0": train_ch4(cfg, data["train"], data["val"], device, False),
              "M1": train_ch4(cfg, data["train"], data["val"], device, True,
                              Pt, Pv)}

    print("== audit ==", flush=True)
    res = {"gates": g, "exact_coverage": cov_e,
           "exact_area_median": float(np.median(reg_e["sizes"] / 4096))}
    ratios = {}
    for name, model in models.items():
        P_c = model_probs(model, data["calib"], device)
        P_t = model_probs(model, data["test"], device)
        rngm = np.random.default_rng(cfg["conformal"]["score_seed"])
        th = tail_threshold(tail_scores(P_c, data["calib"]["true_cell"], rngm),
                            alpha)
        reg = regions(P_t, th, data["test"]["true_cell"])
        ratio = reg["sizes"] / np.maximum(reg_e["sizes"], 1)
        ratios[name] = ratio
        res[name] = {
            "coverage": float(reg["covered"].mean()),
            "area_median": float(np.median(reg["sizes"] / 4096)),
            "ineff_median": float(np.median(ratio)),
            "ineff_mean": float(ratio.mean()),
        }
        np.savez_compressed(data_dir / f"ch4_audit_{name}.npz",
                            sizes=reg["sizes"], exact_sizes=reg_e["sizes"])
        print(f"  {name}: {res[name]}", flush=True)
    w = wilcoxon(np.log(ratios["M0"]), np.log(ratios["M1"]),
                 alternative="greater")
    res["wilcoxon_p"] = float(w.pvalue)
    with open(results_dir / "ch4_results.json", "w") as f:
        json.dump(res, f, indent=2)
    update_json(results_dir / "gates.json", {"CH4": res})

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axes[0]
    lim = [2e-4, 1.2]
    e_area = reg_e["sizes"] / 4096
    ax.plot(lim, lim, "--", color=C_MUT, lw=0.8)
    for name, col in [("M0", C_BASE), ("M1", C_IMPR)]:
        s = np.load(data_dir / f"ch4_audit_{name}.npz")["sizes"] / 4096
        ax.scatter(e_area, s, s=4, alpha=0.2, c=col, edgecolors="none",
                   label=f"{name} (median "
                         f"{res[name]['ineff_median']:.1f}$\\times$)",
                   rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("exact-posterior region area")
    ax.set_ylabel("learned region area")
    ax.set_title("CH4-500: sharpness at matched 90% coverage")
    ax.legend(frameon=False, loc="lower right", markerscale=2.5)
    ax.grid(color=C_GRID_, lw=0.5)

    ax = axes[1]
    i = int(np.argsort(np.abs(e_area - np.median(e_area)))[3])
    d = data["test"]
    ng = 96
    cc = (np.arange(ng) + 0.5) / ng
    X, Y = np.meshgrid(cc, cc)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1))
    field = plume_ppm_per_kgh(pts, torch.tensor(d["xs"][i])[None],
                              torch.tensor(d["u"][i])[None],
                              torch.tensor([stab_of(d, i)])).numpy() * d["q"][i]
    ax.imshow(field.reshape(ng, ng), origin="lower", extent=[0, 500, 0, 500],
              cmap="Purples", norm=PowerNorm(0.4))
    ns = int(d["n_sensors"][i])
    ax.scatter(d["sensors"][i, :ns, 0] * 500, d["sensors"][i, :ns, 1] * 500,
               marker="^", s=20, c="#0b0b0b", edgecolors="white", lw=0.4)
    ax.scatter(*(d["xs"][i] * 500), marker="*", s=100, c="white",
               edgecolors="#0b0b0b", lw=0.9)
    ax.set_title(f"CH4 plume: {d['q'][i]:.0f} kg/h, "
                 f"{np.linalg.norm(d['u'][i]) * U_SCALE:.1f} m/s, "
                 f"class {'ABCDEF'[stab_of(d, i)]}")
    ax.set_xlabel("m"); ax.set_ylabel("m")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig15_methane.pdf")
    print("fig15 done", flush=True)


if __name__ == "__main__":
    main()
