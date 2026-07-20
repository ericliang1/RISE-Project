"""CH4-500 completion campaign: seed stability, raw-teacher ablation,
sensor-count dose-response, and a 3-point data-scaling study.

Everything reuses methane_pipeline machinery; checkpoints saved as
data/checkpoints/ch4_{m0,m1}_s{seed}.pt.  Outputs results/ch4_extras.json,
data/ch4_dose_curves.npz, data/ch4_scaling.npz.

Usage:  python src/methane_extras.py
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from common import cell_centers, get_device, load_config, pos_to_cell, resolve, update_json
from conformal import region_mask, regions, tail_scores, tail_threshold
from inference import to_batch
from methane_model import plume_ppm_per_kgh
from methane_pipeline import (SPLITS, ch4_cfg, ch4_posteriors, gen_split,
                              model_probs, stab_of, train_ch4)
from model import DeepSetsLocalizer
from train import blur_teacher

N_GRID = 64
T = 30


def load_ch4(data_dir, name):
    return dict(np.load(data_dir / f"ch4_{name}.npz", allow_pickle=True))


def audit_model(model, data, P_ex_calib_thr, reg_e_sizes, cfg, device, alpha):
    P_c = model_probs(model, data["calib"], device)
    P_t = model_probs(model, data["test"], device)
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th = tail_threshold(tail_scores(P_c, data["calib"]["true_cell"], rng), alpha)
    reg = regions(P_t, th, data["test"]["true_cell"])
    e = reg_e_sizes / 4096
    ratio = reg["sizes"] / np.maximum(reg_e_sizes, 1)
    ident = e < 0.10
    return {
        "coverage": float(reg["covered"].mean()),
        "ineff_median_identifiable": float(np.median(ratio[ident])),
        "ineff_mean_identifiable": float(ratio[ident].mean()),
        "ineff_median_all": float(np.median(ratio)),
        "area_median": float(np.median(reg["sizes"] / 4096)),
    }, th, reg["sizes"]


def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    ck_dir = resolve(cfg, "checkpoints_dir")
    alpha = cfg["conformal"]["alpha"]
    root = cfg["seeds"]["root_entropy"]

    data = {n: load_ch4(data_dir, n) for n in SPLITS}
    P_ex = {n: np.load(data_dir / f"ch4_{n}_posterior.npz")["probs"]
            for n in SPLITS}
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)),
                      N_GRID, 0.75, device=device)
    Pt_raw = torch.tensor(P_ex["train"].astype(np.float32))
    Pv_raw = torch.tensor(P_ex["val"].astype(np.float32))

    out = {"seeds": [], "raw_teacher": None}
    thresholds = {}

    # ---------------- 1. seed stability -----------------------------------
    print("== seeds ==", flush=True)
    for seed in (1, 2, 3):
        for mode, distill in [("M0", False), ("M1", True)]:
            stem = f"ch4_{mode.lower()}_s{seed}"
            ck = ck_dir / f"{stem}.pt"
            if ck.exists():
                model = DeepSetsLocalizer(cfg).to(device)
                model.load_state_dict(torch.load(ck, weights_only=False))
                model.eval()
            else:
                model = train_ch4(cfg, data["train"], data["val"], device,
                                  distill, Pt if distill else None,
                                  Pv if distill else None, seed=seed)
                torch.save(model.state_dict(), ck)
            r, th, sizes = audit_model(model, data, None, reg_e["sizes"],
                                       cfg, device, alpha)
            r.update(mode=mode, seed=seed)
            out["seeds"].append(r)
            if seed == 1:
                thresholds[mode] = th
                np.savez_compressed(data_dir / f"ch4_audit_{mode}.npz",
                                    sizes=sizes, exact_sizes=reg_e["sizes"])
            print(f"  {mode} s{seed}: {r}", flush=True)

    # ---------------- 2. raw-teacher ablation ------------------------------
    print("== raw teacher ==", flush=True)
    model = train_ch4(cfg, data["train"], data["val"], device, True,
                      Pt_raw, Pv_raw, seed=1)
    r, _, _ = audit_model(model, data, None, reg_e["sizes"], cfg, device, alpha)
    out["raw_teacher"] = r
    print(f"  raw: {r}", flush=True)

    # ---------------- 3. dose-response knob 1 ------------------------------
    print("== dose-response (sensor count) ==", flush=True)
    m0 = DeepSetsLocalizer(cfg).to(device)
    m0.load_state_dict(torch.load(ck_dir / "ch4_m0_s1.pt", weights_only=False))
    m0.eval()
    m1 = DeepSetsLocalizer(cfg).to(device)
    m1.load_state_dict(torch.load(ck_dir / "ch4_m1_s1.pt", weights_only=False))
    m1.eval()
    Ns = list(range(4, 13))
    B = 50
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    curves = {"M0": np.zeros((B, len(Ns))), "M1": np.zeros((B, len(Ns))),
              "exact": np.zeros((B, len(Ns)))}
    for b in range(B):
        srng = np.random.default_rng(700000 + b)
        xs = srng.uniform(0.15, 0.85, 2)
        q = np.exp(srng.uniform(np.log(10.0), np.log(500.0)))
        U = np.exp(srng.uniform(np.log(1.0), np.log(8.0)))
        th_w = srng.uniform(0, 2 * np.pi)
        u_norm = np.array([U * np.cos(th_w), U * np.sin(th_w)]) / 8.0
        stab = int(srng.integers(0, 6))
        sig = np.exp(srng.uniform(np.log(0.2), np.log(3.0)))
        master = srng.uniform(0, 1, (12, 2))
        g_master = plume_ppm_per_kgh(
            torch.tensor(master), torch.tensor(xs)[None],
            torch.tensor(u_norm)[None], torch.tensor([stab])).numpy()
        noise = srng.normal(0, sig, (12, T))
        readings_master = (q * g_master[:, None] + noise).astype(np.float32)
        for j, N in enumerate(Ns):
            pad_s = np.zeros((12, 2)); pad_s[:N] = master[:N]
            pad_r = np.zeros((12, T), np.float32)
            pad_r[:N] = readings_master[:N]
            keep = np.zeros((12, T), bool); keep[:N] = True
            d1 = {"ids": np.array([f"d{b}n{N}"]), "sensors": pad_s[None],
                  "readings": pad_r[None], "keep": keep[None],
                  "times": np.arange(1, T + 1) / T, "u": u_norm[None],
                  "D": np.array([np.exp(stab / 5.0)]),
                  "sigma": np.array([sig]), "n_sensors": np.array([N]),
                  "xs": xs[None], "q": np.array([q]),
                  "true_cell": pos_to_cell(xs[None], N_GRID)}
            pe, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
            curves["exact"][b, j] = region_mask(pe[0], th_e).sum() / 4096
            for mode, model in [("M0", m0), ("M1", m1)]:
                P = model_probs(model, d1, device)[0]
                curves[mode][b, j] = region_mask(
                    P, thresholds[mode]).sum() / 4096
    np.savez_compressed(data_dir / "ch4_dose_curves.npz", Ns=np.array(Ns),
                        **curves)
    dose = {}
    for mode in ("M0", "M1"):
        sl = [np.polyfit(Ns, curves[mode][b], 1)[0] for b in range(B)]
        se = [np.polyfit(Ns, curves["exact"][b], 1)[0] for b in range(B)]
        rho = [spearmanr(curves[mode][b], curves["exact"][b]).statistic
               for b in range(B)]
        dose[mode] = {
            "slope_ratio_mean_curves": float(np.mean(sl) / np.mean(se)),
            "spearman_median": float(np.nanmedian(rho)),
        }
    out["dose_knob1"] = dose
    print(f"  dose: {dose}", flush=True)

    # ---------------- 4. scaling (2.5k / 10k / 40k) ------------------------
    print("== scaling ==", flush=True)
    p_xl = data_dir / "ch4_train_xl.npz"
    if p_xl.exists():
        d_xl = dict(np.load(p_xl, allow_pickle=True))
    else:
        d_xl = gen_split("train", 30000, cfg, root + 7)   # fresh stream
        d_xl["ids"] = np.array([f"ch4xl-{i:06d}" for i in range(30000)])
        np.savez_compressed(p_xl, **d_xl)
    p_xlp = data_dir / "ch4_train_xl_posterior.npz"
    if p_xlp.exists():
        P_xl = np.load(p_xlp)["probs"]
    else:
        P_xl, _ = ch4_posteriors(d_xl, cfg_q, device)
        np.savez_compressed(p_xlp, ids=d_xl["ids"], probs=P_xl)
    big = {}
    for k in data["train"]:
        big[k] = (data["train"][k] if k == "times"
                  else np.concatenate([data["train"][k], d_xl[k]], axis=0))
    P_big = np.concatenate([P_ex["train"], P_xl], axis=0)
    Pt_big = blur_teacher(torch.tensor(P_big.astype(np.float32)), N_GRID,
                          0.75, device=device)
    rows = []
    for size in (2500, 10000, 40000):
        sub = {k: (big[k] if k == "times" else big[k][:size]) for k in big}
        for mode, distill in [("M0", False), ("M1", True)]:
            print(f"  scale {mode} @ {size}", flush=True)
            model = train_ch4(cfg, sub, data["val"], device, distill,
                              Pt_big[:size] if distill else None,
                              Pv if distill else None, seed=1)
            r, _, _ = audit_model(model, data, None, reg_e["sizes"], cfg,
                                  device, alpha)
            r.update(size=size, mode=mode)
            rows.append(r)
            print(f"    {r}", flush=True)
    out["scaling"] = rows
    np.savez_compressed(data_dir / "ch4_scaling.npz",
                        rows=np.array(json.dumps(rows)))

    with open(results_dir / "ch4_extras.json", "w") as f:
        json.dump(out, f, indent=2)
    update_json(results_dir / "gates.json", {"CH4_extras": {
        "n_seed_rows": len(out["seeds"]),
        "raw_teacher_ineff_median_identifiable":
            out["raw_teacher"]["ineff_median_identifiable"],
        "dose_knob1": dose}})
    print("CH4 EXTRAS DONE", flush=True)


if __name__ == "__main__":
    main()
