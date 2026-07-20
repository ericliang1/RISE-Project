"""CH4-T completion campaign: seed stability (M0/M1 x 3), raw-teacher ablation,
sensor-mast dose-response.  Reuses methane_t.  Reports radii in meters.

Usage:  python src/methane_t_campaign.py
"""
import json

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from common import cell_centers, get_device, load_config, pos_to_cell, resolve, update_json
from conformal import region_mask, regions, tail_scores, tail_threshold
from methane_pipeline import ch4_cfg
from methane_t import (DeepSetsT, N_GRID, T, blur_teacher, model_probs,
                       posteriors, sensor_responses_t, to_batch_t, train)

L = 500.0


def rad_m(area_frac):
    return float(np.sqrt(np.median(area_frac) / np.pi) * L)


def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    alpha = cfg["conformal"]["alpha"]
    data = {n: dict(np.load(dd / f"ch4t_{n}.npz", allow_pickle=True))
            for n in ("train", "val", "calib", "test")}
    P_ex = {n: np.load(dd / f"ch4t_{n}_posterior.npz")["probs"]
            for n in ("train", "val", "calib", "test")}
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"], data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    e_sizes = reg_e["sizes"]
    Pt = blur_teacher(torch.tensor(P_ex["train"].astype(np.float32)), N_GRID, 0.75, device=device)
    Pv = blur_teacher(torch.tensor(P_ex["val"].astype(np.float32)), N_GRID, 0.75, device=device)
    Pt_raw = torch.tensor(P_ex["train"].astype(np.float32))
    Pv_raw = torch.tensor(P_ex["val"].astype(np.float32))

    def audit(model, save=None):
        Pc = model_probs(model, data["calib"], device)
        Ptt = model_probs(model, data["test"], device)
        r = np.random.default_rng(cfg["conformal"]["score_seed"])
        th = tail_threshold(tail_scores(Pc, data["calib"]["true_cell"], r), alpha)
        reg = regions(Ptt, th, data["test"]["true_cell"])
        if save:
            np.savez_compressed(dd / save, sizes=reg["sizes"], )
        return {"coverage": float(reg["covered"].mean()),
                "radius_m": rad_m(reg["sizes"] / 4096),
                "ineff_median": float(np.median(reg["sizes"] / np.maximum(e_sizes, 1)))}, reg["sizes"]

    out = {"oracle_radius_m": rad_m(e_sizes / 4096), "seeds": []}
    ckpt = {}
    print("== seeds ==", flush=True)
    for seed in (1, 2, 3):
        for mode, distill in [("M0", False), ("M1", True)]:
            model = train(cfg, data["train"], data["val"], device, distill,
                          Pt if distill else None, Pv if distill else None, seed=seed)
            a, sizes = audit(model)
            a.update(mode=mode, seed=seed)
            out["seeds"].append(a)
            if seed == 1:
                ckpt[mode] = sizes
            print(f"  {mode} s{seed}: {a}", flush=True)

    print("== raw teacher ==", flush=True)
    model = train(cfg, data["train"], data["val"], device, True, Pt_raw, Pv_raw, seed=1)
    a, _ = audit(model)
    out["raw_teacher"] = a
    print(f"  raw: {a}", flush=True)

    # significance on seed-1 M0 vs M1
    r0 = ckpt["M0"] / np.maximum(e_sizes, 1)
    r1 = ckpt["M1"] / np.maximum(e_sizes, 1)
    w = wilcoxon(np.log(r0), np.log(r1), alternative="greater")
    out["wilcoxon_p"] = float(w.pvalue)
    out["frac_M1_sharper"] = float((r1 < r0).mean())

    with open(rr / "ch4t_campaign.json", "w") as f:
        json.dump(out, f, indent=2)
    update_json(rr / "gates.json", {"CH4T_campaign": out})
    print("CH4T CAMPAIGN DONE", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
