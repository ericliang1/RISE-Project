"""Gap decomposition: how much of (student 74 m vs Bayes 7 m) is the tempering
ceiling vs the student's failure to match its teacher?

Conformalizes the *teacher itself* at several blur strengths: for each sigma,
blur calib+test exact posteriors, run the identical conformal harness, report
coverage + median region radius (m).  sigma=0.75 is what M1 trains toward, so
its radius is the exact ceiling tempering imposes on a perfect student.

Usage:  python src/methane_t_diagnose.py
"""
import json

import numpy as np
import torch

from common import get_device, load_config, resolve
from conformal import regions, tail_scores, tail_threshold
from train import blur_teacher

L, N_GRID, N_CELLS = 500.0, 64, 4096


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * L)


def main():
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    alpha = cfg["conformal"]["alpha"]
    P = {s: np.load(dd / f"ch4t_{s}_posterior.npz")["probs"]
         for s in ("calib", "test")}
    tc = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))["true_cell"]
          for s in ("calib", "test")}

    rows = []
    for sig in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0):
        if sig == 0.0:
            Pc, Pt = P["calib"], P["test"]
        else:
            Pc, Pt = (blur_teacher(torch.tensor(P[s].astype(np.float32)),
                                   N_GRID, sig, device=device)
                      .cpu().numpy().astype(np.float64)
                      for s in ("calib", "test"))
        rng = np.random.default_rng(cfg["conformal"]["score_seed"])
        th = tail_threshold(tail_scores(Pc, tc["calib"], rng), alpha)
        reg = regions(Pt, th, tc["test"])
        rows.append({"blur_sigma_cells": sig,
                     "coverage": float(reg["covered"].mean()),
                     "median_radius_m": rad_m(reg["sizes"])})
        print(rows[-1], flush=True)

    out = {"teacher_sweep": rows,
           "students_for_reference_m": {"M1_deepsets": 73.8, "M2_gnn": 50.3,
                                        "M3_settransformer": 56.1},
           "note": "sigma=0.75 row = ceiling a perfect student of the paper's "
                   "teacher could reach; student minus that row = "
                   "approximation/optimization gap, not tempering."}
    with open(resolve(cfg, "results_dir") / "ch4t_gap_decomposition.json",
              "w") as f:
        json.dump(out, f, indent=2)
    print("DIAGNOSE DONE", flush=True)


if __name__ == "__main__":
    main()
