"""Select the physics-distilled variant (mix_lambda) on VALIDATION data only.

Rule (documented): the winning variant minimizes the median validation raw-HPD
area at mass 0.90.  Coverage is later guaranteed by the identical conformal
procedure, so the selection targets sharpness; the selection never touches the
calibration or test splits.  Copies the winner to checkpoints/model2_seed1.pt.

Usage:  python src/select_distill.py --candidates model2_lam10 model2_lam05
"""
import argparse
import json
import shutil

import numpy as np

from common import get_device, load_config, resolve
from conformal import region_mask
from inference import heatmaps, load_model, load_split


def val_stats(stem, cfg, device, d_val, P_val_teacher):
    ckpt = resolve(cfg, "checkpoints_dir") / f"{stem}.pt"
    model, meta = load_model(cfg, ckpt, device)
    P = heatmaps(model, d_val, device).astype(np.float64)
    n_cells = P.shape[1]
    areas = np.array([region_mask(P[i], 0.10).sum() / n_cells
                      for i in range(len(P))])
    covered = np.array([region_mask(P[i], 0.10)[d_val["true_cell"][i]]
                        for i in range(len(P))])
    logp = np.log(np.maximum(P, 1e-300))
    tce = float(-(P_val_teacher * logp).sum(1).mean())
    return {"stem": stem, "epoch": int(meta["epoch"]),
            "val_hpd90_area_median": float(np.median(areas)),
            "val_hpd90_area_mean": float(areas.mean()),
            "val_hpd90_coverage": float(covered.mean()),
            "val_teacher_ce": tce}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    d_val = load_split(data_dir, "val")
    P_val_teacher = np.load(data_dir / "val_posterior.npz")["probs"]

    stats = [val_stats(s, cfg, device, d_val, P_val_teacher)
             for s in args.candidates]
    for s in stats:
        print(s, flush=True)
    winner = min(stats, key=lambda s: s["val_hpd90_area_median"])
    ckpt_dir = resolve(cfg, "checkpoints_dir")
    shutil.copyfile(ckpt_dir / f"{winner['stem']}.pt",
                    ckpt_dir / "model2_seed1.pt")
    out = {"rule": "min median val raw-HPD-0.90 area", "candidates": stats,
           "winner": winner["stem"]}
    with open(resolve(cfg, "results_dir") / "distill_variant_selection.json",
              "w") as f:
        json.dump(out, f, indent=2)
    print("winner:", winner["stem"])


if __name__ == "__main__":
    main()
