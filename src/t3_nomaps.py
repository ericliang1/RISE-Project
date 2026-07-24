"""Table-3 confound cell: grid head + distillation, NO maps, noisy wind.
Isolates the physics-map effect from the head effect on Track 2.
Mirrors s_det_det exactly minus the stats channels (plain DeepSetsT via
methane_t.train, det-oracle teacher, observed-wind tokens)."""
import json

import numpy as np
import torch

from common import get_device, load_config, resolve
from methane_t import N_GRID, SPLITS, model_probs, train
from methane_t_uncertain import audit, with_obs_wind
from train import blur_teacher


def main():
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    data = {}
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        data[name] = with_obs_wind(d, np.load(dd / f"ch4tu_{name}_uobs.npy"))
    P_tr = torch.tensor(np.load(dd / "ch4tu_train_oracle_det.npz")
                        ["probs"].astype(np.float32))
    P_val = torch.tensor(np.load(dd / "ch4tu_val_oracle_det.npz")
                         ["probs"].astype(np.float32))
    t_tr = blur_teacher(P_tr, N_GRID, 0.75, device=device).to(device)
    t_val = blur_teacher(P_val, N_GRID, 0.75, device=device).to(device)
    model = train(cfg, data["train"], data["val"], device, True, t_tr, t_val,
                  seed=1)
    Pc = model_probs(model, data["calib"], device)
    Pt = model_probs(model, data["test"], device)
    ref = np.load(dd / "ch4tu_audit_s_det_det.npz")["sizes"]
    audit(Pc, Pt, data, cfg, "s_nomaps_det", rr, ref_sizes=ref,
          npz_path=dd / "ch4tu_audit_s_nomaps_det.npz")


if __name__ == "__main__":
    main()
