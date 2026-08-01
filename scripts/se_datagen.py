"""Super-emitter datagen: splits, exact posteriors (calib/test), gate.

Generates the four CH4-T splits under the super-emitter population
(q ~ LogUniform(100, 500) kg/h, S in {4..8}; everything else unchanged),
then computes exact posteriors for calib and test only (the training path
uses smoothed labels, not posteriors) and runs the blocking
conformal-on-exact self-calibration gate before anything trains.

Idempotent: existing files are reused, so the chain resumes cleanly after
a wall-clock kill.  Usage: python scripts/se_datagen.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")

from common import load_config, resolve, savez_atomic
from conformal import regions, tail_scores, tail_threshold
from methane_pipeline import ch4_cfg
from methane_t import SPLITS, gen_split, posteriors
from common import get_device

def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    dd = resolve(cfg, "data_dir")
    dd.mkdir(parents=True, exist_ok=True)
    root = cfg["seeds"]["root_entropy"]
    alpha = cfg["conformal"]["alpha"]

    print("== data (super-emitter population) ==", flush=True)
    data = {}
    for name, count in SPLITS.items():
        p = dd / f"ch4t_{name}.npz"
        if p.exists():
            data[name] = dict(np.load(p, allow_pickle=True))
        else:
            data[name] = gen_split(name, count, cfg, root)
            savez_atomic(p, **data[name])
        q = data[name]["q"]
        ns = data[name]["n_sensors"].astype(int)
        assert q.min() >= 100.0 and ns.min() >= 4 and ns.max() <= 8, \
            f"{name}: population violation (q_min {q.min():.1f}, S_min {ns.min()})"
        print(f"  {name}: {len(data[name]['ids'])}  q [{q.min():.0f}, "
              f"{q.max():.0f}] kg/h  masts [{ns.min()}, {ns.max()}]",
              flush=True)

    print("== exact posteriors (calib/test only) ==", flush=True)
    P_ex = {}
    for name in ("calib", "test"):
        p = dd / f"ch4t_{name}_posterior.npz"
        if p.exists():
            P_ex[name] = np.load(p)["probs"]
        else:
            P_ex[name], ne = posteriors(data[name], cfg_q, device)
            savez_atomic(p, ids=data[name]["ids"], probs=P_ex[name])
            print(f"  {name}: norm_err {ne:.2e}", flush=True)

    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    th_e = tail_threshold(tail_scores(P_ex["calib"],
                                      data["calib"]["true_cell"], rng), alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    cov_e = float(reg_e["covered"].mean())
    med_r = float(np.median(np.sqrt(reg_e["sizes"] / 4096 / np.pi) * 500))
    print(f"  conformal-on-exact coverage: {cov_e:.3f}  median radius "
          f"{med_r:.1f} m", flush=True)
    # GUARD: the exact posterior must self-calibrate; if not, the oracle
    # forward model is misspecified and every downstream number is meaningless.
    # Upper bound 0.985, not 0.97: the super-emitter population is sharper
    # (most calib posteriors are near-atomic), and the randomized-score /
    # deterministic-region tie handling over-covers sharp maps; 0.97 would
    # spuriously abort ~30% of the time (review-measured on the subset).
    assert 0.85 <= cov_e <= 0.985, (
        f"conformal-on-exact coverage {cov_e:.3f} outside [0.85, 0.985] -- "
        "aborting before any training.")
    print("== datagen gates passed ==", flush=True)


if __name__ == "__main__":
    main()
