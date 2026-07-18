"""Stage 5 part 2: sharpness audit (H2), D1 scatter, and auxiliary metrics.

Both learned and exact regions sit at their own verified 90% conformal
thresholds (identical procedure).  Reports per-scenario Spearman of areas,
inefficiency factor (mean/median learned/exact area ratio), D1 scatter data,
MAP errors (learned / exact / peak-sensor), NLL, JSD, percentiles, latency.

Usage:  python src/audit.py --seed 1
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from baselines import localization_errors, peak_sensor_estimates
from common import cell_center_of, get_device, load_config, resolve, update_json
from inference import ckpt_stem, load_model, load_split, to_batch


def jsd_rows(P, Q, eps=1e-300):
    """Jensen-Shannon divergence (nats) per row, float64."""
    P = P.astype(np.float64) + eps
    Q = Q.astype(np.float64) + eps
    P /= P.sum(1, keepdims=True)
    Q /= Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    kl = lambda A, B: (A * (np.log(A) - np.log(B))).sum(1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", choices=["model", "model2"], default="model")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    tag = f"{args.arch}_seed{args.seed}"
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    n_grid = cfg["grid"]["n"]
    n_cells = n_grid * n_grid

    d_test = load_split(data_dir, "test")
    P_learn = np.load(data_dir / f"heatmaps_{tag}_test.npz")["probs"]
    post = np.load(data_dir / f"test_posterior.npz")
    P_exact = post["probs"]
    reg_l = np.load(data_dir / f"regions_{tag}_test.npz")
    reg_e = np.load(data_dir / "regions_exact_test.npz")
    area_l = reg_l["sizes"] / n_cells
    area_e = reg_e["sizes"] / n_cells

    # --- H2: sharpness at matched verified 90% coverage -------------------
    rho, rho_p = spearmanr(area_l, area_e)
    ratio = area_l / area_e
    h2 = {"spearman_area": float(rho), "spearman_p": float(rho_p),
          "inefficiency_mean": float(ratio.mean()),
          "inefficiency_median": float(np.median(ratio))}

    # --- MAP errors -------------------------------------------------------
    est_l = cell_center_of(P_learn.argmax(1), n_grid)
    est_e = cell_center_of(P_exact.argmax(1), n_grid)
    err_l = localization_errors(est_l, d_test["xs"])
    err_e = localization_errors(est_e, d_test["xs"])
    err_ps = localization_errors(peak_sensor_estimates(d_test), d_test["xs"])

    # --- NLL / JSD / D1 ---------------------------------------------------
    tc = d_test["true_cell"]
    nll = float(-np.log(np.maximum(P_learn[np.arange(len(tc)), tc], 1e-300)).mean())
    jsd = jsd_rows(P_learn, P_exact)

    # --- latency (batch 1) ------------------------------------------------
    model, _ = load_model(cfg, resolve(cfg, "checkpoints_dir") / f"{ckpt_stem(args.arch, args.seed)}.pt",
                          device)
    n_lat = cfg["evaluation"]["latency_scenarios"]
    with torch.no_grad():
        for i in range(10):                                   # warmup
            model(to_batch(d_test, np.array([i]), device))
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_lat):
            model(to_batch(d_test, np.array([i]), device))
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_lat * 1000.0

    out = {
        "h2": h2,
        "map_error": {
            "learned_mean": float(err_l.mean()), "learned_median": float(np.median(err_l)),
            "exact_mean": float(err_e.mean()), "exact_median": float(np.median(err_e)),
            "peak_sensor_mean": float(err_ps.mean()),
            "peak_sensor_median": float(np.median(err_ps)),
            "learned_p50_p90_p99": [float(np.percentile(err_l, p)) for p in (50, 90, 99)],
        },
        "nll_mean": nll,
        "jsd_mean": float(jsd.mean()), "jsd_median": float(np.median(jsd)),
        "latency_ms_per_scenario": latency_ms,
    }
    with open(results_dir / f"audit_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    # D1 + H2 scatter data for figures
    np.savez_compressed(data_dir / f"audit_scatter_{tag}.npz",
                        area_learned=area_l, area_exact=area_e,
                        map_err_learned=err_l, jsd=jsd)

    update_json(results_dir / "gates.json", {"G4": {
        f"h2_spearman_{tag}": h2["spearman_area"],
        f"h2_inefficiency_mean_{tag}": h2["inefficiency_mean"],
        f"h2_inefficiency_median_{tag}": h2["inefficiency_median"],
        f"h2_measured_{tag}": True}})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
