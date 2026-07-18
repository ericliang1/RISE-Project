"""Appendix E (stretch): 96x96 grid-refinement check on 100 test scenarios.

Recomputes exact posteriors on a 96x96 grid for the full calibration split and
the first 100 test scenarios, re-runs the identical conformal-on-exact
procedure, and reports the change in coverage (on those 100) and median area
vs the 64x64 grid.

Usage:  python src/refine_check.py
"""
import numpy as np

from common import get_device, load_config, pos_to_cell, resolve, update_json
from conformal import conformal_threshold, nonconformity_scores, regions
from exact_posterior import posteriors_for_split
from inference import load_split


def main():
    cfg = load_config()
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    n64 = cfg["grid"]["n"]
    n96 = cfg["grid"]["refine_n"]
    n_test = cfg["evaluation"]["refine_scenarios"]
    alpha = cfg["conformal"]["alpha"]
    rng = np.random.default_rng(cfg["conformal"]["score_seed"] + 1)

    d_calib = load_split(data_dir, "calib")
    d_test = load_split(data_dir, "test")
    d_test_sub = {k: (v[:n_test] if hasattr(v, "shape") and len(v) == len(d_test["ids"])
                      else v) for k, v in d_test.items()}

    print(f"96x96 posteriors: calib ({len(d_calib['ids'])})...", flush=True)
    P_c96, _, ne1 = posteriors_for_split(d_calib, cfg, device, grid_n=n96)
    print(f"96x96 posteriors: test subset ({n_test})...", flush=True)
    P_t96, _, ne2 = posteriors_for_split(d_test_sub, cfg, device, grid_n=n96)

    tc_c96 = pos_to_cell(d_calib["xs"], n96)
    tc_t96 = pos_to_cell(d_test_sub["xs"], n96)
    scores96 = nonconformity_scores(P_c96.astype(np.float64), tc_c96, rng)
    qhat96 = conformal_threshold(scores96, alpha)
    reg96 = regions(P_t96.astype(np.float64), qhat96, tc_t96)
    cov96 = float(reg96["covered"].mean())
    area96 = float(np.median(reg96["sizes"] / (n96 * n96)))

    # 64x64 reference restricted to the same 100 scenarios
    P_c64 = np.load(data_dir / "calib_posterior.npz")["probs"].astype(np.float64)
    P_t64 = np.load(data_dir / "test_posterior.npz")["probs"].astype(np.float64)[:n_test]
    scores64 = nonconformity_scores(P_c64, d_calib["true_cell"],
                                    np.random.default_rng(cfg["conformal"]["score_seed"] + 1))
    qhat64 = conformal_threshold(scores64, alpha)
    reg64 = regions(P_t64, qhat64, d_test_sub["true_cell"])
    cov64 = float(reg64["covered"].mean())
    area64 = float(np.median(reg64["sizes"] / (n64 * n64)))

    out = {"coverage_96_minus_64": cov96 - cov64,
           "median_area_96_minus_64": area96 - area64,
           "coverage_96": cov96, "coverage_64_subset": cov64,
           "median_area_96": area96, "median_area_64_subset": area64,
           "norm_err_96": max(ne1, ne2), "n_scenarios": n_test}
    update_json(results_dir / "gates.json", {"refine_check": out})
    print(out)


if __name__ == "__main__":
    main()
