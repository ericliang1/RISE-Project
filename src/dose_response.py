"""Stage 6: information dose-response experiments (H3; inference only).

Knob 1 (sensor count): nested subsets N = 4..12 of each base scenario's frozen
  master layout in the frozen insertion order; readings are rows of the master
  reading set (a controlled intervention — same noise realization).
Knob 2 (geometry): matched pairs at N = 6, stratified vs confined layouts.
Knob 3 (noise): sigma sweep with regenerated readings on 8 master sensors.

Learned regions use the seed-1 conformal threshold; exact regions use the
exact-oracle conformal threshold (both calibrated at 90% on the calibration
split — region-size language only on these constructed sets).

Slopes: OLS of region area vs N (Knob 1) and vs log10(sigma) (Knob 3);
contraction-slope ratio = slope_learned / slope_exact (mean-curve slopes,
plus the per-scenario ratio distribution).

Usage:  python src/dose_response.py --seed 1
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from common import get_device, load_config, resolve, update_json
from conformal import region_mask, tail_threshold
from exact_posterior import log_posterior_scenario
from inference import heatmaps, load_model, load_split


def subset_dict(d, i, rows):
    """One-scenario dict restricted to sensor rows `rows` of scenario i."""
    S = len(rows)
    pad = lambda a, fill: np.concatenate(
        [a, np.full((12 - S,) + a.shape[1:], fill, dtype=a.dtype)], axis=0)[None]
    return {
        "ids": np.array([f"{d['ids'][i]}-sub{S}"]),
        "sensors": pad(d["sensors"][i, rows], 0.0),
        "readings": pad(d["readings"][i, rows], 0.0),
        "keep": pad(d["keep"][i, rows], False),
        "times": d["times"],
        "u": d["u"][i][None], "D": d["D"][i][None],
        "sigma": d["sigma"][i][None], "n_sensors": np.array([S]),
        "xs": d["xs"][i][None], "q": d["q"][i][None],
        "true_cell": d["true_cell"][i][None],
    }


def area_pair(sub, model, cfg, device, qhat_l, qhat_e, n_cells):
    """(learned area, exact area) for a one-scenario dict."""
    P = heatmaps(model, sub, device)[0].astype(np.float64)
    a_l = region_mask(P, qhat_l).sum() / n_cells
    lp = log_posterior_scenario(sub, 0, cfg, device)
    Pe = torch.exp(lp).cpu().numpy()
    a_e = region_mask(Pe, qhat_e).sum() / n_cells
    return float(a_l), float(a_e)


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def safe_ratio(a, b):
    return float(a / b) if abs(b) > 1e-12 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    tag = f"model_seed{args.seed}"
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    n_cells = cfg["grid"]["n"] ** 2
    alpha = cfg["conformal"]["alpha"]

    qhat_l = tail_threshold(np.load(data_dir / f"scores_{tag}.npz")["tails"], alpha)
    qhat_e = tail_threshold(np.load(data_dir / "scores_exact.npz")["tails"], alpha)
    model, _ = load_model(cfg, resolve(cfg, "checkpoints_dir") / f"seed{args.seed}.pt",
                          device)

    lo_N, hi_N = cfg["dose_response"]["nested_range"]
    Ns = list(range(lo_N, hi_N + 1))

    # ---------------- Knob 1: sensor count --------------------------------
    d1 = load_split(data_dir, "dose1_master")
    B = len(d1["ids"])
    curves_l = np.zeros((B, len(Ns)))
    curves_e = np.zeros((B, len(Ns)))
    for b in range(B):
        order = d1["order"][b]
        for j, N in enumerate(Ns):
            sub = subset_dict(d1, b, order[:N])
            curves_l[b, j], curves_e[b, j] = area_pair(sub, model, cfg, device,
                                                       qhat_l, qhat_e, n_cells)
    k1_rho = [spearmanr(curves_l[b], curves_e[b]).statistic for b in range(B)]
    k1_slope_l = [ols_slope(Ns, curves_l[b]) for b in range(B)]
    k1_slope_e = [ols_slope(Ns, curves_e[b]) for b in range(B)]
    k1_ratio_per = [safe_ratio(sl, se) for sl, se in zip(k1_slope_l, k1_slope_e)]
    knob1 = {
        "spearman_per_scenario_mean": float(np.nanmean(k1_rho)),
        "spearman_per_scenario_median": float(np.nanmedian(k1_rho)),
        "slope_ratio_mean_curves": safe_ratio(np.mean(k1_slope_l), np.mean(k1_slope_e)),
        "slope_ratio_per_scenario_median": float(np.nanmedian(k1_ratio_per)),
        "slope_learned_mean": float(np.mean(k1_slope_l)),
        "slope_exact_mean": float(np.mean(k1_slope_e)),
    }

    # ---------------- Knob 2: geometry pairs at N = 6 ---------------------
    d2s = load_split(data_dir, "dose2_spread")
    d2c = load_split(data_dir, "dose2_confined")
    ratios_l, ratios_e, area_s_l, area_c_l, area_s_e, area_c_e = [], [], [], [], [], []
    for b in range(B):
        rows = np.arange(6)
        asl, ase = area_pair(subset_dict(d2s, b, rows), model, cfg, device,
                             qhat_l, qhat_e, n_cells)
        acl, ace = area_pair(subset_dict(d2c, b, rows), model, cfg, device,
                             qhat_l, qhat_e, n_cells)
        area_s_l.append(asl); area_c_l.append(acl)
        area_s_e.append(ase); area_c_e.append(ace)
        ratios_l.append(safe_ratio(acl, asl))
        ratios_e.append(safe_ratio(ace, ase))
    knob2 = {
        "confined_over_spread_learned_mean": float(np.nanmean(ratios_l)),
        "confined_over_spread_exact_mean": float(np.nanmean(ratios_e)),
        "confined_over_spread_learned_median": float(np.nanmedian(ratios_l)),
        "confined_over_spread_exact_median": float(np.nanmedian(ratios_e)),
        "spearman_ratio_agreement": float(spearmanr(ratios_l, ratios_e).statistic),
        "confined_broader_learned_frac": float(np.mean(np.array(ratios_l) > 1)),
        "confined_broader_exact_frac": float(np.mean(np.array(ratios_e) > 1)),
    }

    # ---------------- Knob 3: noise sweep ---------------------------------
    d3 = load_split(data_dir, "dose3_sweep")
    sig_vals = d3["sigma_values"]
    nsig = len(sig_vals)
    sw_l = np.zeros((B, nsig))
    sw_e = np.zeros((B, nsig))
    for idx in range(len(d3["ids"])):
        b = int(d3["scenario_idx"][idx])
        si = int(d3["sigma_idx"][idx])
        rows = np.arange(int(d3["n_sensors"][idx]))
        sw_l[b, si], sw_e[b, si] = area_pair(subset_dict(d3, idx, rows), model,
                                             cfg, device, qhat_l, qhat_e, n_cells)
    x = np.log10(sig_vals)
    k3_rho = [spearmanr(sw_l[b], sw_e[b]).statistic for b in range(B)]
    k3_slope_l = [ols_slope(x, sw_l[b]) for b in range(B)]
    k3_slope_e = [ols_slope(x, sw_e[b]) for b in range(B)]
    k3_ratio_per = [safe_ratio(sl, se) for sl, se in zip(k3_slope_l, k3_slope_e)]
    knob3 = {
        "spearman_per_scenario_mean": float(np.nanmean(k3_rho)),
        "spearman_per_scenario_median": float(np.nanmedian(k3_rho)),
        "slope_ratio_mean_curves": safe_ratio(np.mean(k3_slope_l), np.mean(k3_slope_e)),
        "slope_ratio_per_scenario_median": float(np.nanmedian(k3_ratio_per)),
        "slope_learned_mean": float(np.mean(k3_slope_l)),
        "slope_exact_mean": float(np.mean(k3_slope_e)),
    }

    out = {"knob1": knob1, "knob2": knob2, "knob3": knob3, "Ns": Ns,
           "sigma_values": list(map(float, sig_vals))}
    with open(results_dir / f"dose_response_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(data_dir / f"dose_curves_{tag}.npz",
                        Ns=np.array(Ns), curves_l=curves_l, curves_e=curves_e,
                        sig_vals=sig_vals, sweep_l=sw_l, sweep_e=sw_e,
                        geo_spread_l=np.array(area_s_l), geo_conf_l=np.array(area_c_l),
                        geo_spread_e=np.array(area_s_e), geo_conf_e=np.array(area_c_e))
    update_json(results_dir / "gates.json", {"H3": {
        "knob1_slope_ratio": knob1["slope_ratio_mean_curves"],
        "knob1_spearman_median": knob1["spearman_per_scenario_median"],
        "knob2_ratio_learned": knob2["confined_over_spread_learned_mean"],
        "knob2_ratio_exact": knob2["confined_over_spread_exact_mean"],
        "knob3_slope_ratio": knob3["slope_ratio_mean_curves"],
        "knob3_spearman_median": knob3["spearman_per_scenario_median"]}})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
