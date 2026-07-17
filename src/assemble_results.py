"""Stage 7: assemble results/placeholder_map.json + main tables (G5).

Regenerates all main tables deterministically from the frozen artifacts
(results/*.json, data/*.npz).  Also emits ready-to-paste LaTeX rows.

Usage:  python src/assemble_results.py [--skip-seed2]
"""
import argparse
import json

import numpy as np

from common import load_config, resolve, update_json


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seed2", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    results_dir = resolve(cfg, "results_dir")
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    gates = load_json(results_dir / "gates.json")
    conf_l = load_json(results_dir / "conformal_model_seed1.json")
    conf_e = load_json(results_dir / "conformal_exact.json")
    audit = load_json(results_dir / "audit_model_seed1.json")
    dose = load_json(results_dir / "dose_response_model_seed1.json")

    m = conf_l["main"]
    e = conf_e["main"]
    a = audit

    pm = {
        # --- headline table (paper Table 3) ---
        "map_error_mean_learned": a["map_error"]["learned_mean"],
        "map_error_median_learned": a["map_error"]["learned_median"],
        "map_error_mean_exact": a["map_error"]["exact_mean"],
        "map_error_median_exact": a["map_error"]["exact_median"],
        "map_error_mean_peak_sensor": a["map_error"]["peak_sensor_mean"],
        "map_error_median_peak_sensor": a["map_error"]["peak_sensor_median"],
        "coverage_90_learned": m["coverage"],
        "coverage_90_learned_ci": m["cp_ci"],
        "coverage_90_exact": e["coverage"],
        "coverage_90_exact_ci": e["cp_ci"],
        "region_area_mean_learned": m["area_mean"],
        "region_area_median_learned": m["area_median"],
        "region_area_mean_exact": e["area_mean"],
        "region_area_median_exact": e["area_median"],
        # --- H2 ---
        "h2_spearman": a["h2"]["spearman_area"],
        "inefficiency_factor_mean": a["h2"]["inefficiency_mean"],
        "inefficiency_factor_median": a["h2"]["inefficiency_median"],
        # --- H3 ---
        "h3_knob1_slope_ratio": dose["knob1"]["slope_ratio_mean_curves"],
        "h3_knob1_spearman_mean": dose["knob1"]["spearman_per_scenario_mean"],
        "h3_knob1_spearman_median": dose["knob1"]["spearman_per_scenario_median"],
        "h3_knob2_ratio_learned": dose["knob2"]["confined_over_spread_learned_mean"],
        "h3_knob2_ratio_exact": dose["knob2"]["confined_over_spread_exact_mean"],
        "h3_knob2_spearman": dose["knob2"]["spearman_ratio_agreement"],
        "h3_knob3_slope_ratio": dose["knob3"]["slope_ratio_mean_curves"],
        "h3_knob3_spearman_mean": dose["knob3"]["spearman_per_scenario_mean"],
        "h3_knob3_spearman_median": dose["knob3"]["spearman_per_scenario_median"],
        # --- appendix auxiliary ---
        "nll_mean": a["nll_mean"],
        "jsd_mean": a["jsd_mean"],
        "map_error_percentiles_50_90_99": a["map_error"]["learned_p50_p90_p99"],
        "latency_ms_per_scenario": a["latency_ms_per_scenario"],
        # --- gates / appendix A-B ---
        "gate_G1": gates.get("G1", {}),
        "gate_G2": gates.get("G2", {}),
        "gate_G3": gates.get("G3", {}),
        "gate_G3_exact": gates.get("G3_exact", {}),
        "gate_G4": {k: v for k, v in gates.get("G4", {}).items()
                    if not k.startswith("h2_")},
        "gate_H3": gates.get("H3", {}),
    }
    for lev in ["0.80", "0.95"]:
        key = f"level_{lev}"
        if key in conf_l["aux"]:
            pm[f"coverage_{lev}_learned"] = conf_l["aux"][key]["coverage"]
            pm[f"area_median_{lev}_learned"] = conf_l["aux"][key]["area_median"]

    if not args.skip_seed2 and (results_dir / "conformal_model_seed2.json").exists():
        c2 = load_json(results_dir / "conformal_model_seed2.json")["main"]
        a2 = load_json(results_dir / "audit_model_seed2.json")
        pm["seed2_coverage_90"] = c2["coverage"]
        pm["seed2_area_median"] = c2["area_median"]
        pm["seed2_map_error_mean"] = a2["map_error"]["learned_mean"]
        pm["seed2_h2_spearman"] = a2["h2"]["spearman_area"]
        pm["seed2_inefficiency_median"] = a2["h2"]["inefficiency_median"]
        pm["seed_spread_map_error_mean"] = abs(
            pm["map_error_mean_learned"] - pm["seed2_map_error_mean"])
        pm["seed_spread_coverage"] = abs(
            pm["coverage_90_learned"] - pm["seed2_coverage_90"])

    with open(results_dir / "placeholder_map.json", "w") as f:
        json.dump(pm, f, indent=2, sort_keys=True)

    # ---------------- headline table CSV + LaTeX --------------------------
    rows = [
        ("MAP error (mean)", a["map_error"]["learned_mean"],
         a["map_error"]["exact_mean"], a["map_error"]["peak_sensor_mean"]),
        ("MAP error (median)", a["map_error"]["learned_median"],
         a["map_error"]["exact_median"], a["map_error"]["peak_sensor_median"]),
        ("Coverage @ 90%", m["coverage"], e["coverage"], None),
        ("CP 95% CI low", m["cp_ci"][0], e["cp_ci"][0], None),
        ("CP 95% CI high", m["cp_ci"][1], e["cp_ci"][1], None),
        ("Region area (mean)", m["area_mean"], e["area_mean"], None),
        ("Region area (median)", m["area_median"], e["area_median"], None),
    ]
    with open(tables_dir / "table3_headline.csv", "w") as f:
        f.write("metric,learned_conformal,exact_oracle,peak_sensor\n")
        for r in rows:
            f.write(",".join([r[0]] + ["" if v is None else f"{v:.4f}"
                                       for v in r[1:]]) + "\n")
    with open(tables_dir / "table3_headline.tex", "w") as f:
        f.write(f"MAP error (mean) & {a['map_error']['learned_mean']:.3f} & "
                f"{a['map_error']['exact_mean']:.3f} & "
                f"{a['map_error']['peak_sensor_mean']:.3f} \\\\\n")
        f.write(f"MAP error (median) & {a['map_error']['learned_median']:.3f} & "
                f"{a['map_error']['exact_median']:.3f} & "
                f"{a['map_error']['peak_sensor_median']:.3f} \\\\\n")
        f.write(f"Coverage @ 90\\% & {m['coverage']:.3f} & {e['coverage']:.3f} & --- \\\\\n")
        f.write(f"\\quad Clopper--Pearson 95\\% CI & "
                f"[{m['cp_ci'][0]:.3f}, {m['cp_ci'][1]:.3f}] & "
                f"[{e['cp_ci'][0]:.3f}, {e['cp_ci'][1]:.3f}] & --- \\\\\n")
        f.write(f"Region area (mean) & {m['area_mean']:.4f} & {e['area_mean']:.4f} & --- \\\\\n")
        f.write(f"Region area (median) & {m['area_median']:.4f} & {e['area_median']:.4f} & --- \\\\\n")

    # ---------------- auxiliary table (paper Table 4) ----------------------
    aux = conf_l["aux"]
    with open(tables_dir / "table4_auxiliary.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"coverage_80,{aux['level_0.80']['coverage']:.4f}\n")
        f.write(f"coverage_90,{m['coverage']:.4f}\n")
        f.write(f"coverage_95,{aux['level_0.95']['coverage']:.4f}\n")
        f.write(f"area_median_80,{aux['level_0.80']['area_median']:.4f}\n")
        f.write(f"area_median_90,{m['area_median']:.4f}\n")
        f.write(f"area_median_95,{aux['level_0.95']['area_median']:.4f}\n")
        f.write(f"nll_mean,{a['nll_mean']:.4f}\n")
        f.write(f"jsd_mean,{a['jsd_mean']:.4f}\n")
        p50, p90, p99 = a["map_error"]["learned_p50_p90_p99"]
        f.write(f"map_err_p50,{p50:.4f}\nmap_err_p90,{p90:.4f}\nmap_err_p99,{p99:.4f}\n")
        f.write(f"latency_ms,{a['latency_ms_per_scenario']:.2f}\n")

    # coverage sweep CSV (fig3 source, referenced by paper as coverage.csv)
    with open(tables_dir / "coverage.csv", "w") as f:
        f.write("nominal,conformal,raw_hpd\n")
        for r in conf_l["sweep"]:
            f.write(f"{r['nominal']:.4f},{r['conformal']:.4f},{r['raw_hpd']:.4f}\n")

    update_json(results_dir / "gates.json",
                {"G5": {"placeholder_map_written": True,
                        "n_placeholder_keys": len(pm)}})
    print(f"placeholder_map.json written with {len(pm)} keys")
    print(json.dumps({k: v for k, v in pm.items() if not k.startswith("gate_")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
