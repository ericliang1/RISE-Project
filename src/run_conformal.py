"""Stage 4 runner: conformal calibration + test coverage (G3 / H1).

Works for both the learned model (--probs model --seed K) and the exact
posterior oracle (--probs exact, after Stage 5), applying the IDENTICAL
procedure: scores on the calibration split, threshold at
ceil((n+1)(1-alpha)), regions on the test split.

Outputs:
  results/conformal_<tag>.json      coverage/areas at alpha and aux levels + sweep
  data/scores_<tag>.npz             calibration scores (reused by dose-response)
  data/regions_<tag>_test.npz       per-scenario region sizes at 90%
  data/heatmaps_<tag>_<split>.npz   cached heatmaps (model only)
"""
import argparse
import json

import numpy as np

from common import get_device, load_config, resolve, update_json
from conformal import coverage_report, nominal_sweep, tail_scores
from inference import ckpt_stem, heatmaps, load_model, load_split


def get_probs(tag, split, cfg, device):
    data_dir = resolve(cfg, "data_dir")
    if tag.startswith(("model_seed", "model2_seed")):
        arch, seed_s = tag.rsplit("_seed", 1)
        seed = int(seed_s)
        ckpt_path = resolve(cfg, "checkpoints_dir") / f"{ckpt_stem(arch, seed)}.pt"
        from common import sha256_file
        ckpt_sha = sha256_file(ckpt_path)
        cache = data_dir / f"heatmaps_{tag}_{split}.npz"
        if cache.exists():
            z = np.load(cache)
            if "ckpt_sha" in z and str(z["ckpt_sha"]) == ckpt_sha:
                return z["probs"]           # cache matches current checkpoint
        model, _ = load_model(cfg, ckpt_path, device)
        d = load_split(data_dir, split)
        P = heatmaps(model, d, device)
        np.savez_compressed(cache, ids=d["ids"], probs=P, ckpt_sha=ckpt_sha)
        return P
    elif tag == "exact":
        return np.load(data_dir / f"{split}_posterior.npz")["probs"].astype(np.float64)
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", choices=["model", "model2", "exact"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    tag = "exact" if args.probs == "exact" else f"{args.probs}_seed{args.seed}"
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")

    d_calib = load_split(data_dir, "calib")
    d_test = load_split(data_dir, "test")
    P_calib = get_probs(tag, "calib", cfg, device)
    P_test = get_probs(tag, "test", cfg, device)

    cc = cfg["conformal"]
    rng = np.random.default_rng(cc["score_seed"])
    tails = tail_scores(P_calib.astype(np.float64), d_calib["true_cell"], rng)
    np.savez_compressed(data_dir / f"scores_{tag}.npz", tails=tails)

    alpha = cc["alpha"]
    rep = coverage_report(P_test.astype(np.float64), d_test["true_cell"],
                          tails, alpha)
    sizes = rep.pop("sizes")
    np.savez_compressed(data_dir / f"regions_{tag}_test.npz",
                        ids=d_test["ids"], sizes=sizes,
                        tail_qhat=rep["tail_qhat"])

    aux = {}
    for lev_alpha in [1.0 - l for l in cc["aux_levels"]]:
        r = coverage_report(P_test.astype(np.float64), d_test["true_cell"],
                            tails, lev_alpha)
        r.pop("sizes")
        aux[f"level_{1-lev_alpha:.2f}"] = r

    lo, hi = cc["nominal_sweep_range"]
    sweep = nominal_sweep(P_test.astype(np.float64), d_test["true_cell"], tails,
                          np.linspace(lo, hi, cc["nominal_sweep"]))

    out = {"tag": tag, "main": rep, "aux": aux, "sweep": sweep}
    with open(results_dir / f"conformal_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    ci = rep["cp_ci"]
    gate_pass = ci[0] <= 1 - alpha <= ci[1]
    gate_key = "G3" if tag == "model_seed1" else f"G3_{tag}"
    update_json(results_dir / "gates.json", {gate_key: {
        "coverage_at_90": rep["coverage"], "cp_ci": ci,
        "ci_contains_nominal": bool(gate_pass),
        "area_mean": rep["area_mean"], "area_median": rep["area_median"]}})
    print(f"[{tag}] coverage@90 = {rep['coverage']:.4f} "
          f"CI [{ci[0]:.4f}, {ci[1]:.4f}] contains 0.90: {gate_pass} | "
          f"area mean {rep['area_mean']:.4f} median {rep['area_median']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
