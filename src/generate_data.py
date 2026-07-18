"""Benchmark + dose-response data generation (Stage 2).

Every scenario has a persistent ID and a deterministic RNG derived from
SeedSequence([root_entropy, split_tag, index]).  Splits are by complete
scenario.  Responses are generated in float32 (validated in float64, G1);
dropped readings are omitted via a keep-mask, never imputed.

Storage layout (npz, padded to 12 sensors):
  ids (S,) str, xs (S,2), q (S,), u (S,2), D (S,), sigma (S,), p_drop (S,),
  n_sensors (S,), sensors (S,12,2) [NaN pad], readings (S,12,30) f32 [0 pad],
  keep (S,12,30) bool, true_cell (S,) int64

Usage:  python src/generate_data.py [--config config/default.yaml]
"""
import argparse
import json

import numpy as np
import torch

from common import (REPO_ROOT, SPLIT_TAGS, get_device, load_config, pos_to_cell,
                    resolve, scenario_rng, sha256_file, update_json)
from forward_model import sensor_responses
from priors import obs_times, sample_params

MAX_SENSORS = 12


def _gen_readings(params_list, sensors_pad, n_sensors, times, cfg, device,
                  noise_rngs, p_drop_override=None, drop_rngs=None):
    """Clean responses (float32, batched on GPU) + noise + dropout mask."""
    ph = cfg["physics"]
    B = len(params_list)
    to32 = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)
    # NaN-padded sensors would poison the batched kernel; substitute zeros
    # (masked out afterwards).
    sp = np.nan_to_num(sensors_pad, nan=0.0)
    xs_all = np.array([p["xs"] for p in params_list])
    u_all = np.array([p["u"] for p in params_list])
    D_all = np.array([p["D"] for p in params_list])
    A = np.empty((B, MAX_SENSORS, len(times)), dtype=np.float32)
    for lo in range(0, B, 2048):
        hi = min(lo + 2048, B)
        A[lo:hi] = sensor_responses(
            to32(sp[lo:hi]), to32(times), to32(xs_all[lo:hi]),
            to32(u_all[lo:hi]), to32(D_all[lo:hi]),
            ph["sigma_s"], ph["quad_nodes"], ph["tau_max"]).cpu().numpy()

    T = len(times)
    readings = np.zeros((B, MAX_SENSORS, T), dtype=np.float32)
    keep = np.zeros((B, MAX_SENSORS, T), dtype=bool)
    for b, p in enumerate(params_list):
        n = n_sensors[b]
        clean = p["q"] * A[b, :n] / ph["c_ref"]
        noise = noise_rngs[b].normal(0.0, p["sigma"], size=(n, T))
        readings[b, :n] = (clean + noise).astype(np.float32)
        pd = p["p_drop"] if p_drop_override is None else p_drop_override
        rng_d = drop_rngs[b] if drop_rngs is not None else noise_rngs[b]
        keep[b, :n] = rng_d.uniform(size=(n, T)) >= pd
        if not keep[b].any():          # cannot happen for p_drop<=0.2, but guard
            keep[b, 0, 0] = True
    return readings, keep


def generate_split(split, count, cfg, device):
    root = cfg["seeds"]["root_entropy"]
    tag = SPLIT_TAGS[split]
    times = obs_times(cfg)
    n_grid = cfg["grid"]["n"]

    ids, params_list, sensors_pad, n_sensors, noise_rngs = [], [], [], [], []
    for i in range(count):
        rng = scenario_rng(root, tag, i)
        p = sample_params(rng, cfg)
        sensors = rng.uniform(0.0, 1.0, size=(p["n_sensors"], 2))
        pad = np.full((MAX_SENSORS, 2), np.nan)
        pad[: p["n_sensors"]] = sensors
        ids.append(f"{split}-{i:06d}")
        params_list.append(p)
        sensors_pad.append(pad)
        n_sensors.append(p["n_sensors"])
        noise_rngs.append(rng)          # continue the same stream for noise+dropout

    sensors_pad = np.array(sensors_pad)
    readings, keep = _gen_readings(params_list, sensors_pad, n_sensors, times,
                                   cfg, device, noise_rngs)
    xs = np.array([p["xs"] for p in params_list])
    return dict(
        ids=np.array(ids),
        xs=xs,
        q=np.array([p["q"] for p in params_list]),
        u=np.array([p["u"] for p in params_list]),
        D=np.array([p["D"] for p in params_list]),
        sigma=np.array([p["sigma"] for p in params_list]),
        p_drop=np.array([p["p_drop"] for p in params_list]),
        n_sensors=np.array(n_sensors, dtype=np.int64),
        sensors=sensors_pad,
        readings=readings,
        keep=keep,
        true_cell=pos_to_cell(xs, n_grid),
        times=times,
    )


# ------------------------------------------------------------ dose-response

def _lhs_sources(n, rng, lo, hi):
    """Latin-hypercube source locations, roughly spanning the domain."""
    out = np.empty((n, 2))
    for d in range(2):
        perm = rng.permutation(n)
        out[:, d] = lo + (hi - lo) * (perm + rng.uniform(size=n)) / n
    return out


def generate_dose_response(cfg, device):
    """Base scenarios + master layouts (Knob 1), geometry pairs (Knob 2),
    sigma sweeps (Knob 3).  p_drop = 0 on all dose-response sets (documented)."""
    dr = cfg["dose_response"]
    root = cfg["seeds"]["root_entropy"]
    times = obs_times(cfg)
    n_grid = cfg["grid"]["n"]
    n_base = dr["n_base"]
    gN = dr["geometry_n"]
    gx, gy = dr["geometry_grid"]

    src_rng = scenario_rng(root, SPLIT_TAGS["dose_base"], 999999)
    lhs = _lhs_sources(n_base, src_rng,
                       cfg["prior"]["source_low"], cfg["prior"]["source_high"])

    base = []
    for i in range(n_base):
        rng = scenario_rng(root, SPLIT_TAGS["dose_base"], i)
        p = sample_params(rng, cfg)
        p["xs"] = lhs[i]                       # spread sources over the domain
        master = rng.uniform(0.0, 1.0, size=(dr["master_sensors"], 2))
        order = rng.permutation(dr["master_sensors"])   # fixed insertion order
        # geometry pair at N = 6
        grng = scenario_rng(root, SPLIT_TAGS["dose_geometry"], i)
        jitter = grng.uniform(size=(gN, 2))
        cells = np.array([(ix, iy) for iy in range(gy) for ix in range(gx)][:gN])
        spread = (cells + jitter) / np.array([gx, gy])       # stratified layout
        quad = grng.integers(0, 4)                           # random corner quadrant
        qoff = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])[quad]
        confined = qoff + 0.5 * grng.uniform(size=(gN, 2))
        base.append(dict(p=p, master=master, order=order, spread=spread,
                         confined=confined, quadrant=int(quad)))

    def make_set(tag, layouts, params, sigmas=None):
        """One reading-set per (scenario, layout); frozen noise per set."""
        B = len(layouts)
        sensors_pad = np.full((B, MAX_SENSORS, 2), np.nan)
        n_sensors = []
        for b, L in enumerate(layouts):
            sensors_pad[b, : len(L)] = L
            n_sensors.append(len(L))
        plist = []
        for b, p in enumerate(params):
            pp = dict(p)
            if sigmas is not None:
                pp["sigma"] = sigmas[b]
            pp["p_drop"] = dr["p_drop"]
            plist.append(pp)
        rngs = [scenario_rng(root, SPLIT_TAGS["dose_sigma"], hash_idx)
                for hash_idx in range(seq_start[0], seq_start[0] + B)]
        seq_start[0] += B
        readings, keep = _gen_readings(plist, sensors_pad, n_sensors, times,
                                       cfg, device, rngs)
        xs = np.array([p["xs"] for p in plist])
        return dict(ids=np.array([f"{tag}-{b:04d}" for b in range(B)]),
                    xs=xs,
                    q=np.array([p["q"] for p in plist]),
                    u=np.array([p["u"] for p in plist]),
                    D=np.array([p["D"] for p in plist]),
                    sigma=np.array([p["sigma"] for p in plist]),
                    p_drop=np.array([p["p_drop"] for p in plist]),
                    n_sensors=np.array(n_sensors, dtype=np.int64),
                    sensors=sensors_pad, readings=readings, keep=keep,
                    true_cell=pos_to_cell(xs, n_grid), times=times)

    seq_start = [0]     # distinct RNG stream index for every constructed set

    # Knob 1: readings on the FULL master layout, frozen once; nested subsets
    # N = 4..12 reuse rows of these readings (a controlled intervention).
    masters = make_set("dose1", [b["master"] for b in base],
                       [b["p"] for b in base])
    masters["order"] = np.array([b["order"] for b in base])

    # Knob 2: matched geometry pairs at N = 6.
    geo_spread = make_set("dose2s", [b["spread"] for b in base],
                          [b["p"] for b in base])
    geo_conf = make_set("dose2c", [b["confined"] for b in base],
                        [b["p"] for b in base])
    geo_conf["quadrant"] = np.array([b["quadrant"] for b in base])

    # Knob 3: sigma sweep on the first `sigma_sweep_sensors` master sensors
    # (in insertion order), readings regenerated per sigma.
    lo, hi = dr["sigma_sweep_range"]
    sig_values = np.exp(np.linspace(np.log(lo), np.log(hi),
                                    dr["sigma_sweep_points"]))
    k = dr["sigma_sweep_sensors"]
    sweep_layouts, sweep_params, sweep_sigmas, sweep_scen, sweep_sig_idx = [], [], [], [], []
    for b_i, b in enumerate(base):
        layout = b["master"][b["order"][:k]]
        for s_i, s in enumerate(sig_values):
            sweep_layouts.append(layout)
            sweep_params.append(b["p"])
            sweep_sigmas.append(s)
            sweep_scen.append(b_i)
            sweep_sig_idx.append(s_i)
    sweep = make_set("dose3", sweep_layouts, sweep_params, sigmas=sweep_sigmas)
    sweep["scenario_idx"] = np.array(sweep_scen)
    sweep["sigma_idx"] = np.array(sweep_sig_idx)
    sweep["sigma_values"] = sig_values

    return dict(dose1_master=masters, dose2_spread=geo_spread,
                dose2_confined=geo_conf, dose3_sweep=sweep)


# ------------------------------------------------------------ audit / main

def leakage_audit(data_dir, splits):
    """Zero ID overlap across splits; no duplicate scenario content."""
    id_sets, content = {}, {}
    for split in splits:
        d = np.load(data_dir / f"{split}.npz", allow_pickle=True)
        id_sets[split] = set(d["ids"].tolist())
        for i in range(len(d["ids"])):
            key = (d["xs"][i].tobytes(), d["sensors"][i].tobytes(),
                   d["readings"][i].tobytes())
            content.setdefault(key, []).append(d["ids"][i])
    report = {"id_overlaps": {}, "duplicate_content": [], "pass": True}
    names = list(splits)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ov = id_sets[names[a]] & id_sets[names[b]]
            report["id_overlaps"][f"{names[a]}|{names[b]}"] = len(ov)
            if ov:
                report["pass"] = False
    dups = [v for v in content.values() if len(v) > 1]
    report["duplicate_content"] = [list(map(str, v)) for v in dups]
    if dups:
        report["pass"] = False
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)

    checksums = {}
    all_splits = dict(cfg["splits"])
    all_splits.update(cfg.get("extra_splits", {}))
    for split, count in all_splits.items():
        path = data_dir / f"{split}.npz"
        if split in cfg.get("extra_splits", {}) and path.exists():
            print(f"{split} exists, skipping", flush=True)
            checksums[f"{split}.npz"] = sha256_file(path)
            continue
        print(f"generating {split} ({count} scenarios)...", flush=True)
        d = generate_split(split, count, cfg, device)
        np.savez_compressed(path, **d)
        checksums[f"{split}.npz"] = sha256_file(path)

    print("generating dose-response sets...", flush=True)
    dose = generate_dose_response(cfg, device)
    for name, d in dose.items():
        path = data_dir / f"{name}.npz"
        np.savez_compressed(path, **d)
        checksums[f"{name}.npz"] = sha256_file(path)

    results_dir = resolve(cfg, "results_dir")
    update_json(results_dir / "checksums.json", checksums)

    audit = leakage_audit(data_dir, list(cfg["splits"].keys())
                          + list(cfg.get("extra_splits", {}).keys()))
    with open(results_dir / "leakage_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    update_json(results_dir / "gates.json",
                {"G2": {"leakage_audit_pass": audit["pass"],
                        "leakage_id_overlaps": audit["id_overlaps"],
                        "leakage_duplicate_content": len(audit["duplicate_content"])}})
    print("leakage audit pass:", audit["pass"])
    total_mb = sum((data_dir / f).stat().st_size for f in checksums) / 1e6
    print(f"total stored: {total_mb:.1f} MB")
    assert audit["pass"], "LEAKAGE AUDIT FAILED"


if __name__ == "__main__":
    main()
