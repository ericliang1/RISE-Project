"""Batch-1 runtime of the full pipeline on real test scenarios.

Measures, per scenario after warm-up, with torch.cuda.synchronize around
each stage (host-device transfers included):
  map_ms  ensr preprocessing: K=8 wind draws, per-draw suffstats via
          cell_responses_t, moment + residual channel assembly
  net_ms  one forward pass of PhysHeadNet (batch size 1)
Writes results/runtime_bench.json with median and p95 of each stage and
the total.  Usage: python scripts/runtime_bench.py [--n 500]
"""
import argparse
import json
import time

import numpy as np
import torch

from common import cell_centers, get_device, load_config, resolve
from methane_t import N_GRID, cell_responses_t, stab_of, to_batch_t
from methane_t_uncertain import PhysHeadNet, perturb_wind, split_tag, \
    with_obs_wind
from paired_wind import K_MARG, R_MAX

N_CELLS = N_GRID * N_GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = ap.parse_args()
    cfg = load_config()
    device = torch.device("cpu") if args.device == "cpu" else get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    root = cfg["seeds"]["root_entropy"]
    d = dict(np.load(dd / "ch4t_test.npz", allow_pickle=True))
    u_obs = np.load(dd / "ch4tu_test_uobs.npy")
    dn = with_obs_wind(d, u_obs)
    tag = split_tag("test")
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    model = PhysHeadNet(cfg, 4).to(device).eval()

    def maps_one(i):
        """ensr-style channels for one scenario from raw inputs."""
        ns = int(d["n_sensors"][i])
        keep = torch.tensor(d["keep"][i, :ns], device=device)
        y = torch.tensor(d["readings"][i, :ns], device=device,
                         dtype=torch.float64)[keep]
        yy = float((y * y).sum())
        zs, Rs, lbs = [], [], []
        for k in range(K_MARG):
            uk = perturb_wind(u_obs[i],
                              np.random.default_rng([root, 93, tag, i, k]))
            g = cell_responses_t(cells, d["sensors"][i, :ns], uk,
                                 stab_of(d, i), device)[:, keep]
            a = g @ y
            b = (g * g).sum(1)
            zs.append(a / (float(d["sigma"][i]) * torch.sqrt(b) + 1e-30))
            qh = torch.clamp(a / (b + 1e-30), min=0.0)
            Rs.append(yy - 2 * qh * a + qh ** 2 * b)
            lbs.append(torch.log10(b + 1e-30))
        Z = torch.stack(zs)
        R = torch.stack(Rs).mean(0)
        R = R - R.min()
        R = torch.clamp(R / (R.median() + 1e-12), max=R_MAX) / R_MAX
        st = torch.stack([Z.mean(0), Z.std(0), torch.stack(lbs).mean(0),
                          R]).float()
        return st.view(1, 4, N_GRID, N_GRID)

    idx = np.arange(min(args.n + 20, len(d["ids"])))
    map_ms, net_ms = [], []
    for j, i in enumerate(idx):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        st = maps_one(int(i))
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.perf_counter()
        b = to_batch_t(dn, np.array([i]), device)
        b["stats"] = st
        with torch.no_grad():
            _ = torch.softmax(model(b).float(), -1)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t2 = time.perf_counter()
        if j >= 20:                      # warm-up excluded
            map_ms.append((t1 - t0) * 1e3)
            net_ms.append((t2 - t1) * 1e3)
    map_ms, net_ms = np.array(map_ms), np.array(net_ms)
    tot = map_ms + net_ms
    hw = (torch.cuda.get_device_name(0) if device.type == "cuda"
          else "CPU (single process)")
    out = {"n": len(map_ms), "hardware": hw,
           "map_ms_median": float(np.median(map_ms)),
           "map_ms_p95": float(np.percentile(map_ms, 95)),
           "net_ms_median": float(np.median(net_ms)),
           "net_ms_p95": float(np.percentile(net_ms, 95)),
           "total_ms_median": float(np.median(tot)),
           "total_ms_p95": float(np.percentile(tot, 95)),
           "note": "batch 1, K=8 ensr preprocessing, transfers included"}
    fn = "runtime_bench.json" if device.type == "cuda" else \
        "runtime_bench_cpu.json"
    with open(rr / fn, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
