import pathlib
import sys

import numpy as np
import torch

_R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_R / "simulator"), str(_R / "localization")]

from common import cell_centers, get_device, load_config, resolve, savez_atomic
from scenarios import (N_GRID, SPLITS, cell_responses_t, obs_wind_split,
                       perturb_wind, split_tag, stab_of)

N_CELLS = N_GRID * N_GRID
K_MAPS = 8


@torch.no_grad()
def ab_maps(d, u_seqs, cells, device):
    n = len(d["ids"])
    A = np.empty((n, N_CELLS), np.float64)
    B = np.empty((n, N_CELLS), np.float64)
    for i in range(n):
        ns = int(d["n_sensors"][i])
        keep = d["keep"][i, :ns]
        g = cell_responses_t(cells, d["sensors"][i, :ns], u_seqs[i],
                             stab_of(d, i), device)
        g = g[:, torch.tensor(keep, device=device)]
        y = torch.tensor(d["readings"][i, :ns][keep], device=device,
                         dtype=torch.float64)
        A[i] = (g @ y).cpu().numpy()
        B[i] = (g * g).sum(1).cpu().numpy()
    return A, B


def zmap(a, b, sig):
    z = a / (sig[:, None] * np.sqrt(b) + 1e-30)
    return np.arcsinh(z / 10.0)


def logbmap(b):
    return (np.log10(b + 1e-30) + 8.0) / 6.0


def _gen_maps(d, u_obs, cells, device, dd, name, tag, root, n):
    zs = np.empty((K_MAPS, n, N_CELLS), np.float32)
    lb = np.empty((K_MAPS, n, N_CELLS), np.float32)
    for j in range(K_MAPS):
        uj = np.stack([perturb_wind(u_obs[i],
                                    np.random.default_rng(
                                        [root, 92, tag, i, j]))
                       for i in range(n)])
        Aj, Bj = ab_maps(d, uj, cells, device)
        zs[j] = zmap(Aj, Bj, d["sigma"])
        lb[j] = logbmap(Bj)
    ens = np.stack([zs.mean(0), zs.std(0) * 3.0, lb.mean(0)], 1)
    savez_atomic(dd / f"ch4tu_{name}_maps_ens.npz",
                 maps=ens.astype(np.float32))
    print(f"  {name}: maps done", flush=True)


@torch.no_grad()
def stage_gen(cfg, device):
    dd = resolve(cfg, "data_dir")
    root = cfg["seeds"]["root_entropy"]
    cells = torch.tensor(cell_centers(N_GRID), device=device,
                         dtype=torch.float64)
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        n = len(d["ids"])
        tag = split_tag(name)
        u_obs = obs_wind_split(d, name, root)
        np.save(dd / f"ch4tu_{name}_uobs.npy", u_obs)

        if (dd / f"ch4tu_{name}_maps_ens.npz").exists():
            print(f"  {name}: maps exist, skipping", flush=True)
        else:
            _gen_maps(d, u_obs, cells, device, dd, name, tag, root, n)


if __name__ == "__main__":
    stage_gen(load_config(), get_device())
