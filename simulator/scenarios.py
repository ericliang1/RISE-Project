import copy

import numpy as np
import torch

from common import pos_to_cell
from plume import U_SCALE, plume_ppm_per_kgh

N_GRID = 64
T = 30
SPLITS = {"train": 10000, "val": 1000, "calib": 2000, "test": 2000}
TAG_BASE = 50
Q_LO, Q_HI = 100.0, 500.0
SIG_TH = np.deg2rad(10.0)
SIG_S = 0.10


def ch4_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["prior"]["q_log_low"] = Q_LO
    c["prior"]["q_log_high"] = Q_HI
    return c


def wind_sequence(rng):
    U0 = np.exp(rng.uniform(np.log(1.0), np.log(8.0)))
    th0 = rng.uniform(0, 2 * np.pi)
    rho = 0.7
    amp_dir = np.deg2rad(rng.uniform(20.0, 40.0))
    amp_spd = rng.uniform(0.10, 0.30)
    ed = es = 0.0
    seq = np.zeros((T, 2))
    for k in range(T):
        ed = rho * ed + np.sqrt(1 - rho ** 2) * rng.normal()
        es = rho * es + np.sqrt(1 - rho ** 2) * rng.normal()
        th = th0 + amp_dir * ed
        U = U0 * np.exp(amp_spd * es)
        seq[k] = [U * np.cos(th), U * np.sin(th)]
    mean_u = seq.mean(0) / U_SCALE
    return seq / U_SCALE, mean_u


def sensor_responses_t(sensors, xs, u_seq, stab, device):
    S = sensors.shape[0]
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)[:, None, :].expand(S, T, 2)
    xx = torch.as_tensor(xs, dtype=torch.float64, device=device)[None, None, :].expand(S, T, 2)
    uu = torch.as_tensor(u_seq, dtype=torch.float64, device=device)[None, :, :].expand(S, T, 2)
    st = torch.full((S, T), int(stab), dtype=torch.long, device=device)
    return plume_ppm_per_kgh(ss, xx, uu, st)


def cell_responses_t(cells, sensors, u_seq, stab, device, chunk=512):
    C, S = cells.shape[0], sensors.shape[0]
    out = torch.empty((C, S, T), device=device, dtype=torch.float64)
    uu = torch.as_tensor(u_seq, dtype=torch.float64, device=device)
    ss = torch.as_tensor(sensors, dtype=torch.float64, device=device)
    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        c = hi - lo
        cc = cells[lo:hi][:, None, None, :].expand(c, S, T, 2)
        s2 = ss[None, :, None, :].expand(c, S, T, 2)
        u2 = uu[None, None, :, :].expand(c, S, T, 2)
        st = torch.full((c, S, T), int(stab), dtype=torch.long, device=device)
        out[lo:hi] = plume_ppm_per_kgh(s2, cc, u2, st)
    return out


def gen_split(name, count, cfg, root):
    tag = TAG_BASE + list(SPLITS).index(name)
    rows = {k: [] for k in ["ids", "xs", "q", "u_seq", "u_mean", "D", "sigma",
                            "n_sensors", "sensors", "readings", "keep"]}
    times = np.arange(1, T + 1) / T
    for i in range(count):
        rng = np.random.default_rng([root, tag, i])
        xs = rng.uniform(0.1, 0.9, 2)
        cell = pos_to_cell(xs[None], N_GRID)[0]
        xs = np.array([(cell % N_GRID + 0.5) / N_GRID,
                       (cell // N_GRID + 0.5) / N_GRID])
        q = np.exp(rng.uniform(np.log(100.0), np.log(500.0)))
        u_seq, u_mean = wind_sequence(rng)
        stab = int(rng.integers(0, 6))
        sig = np.exp(rng.uniform(np.log(0.2), np.log(3.0)))
        pd = rng.uniform(0.0, 0.2)
        n = int(rng.integers(4, 9))
        sensors = rng.uniform(0, 1, (n, 2))
        pad = np.zeros((12, 2)); pad[:n] = sensors
        g = sensor_responses_t(sensors, xs, u_seq, stab, "cpu").numpy()
        readings = np.zeros((12, T), np.float32)
        keep = np.zeros((12, T), bool)
        readings[:n] = (q * g + rng.normal(0, sig, (n, T))).astype(np.float32)
        keep[:n] = rng.uniform(size=(n, T)) >= pd
        if not keep.any():
            keep[0, 0] = True
        rows["ids"].append(f"ch4t-{name}-{i:06d}")
        rows["xs"].append(xs); rows["q"].append(q)
        rows["u_seq"].append(u_seq); rows["u_mean"].append(u_mean)
        rows["D"].append(np.exp(stab / 5.0)); rows["sigma"].append(sig)
        rows["n_sensors"].append(n); rows["sensors"].append(pad)
        rows["readings"].append(readings); rows["keep"].append(keep)
    d = {k: np.array(v) for k, v in rows.items()}
    d["times"] = times
    d["true_cell"] = pos_to_cell(d["xs"], N_GRID)
    return d


def stab_of(d, i):
    return int(round(np.log(d["D"][i]) * 5.0))


def split_tag(name):
    return 60 + list(SPLITS).index(name)


def perturb_wind(u_seq, rng):
    th = np.arctan2(u_seq[:, 1], u_seq[:, 0]) + rng.normal(0, SIG_TH, T)
    sp = np.linalg.norm(u_seq, axis=1) * np.exp(rng.normal(0, SIG_S, T))
    return np.stack([sp * np.cos(th), sp * np.sin(th)], 1)


def obs_wind_split(d, name, root):
    out = np.empty_like(d["u_seq"])
    tag = split_tag(name)
    for i in range(len(d["ids"])):
        rng = np.random.default_rng([root, 90, tag, i])
        out[i] = perturb_wind(d["u_seq"][i], rng)
    return out


def with_obs_wind(d, u_obs):
    d2 = dict(d)
    d2["u_seq"] = u_obs
    d2["u_mean"] = u_obs.mean(1)
    return d2
