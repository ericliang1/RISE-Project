"""Benchmark data-generation prior (paper Table 2). Shared by tests and datagen."""
import numpy as np


def sample_params(rng, cfg):
    """Sample one scenario's physical parameters (no sensors/readings) from the prior."""
    p = cfg["prior"]
    xs = rng.uniform(p["source_low"], p["source_high"], size=2)
    q = np.exp(rng.uniform(np.log(p["q_log_low"]), np.log(p["q_log_high"])))
    speed = rng.uniform(p["wind_speed_low"], p["wind_speed_high"])
    theta = rng.uniform(0.0, 2.0 * np.pi)
    u = np.array([speed * np.cos(theta), speed * np.sin(theta)])
    D = np.exp(rng.uniform(np.log(p["diff_log_low"]), np.log(p["diff_log_high"])))
    sigma = np.exp(rng.uniform(np.log(p["sigma_log_low"]), np.log(p["sigma_log_high"])))
    p_drop = rng.uniform(p["p_drop_low"], p["p_drop_high"])
    n_sensors = int(rng.integers(p["n_sensors_low"], p["n_sensors_high"] + 1))
    return dict(xs=xs, q=q, u=u, D=D, sigma=sigma, p_drop=p_drop, n_sensors=n_sensors)


def obs_times(cfg):
    T = cfg["prior"]["n_times"]
    return np.arange(1, T + 1, dtype=np.float64) / T
