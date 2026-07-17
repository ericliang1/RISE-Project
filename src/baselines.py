"""Peak-sensor heuristic baseline: point estimate at the sensor with the
largest time-integrated concentration.  With dropout, the integral is
estimated as the mean of retained readings (fair across dropout rates;
documented choice)."""
import numpy as np


def peak_sensor_estimates(d):
    """d: split dict -> (S, 2) estimated source positions."""
    readings = d["readings"].astype(np.float64)          # (S, 12, T)
    keep = d["keep"]
    cnt = keep.sum(axis=2)                               # (S, 12)
    mean_conc = np.where(cnt > 0, (readings * keep).sum(axis=2) / np.maximum(cnt, 1),
                         -np.inf)                        # padded/empty sensors: -inf
    best = mean_conc.argmax(axis=1)                      # (S,)
    return d["sensors"][np.arange(len(best)), best]


def localization_errors(est_pos, true_pos):
    return np.linalg.norm(est_pos - true_pos, axis=-1)
