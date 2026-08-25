import numpy as np


def tail_scores(P, true_cells, rng):
    S = P.shape[0]
    p_true = P[np.arange(S), true_cells]
    below = P * (P < p_true[:, None])
    tied = P * (P == p_true[:, None])
    mass_below = below.sum(axis=1)
    mass_tied = tied.sum(axis=1)
    return mass_below + (1.0 - rng.uniform(size=S)) * mass_tied


def tail_threshold(t, alpha):
    n = len(t)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 0.0
    return float(np.sort(t)[n - k])


def regions(P, t_hat, true_cells=None):
    S, C = P.shape
    order = np.argsort(P, axis=1, kind="stable")
    Psort = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Psort, axis=1)
    n_excl = (cum < t_hat).sum(axis=1)
    sizes = C - n_excl
    out = {"sizes": sizes}
    if true_cells is not None:
        ranks = np.empty_like(order)
        ranks[np.arange(S)[:, None], order] = np.arange(C)[None, :]
        pos = ranks[np.arange(S), true_cells]
        out["covered"] = pos >= n_excl
    return out


def region_mask(P_row, t_hat):
    order = np.argsort(P_row, kind="stable")
    cum = np.cumsum(P_row[order])
    n_excl = int((cum < t_hat).sum())
    mask = np.ones(len(P_row), dtype=bool)
    mask[order[:n_excl]] = False
    return mask
