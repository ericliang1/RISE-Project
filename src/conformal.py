"""Stage 4: randomized highest-density-mass split conformal (paper Sec. 4.2).

Score for calibration scenario j with heatmap P_j and true cell c*:
    s_j = sum_{c: P_j(c) > P_j(c*)} P_j(c)  +  U_j * sum_{c: P_j(c) = P_j(c*)} P_j(c)
with U_j ~ Uniform(0,1).  Threshold qhat = ceil((n+1)(1-alpha))-th smallest
score.  Test region: cells in descending-probability order until cumulative
mass >= qhat.  Region size in cells; area = size / n_cells.
"""
import numpy as np
from scipy.stats import beta


def nonconformity_scores(P, true_cells, rng):
    """P (S, C) heatmaps, true_cells (S,) -> randomized HDM scores (S,)."""
    S = P.shape[0]
    p_true = P[np.arange(S), true_cells]                     # (S,)
    above = (P > p_true[:, None]).astype(np.float64)
    tied = (P == p_true[:, None]).astype(np.float64)
    mass_above = (P * above).sum(axis=1)
    mass_tied = (P * tied).sum(axis=1)
    return mass_above + rng.uniform(size=S) * mass_tied


def conformal_threshold(scores, alpha):
    """ceil((n+1)(1-alpha))-th smallest score (1-indexed); capped at 1.0."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


def regions(P, qhat, true_cells=None):
    """Smallest top-ranked set with cumulative mass >= qhat, per scenario.

    Returns dict with sizes (S,), covered (S,) if true_cells given, and
    rank (S, C) = descending-probability order (for region membership tests).
    """
    order = np.argsort(-P, axis=1, kind="stable")            # (S, C)
    Psort = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Psort, axis=1)
    # first index where cum >= qhat (region always includes >= 1 cell)
    sizes = 1 + (cum < qhat).sum(axis=1)
    sizes = np.minimum(sizes, P.shape[1])
    out = {"sizes": sizes}
    if true_cells is not None:
        ranks = np.empty_like(order)
        S, C = P.shape
        ranks[np.arange(S)[:, None], order] = np.arange(C)[None, :]
        out["covered"] = ranks[np.arange(S), true_cells] < sizes
    return out


def region_mask(P_row, qhat):
    """Boolean membership mask of the conformal region for one heatmap."""
    order = np.argsort(-P_row, kind="stable")
    cum = np.cumsum(P_row[order])
    size = int(1 + (cum < qhat).sum())
    size = min(size, len(P_row))
    mask = np.zeros(len(P_row), dtype=bool)
    mask[order[:size]] = True
    return mask


def clopper_pearson(k, n, conf=0.95):
    """Exact CI for a binomial proportion."""
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def coverage_report(P_test, true_cells_test, scores_calib, alpha):
    """Coverage + region-size stats at one alpha."""
    qhat = conformal_threshold(scores_calib, alpha)
    reg = regions(P_test, qhat, true_cells_test)
    n = len(true_cells_test)
    k = int(reg["covered"].sum())
    lo, hi = clopper_pearson(k, n)
    frac = reg["sizes"] / P_test.shape[1]
    return {
        "alpha": alpha, "qhat": qhat, "coverage": k / n,
        "cp_ci": [lo, hi], "n": n, "covered": k,
        "area_mean": float(frac.mean()), "area_median": float(np.median(frac)),
        "sizes": reg["sizes"],
    }


def nominal_sweep(P_test, true_cells_test, scores_calib, levels):
    """Raw-HPD vs conformal empirical coverage across nominal levels (fig3)."""
    rows = []
    for lev in levels:
        # conformal at alpha = 1 - lev
        qhat = conformal_threshold(scores_calib, 1.0 - lev)
        cov_c = regions(P_test, qhat, true_cells_test)["covered"].mean()
        # raw HPD: threshold = nominal mass itself (uncalibrated)
        cov_r = regions(P_test, lev, true_cells_test)["covered"].mean()
        rows.append({"nominal": float(lev), "conformal": float(cov_c),
                     "raw_hpd": float(cov_r)})
    return rows
