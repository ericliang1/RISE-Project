"""Stage 4: randomized highest-density-mass split conformal (paper Sec. 4.2).

Paper score for calibration scenario j with heatmap P_j and true cell c*:
    s_j = sum_{c: P_j(c) > P_j(c*)} P_j(c) + U_j * sum_{c: P_j(c) = P_j(c*)} P_j(c)
threshold qhat = ceil((n+1)(1-alpha))-th smallest score; test region = cells in
descending-probability order until cumulative mass >= qhat.

NUMERICAL IMPLEMENTATION (documented; exact-arithmetic-equivalent): sharp
heatmaps put the true cell's exceedance mass at 1 - epsilon with epsilon down
to ~1e-30; in floating point  s = mass_above  catastrophically rounds to
exactly 1.0, creating an atom of tied scores at the top whose quantile breaks
coverage (observed: conformal-on-exact-posterior at 0.84 instead of 0.90).
We therefore work throughout with the COMPLEMENT ("tail mass")

    t_j = sum_{c: P_j(c) < P_j(c*)} P_j(c) + (1-U_j) * (tied mass incl. c*),

which satisfies s_j = total_j - t_j exactly and is representable at both ends
(t ~ 1e-300 is fine).  The k-th smallest s is the k-th largest t; the region
"smallest top-set with cumulative mass >= qhat" is identically "exclude the
largest bottom-set whose mass is <= that", computed by ascending cumsum
(small numbers summed first - no cancellation).  All coverage semantics are
unchanged in exact arithmetic.
"""
import numpy as np
from scipy.stats import beta


def tail_scores(P, true_cells, rng):
    """Complement nonconformity scores t_j (see header).  P (S, C) float64."""
    S = P.shape[0]
    p_true = P[np.arange(S), true_cells]
    below = P * (P < p_true[:, None])
    tied = P * (P == p_true[:, None])
    mass_below = below.sum(axis=1)
    mass_tied = tied.sum(axis=1)
    return mass_below + (1.0 - rng.uniform(size=S)) * mass_tied


def tail_threshold(t, alpha):
    """Tail-space threshold: the ceil((n+1)(1-alpha))-th smallest paper score
    is the same-k largest tail score.  Returns t_hat (>= 0)."""
    n = len(t)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 0.0        # paper qhat = 1 (full region): exclude nothing
    return float(np.sort(t)[n - k])


def regions(P, t_hat, true_cells=None):
    """Per scenario: exclude the largest ascending-probability prefix whose
    mass is <= t_hat; the region is everything else (= smallest top-set whose
    cumulative mass reaches total - t_hat).  Returns sizes and coverage."""
    S, C = P.shape
    order = np.argsort(P, axis=1, kind="stable")             # ascending
    Psort = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Psort, axis=1)
    n_excl = (cum <= t_hat).sum(axis=1)                      # excluded tail cells
    sizes = C - n_excl
    out = {"sizes": sizes}
    if true_cells is not None:
        ranks = np.empty_like(order)
        ranks[np.arange(S)[:, None], order] = np.arange(C)[None, :]
        pos = ranks[np.arange(S), true_cells]                # ascending position
        out["covered"] = pos >= n_excl
    return out


def region_mask(P_row, t_hat):
    """Boolean membership mask of the conformal region for one heatmap row."""
    order = np.argsort(P_row, kind="stable")
    cum = np.cumsum(P_row[order])
    n_excl = int((cum <= t_hat).sum())
    mask = np.ones(len(P_row), dtype=bool)
    mask[order[:n_excl]] = False
    return mask


def clopper_pearson(k, n, conf=0.95):
    """Exact CI for a binomial proportion."""
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def coverage_report(P_test, true_cells_test, tails_calib, alpha):
    """Coverage + region-size stats at one alpha."""
    t_hat = tail_threshold(tails_calib, alpha)
    reg = regions(P_test, t_hat, true_cells_test)
    n = len(true_cells_test)
    k = int(reg["covered"].sum())
    lo, hi = clopper_pearson(k, n)
    frac = reg["sizes"] / P_test.shape[1]
    return {
        "alpha": alpha, "tail_qhat": t_hat, "coverage": k / n,
        "cp_ci": [lo, hi], "n": n, "covered": k,
        "area_mean": float(frac.mean()), "area_median": float(np.median(frac)),
        "sizes": reg["sizes"],
    }


def nominal_sweep(P_test, true_cells_test, tails_calib, levels):
    """Raw-HPD vs conformal empirical coverage across nominal levels (fig3)."""
    rows = []
    row_tail_total = None
    for lev in levels:
        t_hat = tail_threshold(tails_calib, 1.0 - lev)
        cov_c = regions(P_test, t_hat, true_cells_test)["covered"].mean()
        # raw HPD at nominal mass `lev` = exclude tail of mass total - lev;
        # computed per-row against each row's own total to stay exact.
        if row_tail_total is None:
            row_tail_total = P_test.sum(axis=1)
        # per-row tail threshold: total - lev (>=0)
        S, C = P_test.shape
        order = np.argsort(P_test, axis=1, kind="stable")
        Psort = np.take_along_axis(P_test, order, axis=1)
        cum = np.cumsum(Psort, axis=1)
        thr = np.maximum(row_tail_total - lev, 0.0)[:, None]
        n_excl = (cum <= thr).sum(axis=1)
        ranks = np.empty_like(order)
        ranks[np.arange(S)[:, None], order] = np.arange(C)[None, :]
        pos = ranks[np.arange(S), true_cells_test]
        cov_r = (pos >= n_excl).mean()
        rows.append({"nominal": float(lev), "conformal": float(cov_c),
                     "raw_hpd": float(cov_r)})
    return rows
