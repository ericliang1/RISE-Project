"""G1 gate tests: quadrature convergence, zero-wind limit, symmetry, linearity.

Measured values are recorded in results/gates.json (they fill Appendix A
placeholders).
"""
import numpy as np
import torch

from common import REPO_ROOT, load_config, update_json
from forward_model import unit_response, zero_wind_exact
from priors import sample_params

CFG = load_config()
SIGMA_S = CFG["physics"]["sigma_s"]
TAU_MAX = CFG["physics"]["tau_max"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GATES = REPO_ROOT / "results" / "gates.json"


def _random_probes(n, seed):
    """Random (x, t, xs, u, D) probes drawn from the benchmark prior."""
    rng = np.random.default_rng(seed)
    xs_l, u_l, D_l = [], [], []
    for _ in range(n):
        p = sample_params(rng, CFG)
        xs_l.append(p["xs"]); u_l.append(p["u"]); D_l.append(p["D"])
    to = lambda a: torch.tensor(np.array(a), device=DEV, dtype=torch.float64)
    x = to(rng.uniform(0, 1, size=(n, 2)))
    t = to(rng.uniform(1.0 / 30.0, 1.0, size=n))
    return x, t, to(xs_l), to(u_l), to(D_l)


def test_quadrature_convergence():
    """Median relative 64- vs 128-node difference < 1e-6 over ~1000 probes."""
    x, t, xs, u, D = _random_probes(1000, seed=101)
    a64 = unit_response(x, t, xs, u, D, SIGMA_S, 64, TAU_MAX)
    a128 = unit_response(x, t, xs, u, D, SIGMA_S, 128, TAU_MAX)
    denom = torch.clamp(a128.abs(), min=1e-300)
    rel = ((a64 - a128).abs() / denom).cpu().numpy()
    med = float(np.median(rel))
    update_json(GATES, {"G1": {"quadrature_median_rel_err_64v128": med,
                               "quadrature_p90_rel_err_64v128": float(np.quantile(rel, 0.9))}})
    assert med < 1e-6, f"median rel err {med:.3e}"


def test_zero_wind_closed_form():
    """u = 0 reduces to the E1 closed form."""
    rng = np.random.default_rng(202)
    n = 500
    x = rng.uniform(0, 1, size=(n, 2))
    xs = rng.uniform(0.1, 0.9, size=(n, 2))
    t = rng.uniform(1.0 / 30.0, 1.0, size=n)
    D = np.exp(rng.uniform(np.log(0.002), np.log(0.02), size=n))
    to = lambda a: torch.tensor(a, device=DEV, dtype=torch.float64)
    a_quad = unit_response(to(x), to(t), to(xs), to(np.zeros((n, 2))), to(D),
                           SIGMA_S, 64, TAU_MAX).cpu().numpy()
    a_exact = zero_wind_exact(x, t, xs, D, SIGMA_S)
    keep = a_exact > 1e-290                     # closed form underflows for far pairs
    rel = np.abs(a_quad[keep] - a_exact[keep]) / a_exact[keep]
    med = float(np.median(rel))
    update_json(GATES, {"G1": {"zero_wind_median_rel_err": med,
                               "zero_wind_max_rel_err": float(rel.max())}})
    assert med < 1e-6, f"median rel err {med:.3e}"


def test_symmetry_rotation_reflection():
    """Rotating source + sensors + wind together leaves readings invariant."""
    x, t, xs, u, D = _random_probes(500, seed=303)
    base = unit_response(x, t, xs, u, D, SIGMA_S, 64, TAU_MAX)
    c = torch.tensor([0.5, 0.5], device=DEV, dtype=torch.float64)
    worst = 0.0
    for phi in [0.37, 1.9, np.pi / 2]:
        R = torch.tensor([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]], device=DEV, dtype=torch.float64)
        rot = lambda p: (p - c) @ R.T + c
        a = unit_response(rot(x), t, rot(xs), u @ R.T, D, SIGMA_S, 64, TAU_MAX)
        rel = ((a - base).abs() / torch.clamp(base.abs(), min=1e-300)).max()
        worst = max(worst, float(rel))
    # reflection about x = 0.5
    F = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], device=DEV, dtype=torch.float64)
    refl = lambda p: (p - c) @ F.T + c
    a = unit_response(refl(x), t, refl(xs), u @ F.T, D, SIGMA_S, 64, TAU_MAX)
    rel = ((a - base).abs() / torch.clamp(base.abs(), min=1e-300)).max()
    worst = max(worst, float(rel))
    update_json(GATES, {"G1": {"symmetry_max_rel_err": worst}})
    assert worst < 1e-10, f"max rel err {worst:.3e}"


def test_linearity():
    """q1*A + q2*A == (q1+q2)*A to floating-point rounding."""
    x, t, xs, u, D = _random_probes(200, seed=404)
    a = unit_response(x, t, xs, u, D, SIGMA_S, 64, TAU_MAX)
    q1, q2 = 0.7, 3.1
    lhs = q1 * a + q2 * a
    rhs = (q1 + q2) * a
    rel = ((lhs - rhs).abs() / torch.clamp(rhs.abs(), min=1e-300)).max()
    update_json(GATES, {"G1": {"linearity_max_rel_err": float(rel)}})
    assert float(rel) < 1e-12
