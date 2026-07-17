"""G1 slow gate: finite-difference cross-check of the analytic forward model.

Metric (documented in GATES.md): per-scenario relative L2 over all
(sensor, time) readings when the scenario has non-negligible signal
(reference RMS > 1e-6 * C_ref); scenarios whose sensors see essentially no
plume (all readings orders of magnitude below the smallest noise std 0.01)
are gated on ABSOLUTE RMS error < 1e-4 * C_ref instead — the relative metric
is ill-conditioned at zero signal, while 1e-4 is 1% of the smallest noise std.

Run with:  pytest tests/test_fd_check.py -m slow -s
"""
import numpy as np
import pytest
import torch

from common import REPO_ROOT, load_config, update_json
from fd_solver import analytic_sensor_readings, fd_sensor_readings, rel_l2
from priors import obs_times, sample_params

CFG = load_config()
FD = CFG["fd_check"]
PH = CFG["physics"]
GATES = REPO_ROOT / "results" / "gates.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

RMS_SIGNAL_FLOOR = 1e-6      # below this reference RMS -> absolute gate
ABS_TOL = 1e-4               # 1% of the smallest noise std (sigma_min = 0.01)

pytestmark = pytest.mark.slow


def _scenario(rng):
    p = sample_params(rng, CFG)
    sensors = rng.uniform(0, 1, size=(p["n_sensors"], 2))
    return p, sensors


def _ref(p, sensors, times):
    return analytic_sensor_readings(p["xs"], p["u"], p["D"], p["q"],
                                    PH["sigma_s"], sensors, times,
                                    n_nodes=PH["quad_nodes"],
                                    tau_max=PH["tau_max"], device=DEV)


@pytest.mark.slow
def test_fd_self_convergence():
    """FD error at sensors shrinks monotonically across h = 1/256, 1/384, 1/512."""
    rng = np.random.default_rng(11)
    times = obs_times(CFG)
    rows = []
    for _ in range(FD["n_convergence_scenarios"]):
        p, sensors = _scenario(rng)
        ref = _ref(p, sensors, times)
        errs = []
        for h in FD["convergence_h"]:
            fd = fd_sensor_readings(p["xs"], p["u"], p["D"], p["q"], PH["sigma_s"],
                                    sensors, times, h, FD["cfl_safety"],
                                    FD["margin"], device=DEV)
            errs.append(rel_l2(fd, ref))
        rows.append(errs)
        print("self-convergence rel L2 by h:", errs, flush=True)
    rows = np.array(rows)
    update_json(GATES, {"G1": {"fd_self_convergence_relL2_by_h":
                               {str(h): list(map(float, rows[:, i]))
                                for i, h in enumerate(FD["convergence_h"])}}})
    assert np.all(np.diff(rows, axis=1) < 0), rows


@pytest.mark.slow
def test_fd_cross_check_20_scenarios():
    """20 random scenarios agree with the analytic model (< 1% rel L2, or
    absolute agreement far below the noise floor when there is no signal)."""
    rng = np.random.default_rng(42)
    times = obs_times(CFG)
    rel_errs, abs_only = [], []
    for i in range(FD["n_scenarios"]):
        p, sensors = _scenario(rng)
        ref = _ref(p, sensors, times)
        fd = fd_sensor_readings(p["xs"], p["u"], p["D"], p["q"], PH["sigma_s"],
                                sensors, times, FD["h"], FD["cfl_safety"],
                                FD["margin"], device=DEV)
        rms = float(np.sqrt((ref ** 2).mean()))
        if rms > RMS_SIGNAL_FLOOR:
            e = rel_l2(fd, ref)
            rel_errs.append(e)
            print(f"scenario {i:2d}: rms={rms:.3e} relL2={e:.4%}", flush=True)
        else:
            e = float(np.sqrt(((fd - ref) ** 2).mean()))
            abs_only.append(e)
            print(f"scenario {i:2d}: rms={rms:.3e} (no signal) absRMS={e:.3e}",
                  flush=True)
    rel_errs = np.array(rel_errs)
    update_json(GATES, {"G1": {
        "fd_cross_check_relL2_max": float(rel_errs.max()),
        "fd_cross_check_relL2_median": float(np.median(rel_errs)),
        "fd_cross_check_relL2_all": list(map(float, rel_errs)),
        "fd_cross_check_n_signal": int(len(rel_errs)),
        "fd_cross_check_absRMS_no_signal": list(map(float, abs_only)),
        "fd_cross_check_h": FD["h"]}})
    assert rel_errs.max() < FD["tol_rel_l2"], f"max rel L2 {rel_errs.max():.4%}"
    assert all(e < ABS_TOL for e in abs_only), abs_only
