"""CH4-500 completion campaign 2: geometry & noise knobs, memorization
ablation, proper mode-recovery sanity, full audit metrics, stratified +
significance analysis, coverage sweep.

Outputs: results/ch4_extras2.json, results/ch4_memorization.csv,
data/ch4_dose23.npz.  Requires methane_extras.py to have run (checkpoints).

Usage:  python src/methane_extras2.py
"""
import csv
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from common import (cell_center_of, get_device, load_config, pos_to_cell,
                    resolve, update_json)
from conformal import region_mask, regions, tail_scores, tail_threshold
from inference import to_batch
from methane_model import U_SCALE, plume_ppm_per_kgh
from methane_pipeline import (SPLITS, ch4_cfg, ch4_posteriors, model_probs,
                              stab_of)
from methane_extras import load_ch4
from model import DeepSetsLocalizer, smoothed_targets
from train import d4_augment, val_nll

N_GRID = 64
T = 30


def boot_ci_median(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    return [float(np.median(x))] + [float(v) for v in
                                    np.quantile(meds, [0.025, 0.975])]


def base_scenario(b):
    srng = np.random.default_rng(700000 + b)
    xs = srng.uniform(0.15, 0.85, 2)
    q = np.exp(srng.uniform(np.log(10.0), np.log(500.0)))
    U = np.exp(srng.uniform(np.log(1.0), np.log(8.0)))
    th_w = srng.uniform(0, 2 * np.pi)
    u_norm = np.array([U * np.cos(th_w), U * np.sin(th_w)]) / U_SCALE
    stab = int(srng.integers(0, 6))
    sig = np.exp(srng.uniform(np.log(0.2), np.log(3.0)))
    master = srng.uniform(0, 1, (12, 2))
    return srng, xs, q, u_norm, stab, sig, master


def one_scenario(xs, q, u_norm, stab, sig, sensors, rng):
    n = len(sensors)
    g = plume_ppm_per_kgh(torch.tensor(sensors), torch.tensor(xs)[None],
                          torch.tensor(u_norm)[None],
                          torch.tensor([stab])).numpy()
    pad_s = np.zeros((12, 2)); pad_s[:n] = sensors
    pad_r = np.zeros((12, T), np.float32)
    pad_r[:n] = (q * g[:, None] + rng.normal(0, sig, (n, T))).astype(np.float32)
    keep = np.zeros((12, T), bool); keep[:n] = True
    return {"ids": np.array(["x"]), "sensors": pad_s[None],
            "readings": pad_r[None], "keep": keep[None],
            "times": np.arange(1, T + 1) / T, "u": u_norm[None],
            "D": np.array([np.exp(stab / 5.0)]), "sigma": np.array([sig]),
            "n_sensors": np.array([n]), "xs": xs[None], "q": np.array([q]),
            "true_cell": pos_to_cell(xs[None], N_GRID)}, g


def main():
    cfg = load_config()
    cfg_q = ch4_cfg(cfg)
    device = get_device()
    data_dir = resolve(cfg, "data_dir")
    results_dir = resolve(cfg, "results_dir")
    ck_dir = resolve(cfg, "checkpoints_dir")
    alpha = cfg["conformal"]["alpha"]

    data = {n: load_ch4(data_dir, n) for n in SPLITS}
    P_ex = {n: np.load(data_dir / f"ch4_{n}_posterior.npz")["probs"]
            for n in ("calib", "test")}
    rng = np.random.default_rng(cfg["conformal"]["score_seed"])
    tails_e = tail_scores(P_ex["calib"], data["calib"]["true_cell"], rng)
    th_e = tail_threshold(tails_e, alpha)
    reg_e = regions(P_ex["test"], th_e, data["test"]["true_cell"])
    models = {}
    for mode in ("M0", "M1"):
        m = DeepSetsLocalizer(cfg).to(device)
        m.load_state_dict(torch.load(ck_dir / f"ch4_{mode.lower()}_s1.pt",
                                     weights_only=False))
        m.eval()
        models[mode] = m
    out = {}

    # ---------------- audit metrics + coverage sweep -----------------------
    print("== audit metrics ==", flush=True)
    P_t = {m: model_probs(models[m], data["test"], device) for m in models}
    P_c = {m: model_probs(models[m], data["calib"], device) for m in models}
    tails = {m: tail_scores(P_c[m], data["calib"]["true_cell"],
                            np.random.default_rng(cfg["conformal"]["score_seed"]))
             for m in models}
    d_t = data["test"]
    xs_t = d_t["xs"]
    # peak-sensor baseline
    w = (d_t["readings"] * d_t["keep"]).sum(2) / np.maximum(
        d_t["keep"].sum(2), 1)
    w[~(d_t["keep"].any(2))] = -np.inf
    ps_est = d_t["sensors"][np.arange(len(w)), w.argmax(1)]
    audit = {"peak_sensor_map_median":
             float(np.median(np.linalg.norm(ps_est - xs_t, axis=1)))}
    e_map = cell_center_of(P_ex["test"].argmax(1), N_GRID)
    audit["exact_map_median"] = float(np.median(
        np.linalg.norm(e_map - xs_t, axis=1)))
    sweep = []
    for m in models:
        est = cell_center_of(P_t[m].argmax(1), N_GRID)
        err = np.linalg.norm(est - xs_t, axis=1)
        eps = 1e-300
        Pn = P_t[m] + eps; Pn /= Pn.sum(1, keepdims=True)
        Qn = P_ex["test"] + eps; Qn /= Qn.sum(1, keepdims=True)
        Mx = 0.5 * (Pn + Qn)
        jsd = 0.5 * (Pn * (np.log(Pn) - np.log(Mx))).sum(1) \
            + 0.5 * (Qn * (np.log(Qn) - np.log(Mx))).sum(1)
        tc = d_t["true_cell"]
        audit[m] = {
            "map_median": float(np.median(err)), "map_mean": float(err.mean()),
            "map_p50_90_99": [float(np.percentile(err, p))
                              for p in (50, 90, 99)],
            "jsd_mean": float(jsd.mean()),
            "nll_mean": float(-np.log(np.maximum(
                P_t[m][np.arange(len(tc)), tc], 1e-300)).mean()),
        }
        for lev in (0.80, 0.90, 0.95):
            th = tail_threshold(tails[m], 1.0 - lev)
            reg = regions(P_t[m], th, tc)
            audit[m][f"coverage_{lev:.2f}"] = float(reg["covered"].mean())
            audit[m][f"area_median_{lev:.2f}"] = float(
                np.median(reg["sizes"] / 4096))
        # coverage-vs-nominal sweep (raw HPD vs conformal) for fig3
        for lev in np.linspace(0.50, 0.98, 25):
            th = tail_threshold(tails[m], 1.0 - lev)
            cov_c = regions(P_t[m], th, tc)["covered"].mean()
            tot = P_t[m].sum(1)
            covr = []
            order = np.argsort(P_t[m], axis=1)
            Ps = np.take_along_axis(P_t[m], order, axis=1)
            cum = np.cumsum(Ps, axis=1)
            thr = np.maximum(tot - lev, 0.0)[:, None]
            n_excl = (cum <= thr).sum(1)
            ranks = np.empty_like(order)
            ranks[np.arange(len(tc))[:, None], order] = np.arange(4096)[None]
            cov_r = (ranks[np.arange(len(tc)), tc] >= n_excl).mean()
            sweep.append({"model": m, "nominal": float(lev),
                          "conformal": float(cov_c), "raw_hpd": float(cov_r)})
        # latency
        with torch.no_grad():
            for i in range(10):
                models[m](to_batch(d_t, np.array([i]), device))
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(200):
                models[m](to_batch(d_t, np.array([i]), device))
            torch.cuda.synchronize()
        audit[m]["latency_ms"] = (time.time() - t0) / 200 * 1000
    out["audit"] = audit
    out["coverage_sweep"] = sweep

    # ---------------- stratified + significance ----------------------------
    print("== stratified ==", flush=True)
    e_area = reg_e["sizes"] / 4096
    r = {m: np.load(data_dir / f"ch4_audit_{m}.npz")["sizes"]
         / np.maximum(reg_e["sizes"], 1) for m in models}
    cov = {m: regions(P_t[m], tail_threshold(tails[m], alpha),
                      d_t["true_cell"])["covered"] for m in models}
    strata = {"identifiable (<10%)": e_area < 0.10,
              "mid (10-50%)": (e_area >= 0.10) & (e_area < 0.5),
              "weak (>=50%)": e_area >= 0.5,
              "N 4-6": d_t["n_sensors"] <= 6,
              "N 7-9": (d_t["n_sensors"] >= 7) & (d_t["n_sensors"] <= 9),
              "N 10-12": d_t["n_sensors"] >= 10}
    tab = {}
    for name, msk in strata.items():
        tab[name] = {"n": int(msk.sum())}
        for m in models:
            tab[name][f"cov_{m}"] = float(cov[m][msk].mean())
            tab[name][f"ineff_med_{m}"] = float(np.median(r[m][msk]))
    out["strata"] = tab
    ident = e_area < 0.10
    w_all = wilcoxon(np.log(r["M0"]), np.log(r["M1"]), alternative="greater")
    w_id = wilcoxon(np.log(r["M0"][ident]), np.log(r["M1"][ident]),
                    alternative="greater")
    out["significance"] = {
        "wilcoxon_p_all": float(w_all.pvalue),
        "wilcoxon_p_identifiable": float(w_id.pvalue),
        "frac_M1_sharper_identifiable":
            float((r["M1"][ident] < r["M0"][ident]).mean()),
        "ineff_median_ci_M0_identifiable": boot_ci_median(r["M0"][ident]),
        "ineff_median_ci_M1_identifiable": boot_ci_median(r["M1"][ident]),
    }

    # ---------------- knobs 2 (geometry) and 3 (noise) ---------------------
    print("== knobs 2-3 ==", flush=True)
    B = 50
    sig_vals = np.exp(np.linspace(np.log(0.2), np.log(3.0), 7))
    k2 = {k: np.zeros(B) for k in
          ["spread_M0", "conf_M0", "spread_M1", "conf_M1",
           "spread_ex", "conf_ex"]}
    k3 = {m: np.zeros((B, len(sig_vals))) for m in ["M0", "M1", "ex"]}
    ths = {m: tail_threshold(tails[m], alpha) for m in models}
    for b in range(B):
        srng, xs, q, u_norm, stab, sig, master = base_scenario(b)
        grng = np.random.default_rng(800000 + b)
        cells6 = np.array([(ix, iy) for iy in range(2) for ix in range(3)])
        spread = (cells6 + grng.uniform(size=(6, 2))) / np.array([3, 2])
        quad = grng.integers(0, 4)
        qoff = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])[quad]
        conf = qoff + 0.5 * grng.uniform(size=(6, 2))
        for lay, key in [(spread, "spread"), (conf, "conf")]:
            d1, _ = one_scenario(xs, q, u_norm, stab, sig, lay, grng)
            pe, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
            k2[f"{key}_ex"][b] = region_mask(pe[0], th_e).sum() / 4096
            for m in models:
                P = model_probs(models[m], d1, device)[0]
                k2[f"{key}_{m}"][b] = region_mask(P, ths[m]).sum() / 4096
        lay8 = master[:8]
        for si, s in enumerate(sig_vals):
            d1, _ = one_scenario(xs, q, u_norm, stab, s, lay8,
                                 np.random.default_rng(900000 + b * 10 + si))
            pe, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
            k3["ex"][b, si] = region_mask(pe[0], th_e).sum() / 4096
            for m in models:
                P = model_probs(models[m], d1, device)[0]
                k3[m][b, si] = region_mask(P, ths[m]).sum() / 4096
    np.savez_compressed(data_dir / "ch4_dose23.npz", sig_vals=sig_vals,
                        **k2, **{f"k3_{m}": k3[m] for m in k3})
    x = np.log10(sig_vals)
    dose23 = {}
    for m in ("M0", "M1"):
        sl = [np.polyfit(x, k3[m][b], 1)[0] for b in range(B)]
        se = [np.polyfit(x, k3["ex"][b], 1)[0] for b in range(B)]
        rho = [spearmanr(k3[m][b], k3["ex"][b]).statistic for b in range(B)]
        ratios_m = k2[f"conf_{m}"] / np.maximum(k2[f"spread_{m}"], 1e-9)
        ratios_e = k2["conf_ex"] / np.maximum(k2["spread_ex"], 1e-9)
        dose23[m] = {
            "noise_slope_ratio": float(np.mean(sl) / np.mean(se)),
            "noise_spearman_median": float(np.nanmedian(rho)),
            "geo_ratio_median": float(np.median(ratios_m)),
            "geo_ratio_exact_median": float(np.median(ratios_e)),
            "geo_agreement": float(spearmanr(ratios_m, ratios_e).statistic),
        }
    out["dose_knob23"] = dose23

    # ---------------- memorization ablation --------------------------------
    print("== memorization ==", flush=True)
    with open(results_dir / "ch4_memorization.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["aug", "epoch", "train_loss", "val_nll"])
        for use_aug in (True, False):
            torch.manual_seed(4001)
            arng = np.random.default_rng(1)
            model = DeepSetsLocalizer(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                    weight_decay=1e-4)
            mcfg = cfg["model"]
            n_train = len(data["train"]["ids"])
            for ep in range(40):
                model.train()
                perm = arng.permutation(n_train)
                tot, nb = 0.0, 0
                for lo in range(0, n_train, 256):
                    idx = perm[lo:lo + 256]
                    batch = to_batch(data["train"], idx, device)
                    xs_b = data["train"]["xs"][idx]
                    if use_aug:
                        batch, xs_b, _ = d4_augment(batch, xs_b, arng, device)
                    tc = torch.tensor(pos_to_cell(xs_b, N_GRID))
                    tgt = smoothed_targets(tc, N_GRID,
                                           mcfg["target_smooth_std_cells"],
                                           mcfg["target_trunc_sigmas"], device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(batch)
                    logp = torch.log_softmax(logits.float(), -1)
                    loss = -(tgt * logp).sum(1).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot += float(loss); nb += 1
                vn = val_nll(model, data["val"], device)
                wtr.writerow([int(use_aug), ep, tot / nb, vn])
            print(f"  aug={use_aug} final val {vn:.3f}", flush=True)

    # ---------------- proper mode-recovery sanity ---------------------------
    print("== mode recovery ==", flush=True)
    hits, tried, attempts = 0, 0, 0
    mrng = np.random.default_rng(11)
    while tried < 24 and attempts < 500:
        attempts += 1
        srng, xs, q, u_norm, stab, sig, master = base_scenario(1000 + attempts)
        cell = pos_to_cell(xs[None], N_GRID)[0]
        xs_c = np.array([(cell % N_GRID + 0.5) / N_GRID,
                         (cell // N_GRID + 0.5) / N_GRID])
        sensors = master[:8]
        g = plume_ppm_per_kgh(torch.tensor(sensors), torch.tensor(xs_c)[None],
                              torch.tensor(u_norm)[None],
                              torch.tensor([stab])).numpy()
        if (q * g).max() < 1.0:
            continue
        tried += 1
        d1, _ = one_scenario(xs_c, q, u_norm, stab, 1e-4, sensors, mrng)
        pe, _ = ch4_posteriors(d1, cfg_q, device, verbose_every=0)
        hits += int(pe[0].argmax() == cell)
    out["mode_recovery"] = f"{hits}/{tried}"
    print(f"  mode recovery: {hits}/{tried}", flush=True)

    with open(results_dir / "ch4_extras2.json", "w") as f:
        json.dump(out, f, indent=2)
    update_json(results_dir / "gates.json",
                {"CH4_extras2": {"mode_recovery": out["mode_recovery"],
                                 "significance": out["significance"],
                                 "dose_knob23": dose23}})
    print("CH4 EXTRAS2 DONE", flush=True)


if __name__ == "__main__":
    main()
