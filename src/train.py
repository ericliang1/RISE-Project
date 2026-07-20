"""Stage 3: train the DeepSets localizer (paper Sec. 4.1).

Cross-entropy to a smoothed Gaussian-bump target; AdamW, cosine lr decay,
early stopping on validation NLL (true-cell NLL), checkpoint by val NLL only.

DOCUMENTED DEVIATION (training protocol only): each training batch is
transformed by a random D4 symmetry of the unit square (rotations by k*90deg
+ reflection, applied jointly to sensors, wind, and source).  This is an
EXACT physics symmetry (validated by the G1 rotation/reflection gate) and the
benchmark prior is D4-invariant, so augmented scenarios are exact draws from
the same prior.  Without it, the paper's recipe memorizes the 10k training
scenarios with zero generalization (val NLL never better than uniform); with
it, val NLL tracks train loss.  See GATES.md (G2).

Usage:  python src/train.py --seed 1 [--config ...]
"""
import argparse
import csv
import time

import numpy as np
import torch

from baselines import localization_errors, peak_sensor_estimates
from common import (cell_center_of, get_device, load_config, pos_to_cell,
                    resolve, update_json)
from inference import heatmaps, load_split, to_batch
from model import DeepSetsLocalizer, count_params, smoothed_targets


def d4_augment(batch, xs, rng, device):
    """Random per-scenario D4 symmetry about the domain center applied to
    sensors, wind, and source.  Returns (augmented batch, augmented xs)."""
    B = batch["sensors"].shape[0]
    k = torch.tensor(rng.integers(0, 4, B), device=device)
    r = torch.tensor(rng.integers(0, 2, B), device=device)
    c = 0.5
    sx = batch["sensors"][..., 0] - c
    sy = batch["sensors"][..., 1] - c
    ux, uy = batch["u"][:, 0], batch["u"][:, 1]
    xsx = torch.tensor(xs[:, 0], device=device, dtype=torch.float32) - c
    xsy = torch.tensor(xs[:, 1], device=device, dtype=torch.float32) - c
    # reflect about the x-axis, then rotate by k*90deg
    sy = torch.where(r[:, None] == 1, -sy, sy)
    uy = torch.where(r == 1, -uy, uy)
    xsy = torch.where(r == 1, -xsy, xsy)

    def rot(px, py, kk):
        ox = torch.where(kk == 0, px, torch.where(kk == 1, -py,
                         torch.where(kk == 2, -px, py)))
        oy = torch.where(kk == 0, py, torch.where(kk == 1, px,
                         torch.where(kk == 2, -py, -px)))
        return ox, oy

    sx, sy = rot(sx, sy, k[:, None])
    ux, uy = rot(ux, uy, k)
    xsx, xsy = rot(xsx, xsy, k)
    out = dict(batch)
    out["sensors"] = torch.stack([sx + c, sy + c], dim=-1)
    out["u"] = torch.stack([ux, uy], dim=-1)
    xs_new = torch.stack([xsx + c, xsy + c], dim=-1).double().cpu().numpy()
    codes = r * 4 + k                     # (B,) D4 op code for target permutation
    return out, xs_new, codes


def blur_teacher(P, n, std_cells, trunc=3.0, device="cpu"):
    """Convolve teacher posteriors (S, n*n) with an isotropic Gaussian
    (std in cells, truncated at trunc*std), renormalized.  D4-equivariant, so
    it commutes with the augmentation permutation.  std <= 0.02 -> identity
    (raw teacher).  Computed in chunks on `device`, returned on CPU."""
    if std_cells <= 0.02:
        return P
    r = int(np.ceil(trunc * std_cells))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    g1 = torch.exp(-xs ** 2 / (2 * std_cells ** 2))
    k2 = torch.outer(g1, g1)
    k2 = ((k2 / k2.sum())[None, None]).to(device)
    img = P.reshape(-1, 1, n, n)
    out = torch.empty_like(img)
    for lo in range(0, img.shape[0], 2048):
        hi = min(lo + 2048, img.shape[0])
        out[lo:hi] = torch.nn.functional.conv2d(
            img[lo:hi].to(device), k2, padding=r).cpu()
    out = out.reshape(-1, n * n)
    return out / out.sum(dim=1, keepdim=True)


def curriculum_sigma(epoch, cs):
    """Teacher blur schedule: hold at sigma_start, linear anneal to sigma_end
    between hold_epochs and anneal_end, then hold at sigma_end."""
    s0 = float(cs["curriculum_sigma_start"])
    s1 = float(cs["curriculum_sigma_end"])
    hold = int(cs["curriculum_hold_epochs"])
    end = int(cs["curriculum_anneal_end"])
    if epoch < hold:
        return s0
    if epoch >= end:
        return s1
    frac = (epoch - hold) / max(end - hold, 1)
    return s0 + (s1 - s0) * frac


def val_area_criterion(model, d, device, mass=0.90, cov_floor=0.85):
    """Checkpoint criterion for curriculum runs: median validation raw-HPD
    area at `mass`, guarded by a raw-coverage floor (an epoch whose raw HPD
    coverage collapses cannot win on sharpness alone).  Returns +inf when the
    floor is violated.  Same quantity as the cross-variant selection rule."""
    from conformal import region_mask
    P = heatmaps(model, d, device).astype(np.float64)
    n_cells = P.shape[1]
    areas = np.empty(len(P))
    covered = np.empty(len(P), dtype=bool)
    for i in range(len(P)):
        m = region_mask(P[i], 1.0 - mass)
        areas[i] = m.sum() / n_cells
        covered[i] = m[d["true_cell"][i]]
    if covered.mean() < cov_floor:
        return float("inf")
    return float(np.median(areas))


def teacher_ce(model, d, P_teacher, device, batch_size=512):
    """Mean CE of model heatmaps against exact-posterior soft targets."""
    model.eval()
    n = len(d["ids"])
    tot = 0.0
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            idx = np.arange(lo, min(lo + batch_size, n))
            logits = model(to_batch(d, idx, device)).float()
            logp = torch.log_softmax(logits, dim=-1)
            tgt = P_teacher[idx].to(device)
            tot += float(-(tgt * logp).sum(dim=1).sum())
    return tot / n


def val_nll(model, d, device, batch_size=512):
    """Mean true-cell NLL over a split."""
    model.eval()
    n = len(d["ids"])
    tot = 0.0
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            idx = np.arange(lo, min(lo + batch_size, n))
            logits = model(to_batch(d, idx, device)).float()
            logp = torch.log_softmax(logits, dim=-1)
            tc = torch.tensor(d["true_cell"][idx], device=device)
            tot += float(-logp.gather(1, tc[:, None]).sum())
    return tot / n


def map_error(model, d, cfg, device):
    """Mean/median distance from argmax cell center to the true source."""
    P = heatmaps(model, d, device)
    est = cell_center_of(P.argmax(axis=1), cfg["grid"]["n"])
    err = localization_errors(est, d["xs"])
    return float(err.mean()), float(np.median(err))


def d4_target_perms(n, device):
    """Inverse cell permutations for the 8 D4 ops, matching d4_augment exactly:
    reflection (y -> -y) first, then k x 90-degree rotation.  Returns int64
    tensor (8, n*n) with code = r*4 + k such that
        target_new = target_old[:, inv[code]] .
    """
    from common import cell_centers
    centers = cell_centers(n)
    inv = np.empty((8, n * n), dtype=np.int64)
    for r in range(2):
        for k in range(4):
            x = centers[:, 0] - 0.5
            y = centers[:, 1] - 0.5
            if r:
                y = -y
            for _ in range(k):
                x, y = -y, x
            new_cell = pos_to_cell(np.stack([x + 0.5, y + 0.5], -1), n)
            code = r * 4 + k
            inv[code, new_cell] = np.arange(n * n)
    return torch.tensor(inv, device=device)


def concat_scenarios(a, b):
    """Concatenate two scenario dicts (shared `times`)."""
    out = {}
    for k in a:
        out[k] = a[k] if k == "times" else np.concatenate([a[k], b[k]], axis=0)
    return out


def slice_scenarios(d, n):
    """First n scenarios of a scenario dict."""
    return {k: (v if k == "times" else v[:n]) for k, v in d.items()}


def train_one_seed(cfg, seed, device, distill=False, curriculum=False,
                   train_size=None, use_xl=False, stem_override=None):
    tr = cfg["training"]
    torch.manual_seed(cfg["seeds"]["torch_train_base"] + seed
                      + (9000 if curriculum else 5000 if distill else 0))
    np_rng = np.random.default_rng(seed)

    data_dir = resolve(cfg, "data_dir")
    d_train = load_split(data_dir, "train")
    if use_xl:
        d_train = concat_scenarios(d_train, load_split(data_dir, "train_xl"))
    if train_size is not None:
        d_train = slice_scenarios(d_train, train_size)
    d_val = load_split(data_dir, "val")
    n_g = cfg["grid"]["n"]
    P_teacher = P_train_raw = None
    cur_sigma = None
    cs = cfg["distill"]
    if distill:
        z = np.load(data_dir / "train_posterior.npz")
        probs = z["probs"]
        ids = z["ids"]
        if use_xl:
            zx = np.load(data_dir / "train_xl_posterior.npz")
            probs = np.concatenate([probs, zx["probs"]], axis=0)
            ids = np.concatenate([ids, zx["ids"]])
        if train_size is not None:
            probs, ids = probs[:train_size], ids[:train_size]
        assert (ids == d_train["ids"]).all()
        P_train_raw = torch.tensor(probs.astype(np.float32))
        inv_perms = d4_target_perms(n_g, device)
        zv = np.load(data_dir / "val_posterior.npz")
        assert (zv["ids"] == d_val["ids"]).all()
        P_val_raw = torch.tensor(zv["probs"].astype(np.float32))
        mix = float(cs.get("mix_lambda", 1.0))
        if curriculum:
            # selection criterion is fixed: CE vs the FINAL (sigma_end) teacher
            P_val_teacher = blur_teacher(P_val_raw, n_g,
                                         float(cs["curriculum_sigma_end"]),
                                         device=device)
        else:
            blur = float(cs.get("teacher_blur_std_cells", 0.0))
            P_teacher = blur_teacher(P_train_raw, n_g, blur, device=device)
            P_val_teacher = blur_teacher(P_val_raw, n_g, blur, device=device)

    model = DeepSetsLocalizer(cfg).to(device)
    n_params = count_params(model)
    print(f"{'distill' if distill else 'baseline'} seed {seed}: "
          f"{n_params/1e6:.2f}M params", flush=True)

    max_epochs = cfg["distill"]["max_epochs"] if distill else tr["max_epochs"]
    patience = (cfg["distill"]["early_stop_patience"] if distill
                else tr["early_stop_patience"])
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs, eta_min=tr["lr_final"])
    amp_dtype = getattr(torch, tr["amp_dtype"])

    n_grid = cfg["grid"]["n"]
    m = cfg["model"]
    use_aug = tr.get("d4_augment", True)
    n_train = len(d_train["ids"])
    bs = tr["batch_size"]

    ckpt_dir = resolve(cfg, "checkpoints_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_override or (f"model3_seed{seed}" if curriculum
                             else f"model2_seed{seed}" if distill
                             else f"seed{seed}")
    ckpt_path = ckpt_dir / f"{stem}.pt"
    log_path = ckpt_dir / f"train_log_{stem}.csv"

    best_nll, best_epoch, since_best = np.inf, -1, 0
    t0 = time.time()
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_nll", "lr", "wall_s"])
        for epoch in range(max_epochs):
            if curriculum:
                sig_e = round(curriculum_sigma(epoch, cs), 4)
                if sig_e != cur_sigma:
                    P_teacher = blur_teacher(P_train_raw, n_g, sig_e,
                                             device=device)
                    cur_sigma = sig_e
            model.train()
            perm = np_rng.permutation(n_train)
            tot_loss, n_batches = 0.0, 0
            for lo in range(0, n_train, bs):
                idx = perm[lo: lo + bs]
                batch = to_batch(d_train, idx, device)
                xs = d_train["xs"][idx]
                codes = None
                if use_aug:
                    batch, xs, codes = d4_augment(batch, xs, np_rng, device)
                tc = torch.tensor(pos_to_cell(xs, n_grid))
                tgt = smoothed_targets(tc, n_grid,
                                       m["target_smooth_std_cells"],
                                       m["target_trunc_sigmas"], device)
                if distill:
                    # physics-informed soft target: exact posterior, permuted
                    # by the same D4 op applied to the inputs; mixed with the
                    # standard smoothed target by mix_lambda
                    tp = P_teacher[idx].to(device)
                    if codes is not None:
                        tp = tp.gather(1, inv_perms[codes])
                    tgt = mix * tp + (1.0 - mix) * tgt
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(batch)
                logp = torch.log_softmax(logits.float(), dim=-1)
                loss = -(tgt * logp).sum(dim=1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                opt.step()
                tot_loss += float(loss)
                n_batches += 1
            sched.step()
            if curriculum:
                # sharpness-first criterion with a raw-coverage floor: CE vs a
                # sharp teacher penalizes slightly-misplaced sharpness so hard
                # that it would select the blurry mid-anneal student
                vnll = val_area_criterion(model, d_val, device)
            elif distill:
                # early-stop criterion matches the objective: mean CE of the
                # model's val heatmaps against the exact val posteriors
                vnll = teacher_ce(model, d_val, P_val_teacher, device)
            else:
                vnll = val_nll(model, d_val, device)
            writer.writerow([epoch, tot_loss / n_batches, vnll,
                             sched.get_last_lr()[0], time.time() - t0])
            logf.flush()
            if vnll < best_nll:
                best_nll, best_epoch, since_best = vnll, epoch, 0
                torch.save({"model_state": model.state_dict(), "arch": "v1",
                            "distill": distill, "curriculum": curriculum,
                            "epoch": epoch, "val_nll": vnll,
                            "seed": seed, "n_params": n_params}, ckpt_path)
            elif not curriculum or epoch >= int(cs["curriculum_anneal_end"]):
                # curriculum runs may not early-stop before the anneal completes
                since_best += 1
            if epoch % 10 == 0 or since_best == 0:
                extra = f" sigma {cur_sigma:.2f}" if curriculum else ""
                print(f"epoch {epoch:3d} loss {tot_loss/n_batches:.4f} "
                      f"val_nll {vnll:.4f} best {best_nll:.4f}@{best_epoch}{extra}",
                      flush=True)
            if since_best >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    wall_h = (time.time() - t0) / 3600.0
    return ckpt_path, best_nll, best_epoch, wall_h, n_params, d_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--distill", action="store_true",
                    help="physics-likelihood distillation (tag model2)")
    ap.add_argument("--mix-lambda", type=float, default=None,
                    help="override distill.mix_lambda")
    ap.add_argument("--teacher-blur", type=float, default=None,
                    help="override distill.teacher_blur_std_cells")
    ap.add_argument("--curriculum", action="store_true",
                    help="progressive teacher sharpening (tag model3; implies --distill)")
    ap.add_argument("--sigma-end", type=float, default=None,
                    help="override distill.curriculum_sigma_end")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mix_lambda is not None:
        cfg["distill"]["mix_lambda"] = args.mix_lambda
    if args.teacher_blur is not None:
        cfg["distill"]["teacher_blur_std_cells"] = args.teacher_blur
    if args.sigma_end is not None:
        cfg["distill"]["curriculum_sigma_end"] = args.sigma_end
    if args.curriculum:
        args.distill = True
    device = get_device()

    ckpt_path, best_nll, best_epoch, wall_h, n_params, d_val = \
        train_one_seed(cfg, args.seed, device, args.distill, args.curriculum)

    # G2 evaluation on validation: model MAP vs peak-sensor heuristic
    model = DeepSetsLocalizer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    model.eval()
    m_mean, m_med = map_error(model, d_val, cfg, device)
    ps_err = localization_errors(peak_sensor_estimates(d_val), d_val["xs"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    results_dir = resolve(cfg, "results_dir")
    key = (f"model3_seed{args.seed}" if args.curriculum
           else f"model2_seed{args.seed}" if args.distill else f"seed{args.seed}")
    entry = {
        f"{key}_val_map_error_mean": m_mean,
        f"{key}_val_map_error_median": m_med,
        f"{key}_best_val_nll": best_nll,
        f"{key}_best_epoch": best_epoch,
        f"{key}_gpu_hours": wall_h,
        f"{key}_param_count": n_params,
        "val_peak_sensor_error_mean": float(ps_err.mean()),
        "val_peak_sensor_error_median": float(np.median(ps_err)),
        "gpu_model": gpu,
    }
    if args.seed == 1 and not args.distill:
        entry["model_beats_peak_sensor"] = bool(m_mean < ps_err.mean())
    update_json(results_dir / "gates.json", {"G2": entry})
    print(f"[G2] {key}: val MAP mean {m_mean:.4f} median {m_med:.4f} | "
          f"peak-sensor mean {ps_err.mean():.4f} median {np.median(ps_err):.4f} | "
          f"{wall_h:.2f} GPU-h", flush=True)


if __name__ == "__main__":
    main()
