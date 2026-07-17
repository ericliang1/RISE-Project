"""Stage 3: train the DeepSets localizer (paper Sec. 4.1).

Cross-entropy to a smoothed Gaussian-bump target; AdamW, cosine lr decay,
early stopping on validation NLL (true-cell NLL), checkpoint by val NLL only.

Usage:  python src/train.py --seed 1 [--config ...]
"""
import argparse
import csv
import time

import numpy as np
import torch

from baselines import localization_errors, peak_sensor_estimates
from common import cell_center_of, get_device, load_config, resolve, update_json
from inference import heatmaps, load_split, to_batch
from model import DeepSetsLocalizer, count_params, smoothed_targets


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


def train_one_seed(cfg, seed, device):
    tr = cfg["training"]
    torch.manual_seed(cfg["seeds"]["torch_train_base"] + seed)
    np_rng = np.random.default_rng(seed)

    data_dir = resolve(cfg, "data_dir")
    d_train = load_split(data_dir, "train")
    d_val = load_split(data_dir, "val")

    model = DeepSetsLocalizer(cfg).to(device)
    n_params = count_params(model)
    print(f"seed {seed}: {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tr["max_epochs"], eta_min=tr["lr_final"])
    amp_dtype = getattr(torch, tr["amp_dtype"])

    n_grid = cfg["grid"]["n"]
    m = cfg["model"]
    true_cells_train = torch.tensor(d_train["true_cell"])
    n_train = len(d_train["ids"])
    bs = tr["batch_size"]

    ckpt_dir = resolve(cfg, "checkpoints_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"seed{seed}.pt"
    log_path = ckpt_dir / f"train_log_seed{seed}.csv"

    best_nll, best_epoch, since_best = np.inf, -1, 0
    t0 = time.time()
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_nll", "lr", "wall_s"])
        for epoch in range(tr["max_epochs"]):
            model.train()
            perm = np_rng.permutation(n_train)
            tot_loss, n_batches = 0.0, 0
            for lo in range(0, n_train, bs):
                idx = perm[lo: lo + bs]
                batch = to_batch(d_train, idx, device)
                tgt = smoothed_targets(true_cells_train[idx], n_grid,
                                       m["target_smooth_std_cells"],
                                       m["target_trunc_sigmas"], device)
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
            vnll = val_nll(model, d_val, device)
            writer.writerow([epoch, tot_loss / n_batches, vnll,
                             sched.get_last_lr()[0], time.time() - t0])
            logf.flush()
            if vnll < best_nll:
                best_nll, best_epoch, since_best = vnll, epoch, 0
                torch.save({"model_state": model.state_dict(),
                            "epoch": epoch, "val_nll": vnll, "seed": seed,
                            "n_params": n_params}, ckpt_path)
            else:
                since_best += 1
            if epoch % 10 == 0 or since_best == 0:
                print(f"epoch {epoch:3d} loss {tot_loss/n_batches:.4f} "
                      f"val_nll {vnll:.4f} best {best_nll:.4f}@{best_epoch}",
                      flush=True)
            if since_best >= tr["early_stop_patience"]:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    wall_h = (time.time() - t0) / 3600.0
    return ckpt_path, best_nll, best_epoch, wall_h, n_params, d_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()

    ckpt_path, best_nll, best_epoch, wall_h, n_params, d_val = \
        train_one_seed(cfg, args.seed, device)

    # G2 evaluation on validation: model MAP vs peak-sensor heuristic
    model = DeepSetsLocalizer(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    model.eval()
    m_mean, m_med = map_error(model, d_val, cfg, device)
    ps_err = localization_errors(peak_sensor_estimates(d_val), d_val["xs"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    results_dir = resolve(cfg, "results_dir")
    entry = {
        f"seed{args.seed}_val_map_error_mean": m_mean,
        f"seed{args.seed}_val_map_error_median": m_med,
        f"seed{args.seed}_best_val_nll": best_nll,
        f"seed{args.seed}_best_epoch": best_epoch,
        f"seed{args.seed}_gpu_hours": wall_h,
        f"seed{args.seed}_param_count": n_params,
        "val_peak_sensor_error_mean": float(ps_err.mean()),
        "val_peak_sensor_error_median": float(np.median(ps_err)),
        "gpu_model": gpu,
    }
    if args.seed == 1:
        entry["model_beats_peak_sensor"] = bool(m_mean < ps_err.mean())
    update_json(results_dir / "gates.json", {"G2": entry})
    print(f"[G2] seed {args.seed}: val MAP mean {m_mean:.4f} median {m_med:.4f} | "
          f"peak-sensor mean {ps_err.mean():.4f} median {np.median(ps_err):.4f} | "
          f"{wall_h:.2f} GPU-h", flush=True)


if __name__ == "__main__":
    main()
