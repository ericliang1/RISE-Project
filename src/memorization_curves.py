"""Regenerate the D4-augmentation memorization ablation curves (fig11).

Trains the baseline recipe for 40 epochs with and without augmentation,
recording train loss and validation NLL per epoch.  No checkpoints written.
"""
import csv

import numpy as np
import torch

from common import get_device, load_config, pos_to_cell, resolve
from inference import load_split, to_batch
from model import DeepSetsLocalizer, smoothed_targets
from train import d4_augment, val_nll


def run(cfg, device, use_aug, epochs, writer):
    tr = cfg["training"]
    torch.manual_seed(cfg["seeds"]["torch_train_base"] + 1)
    rng = np.random.default_rng(1)
    data_dir = resolve(cfg, "data_dir")
    d_train = load_split(data_dir, "train")
    d_val = load_split(data_dir, "val")
    model = DeepSetsLocalizer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    m = cfg["model"]
    n_grid = cfg["grid"]["n"]
    n_train = len(d_train["ids"])
    bs = tr["batch_size"]
    amp_dtype = getattr(torch, tr["amp_dtype"])
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        tot, nb = 0.0, 0
        for lo in range(0, n_train, bs):
            idx = perm[lo: lo + bs]
            batch = to_batch(d_train, idx, device)
            xs = d_train["xs"][idx]
            if use_aug:
                batch, xs, _ = d4_augment(batch, xs, rng, device)
            tc = torch.tensor(pos_to_cell(xs, n_grid))
            tgt = smoothed_targets(tc, n_grid, m["target_smooth_std_cells"],
                                   m["target_trunc_sigmas"], device)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(batch)
            logp = torch.log_softmax(logits.float(), dim=-1)
            loss = -(tgt * logp).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            tot += float(loss)
            nb += 1
        vnll = val_nll(model, d_val, device)
        writer.writerow([int(use_aug), epoch, tot / nb, vnll])
        print(f"aug={use_aug} epoch {epoch:2d} loss {tot/nb:.4f} "
              f"val {vnll:.4f}", flush=True)


def main():
    cfg = load_config()
    device = get_device()
    out = resolve(cfg, "results_dir") / "memorization_curves.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aug", "epoch", "train_loss", "val_nll"])
        run(cfg, device, True, 40, w)
        run(cfg, device, False, 40, w)


if __name__ == "__main__":
    main()
