"""Noisy-wind students trained with ORDINARY LABELS (no teacher anywhere):
det maps (2ch) and ensemble maps (3ch).  Mirrors train_u minus the teacher;
val criterion = true-cell NLL.  Usage: python src/t3_labels.py --maps det|ens
[--seed 1]"""
import argparse
import json

import numpy as np
import torch

from common import get_device, load_config, pos_to_cell, resolve
from methane_t import N_GRID, SPLITS, d4_augment_t, to_batch_t
from methane_t_uncertain import PhysHeadNet, audit, probs_of, with_obs_wind
from model import smoothed_targets
from train import d4_target_perms

N_CELLS = N_GRID * N_GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", choices=["det", "ens"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    m = cfg["model"]
    data = {}
    for name in SPLITS:
        d = dict(np.load(dd / f"ch4t_{name}.npz", allow_pickle=True))
        data[name] = with_obs_wind(d, np.load(dd / f"ch4tu_{name}_uobs.npy"))
    stats = {n: torch.tensor(
        np.load(dd / f"ch4tu_{n}_maps_{args.maps}.npz")["maps"])
        for n in SPLITS}
    n_ch = stats["train"].shape[1]

    torch.manual_seed(6000 + args.seed + 1100)
    rng = np.random.default_rng(args.seed)
    model = PhysHeadNet(cfg, n_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    d_train, d_val = data["train"], data["val"]
    n_train = len(d_train["ids"])
    best, best_state = np.inf, None
    for ep in range(200):
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            batch = to_batch_t(d_train, idx, device)
            batch, xs, codes = d4_augment_t(batch, d_train["xs"][idx],
                                            batch["u_seq"], rng, device)
            st = stats["train"][idx].to(device)
            batch["stats"] = st.reshape(len(idx), n_ch, N_CELLS).gather(
                2, inv[codes][:, None, :].expand(-1, n_ch, -1)).reshape(
                len(idx), n_ch, N_GRID, N_GRID)
            tc = torch.tensor(pos_to_cell(xs, N_GRID))
            tgt = smoothed_targets(tc, N_GRID, m["target_smooth_std_cells"],
                                   m["target_trunc_sigmas"], device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            logp = torch.log_softmax(logits.float(), -1)
            loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError("non-finite loss/grad on first epoch")
            opt.step()
        sched.step()
        tot = 0.0
        with torch.no_grad():
            for lo in range(0, len(d_val["ids"]), 512):
                idx = np.arange(lo, min(lo + 512, len(d_val["ids"])))
                b = to_batch_t(d_val, idx, device)
                b["stats"] = stats["val"][idx].to(device).view(
                    len(idx), n_ch, N_GRID, N_GRID)
                logp = torch.log_softmax(model(b).float(), -1)
                tcv = torch.tensor(d_val["true_cell"][idx], device=device)
                tot += float(-logp.gather(1, tcv[:, None]).sum())
        crit = tot / len(d_val["ids"])
        if crit < best:
            best = crit
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep % 25 == 0:
            print(f"  ep {ep} val_nll {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch improved")
    model.load_state_dict(best_state)
    model.eval()
    Pc = probs_of(model, data["calib"], device, stats["calib"])
    Pt = probs_of(model, data["test"], device, stats["test"])
    tag = f"s_{args.maps}_labels" + ("" if args.seed == 1
                                     else f"_seed{args.seed}")
    audit(Pc, Pt, data, cfg, tag, rr,
          npz_path=dd / f"ch4tu_audit_{tag}.npz")


if __name__ == "__main__":
    main()
