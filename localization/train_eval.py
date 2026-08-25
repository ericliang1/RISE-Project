import argparse
import fcntl
import json
import os
import pathlib
import sys

import numpy as np
import torch

_R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_R / "simulator"), str(_R / "localization")]

from common import (cell_centers, get_device, load_config, pos_to_cell,
                    resolve, savez_atomic)
from conformal import region_mask, regions, tail_scores, tail_threshold
from networks import (DeepSetsT, GNNT, PhysHeadNet, SetTransformerT,
                      phys_head_on)
from scenarios import N_GRID, SPLITS, with_obs_wind

N_CELLS = N_GRID * N_GRID
NOISY_ONLY_MAPS = ("ens",)


def rad_m(sizes):
    return float(np.sqrt(np.median(sizes / N_CELLS) / np.pi) * 500)


def to_batch_t(d, idx, device):
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return {"sensors": t(d["sensors"][idx], torch.float32),
            "readings": t(d["readings"][idx], torch.float32),
            "keep": t(d["keep"][idx], torch.bool),
            "times": t(d["times"], torch.float32),
            "u_seq": t(d["u_seq"][idx], torch.float32),
            "u_mean": t(d["u_mean"][idx], torch.float32),
            "D": t(d["D"][idx], torch.float32),
            "sigma": t(d["sigma"][idx], torch.float32),
            "n_sensors": t(d["n_sensors"][idx], torch.float32)}


def d4_augment_t(batch, xs, u_seq, rng, device):
    B = batch["sensors"].shape[0]
    k = torch.tensor(rng.integers(0, 4, B), device=device)
    r = torch.tensor(rng.integers(0, 2, B), device=device)
    c = 0.5

    def rot(px, py, kk):
        ox = torch.where(kk == 0, px, torch.where(kk == 1, -py,
                         torch.where(kk == 2, -px, py)))
        oy = torch.where(kk == 0, py, torch.where(kk == 1, px,
                         torch.where(kk == 2, -py, -px)))
        return ox, oy

    sx = batch["sensors"][..., 0] - c
    sy = batch["sensors"][..., 1] - c
    sy = torch.where(r[:, None] == 1, -sy, sy)
    sx, sy = rot(sx, sy, k[:, None])
    out = dict(batch)
    out["sensors"] = torch.stack([sx + c, sy + c], dim=-1)
    ux, uy = u_seq[..., 0], u_seq[..., 1]
    uy = torch.where(r[:, None] == 1, -uy, uy)
    ux, uy = rot(ux, uy, k[:, None])
    out["u_seq"] = torch.stack([ux, uy], dim=-1)
    out["u_mean"] = out["u_seq"].mean(1)
    xsx = torch.tensor(xs[:, 0], device=device) - c
    xsy = torch.tensor(xs[:, 1], device=device) - c
    xsy = torch.where(r == 1, -xsy, xsy)
    xsx, xsy = rot(xsx, xsy, k)
    xs_new = torch.stack([xsx + c, xsy + c], dim=-1).double().cpu().numpy()
    return out, xs_new, (r * 4 + k)


def d4_target_perms(n, device):
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


def smoothed_targets(true_cells, n, std_cells, trunc_sigmas, device):
    ix = (true_cells % n).to(device)
    iy = (true_cells // n).to(device)
    r = int(np.ceil(trunc_sigmas * std_cells))
    offs = torch.arange(-r, r + 1, device=device)
    dx, dy = torch.meshgrid(offs, offs, indexing="xy")
    d2 = (dx ** 2 + dy ** 2).float()
    w = torch.exp(-d2 / (2.0 * std_cells ** 2))
    w = w * (d2.sqrt() <= trunc_sigmas * std_cells)

    B = true_cells.shape[0]
    tgt = torch.zeros(B, n * n, device=device)
    gx = (ix[:, None, None] + dx[None]).clamp_(0, n - 1)
    gy = (iy[:, None, None] + dy[None]).clamp_(0, n - 1)
    flat = (gy * n + gx).reshape(B, -1)
    tgt.scatter_add_(1, flat, w.reshape(1, -1).expand(B, -1))
    return tgt / tgt.sum(dim=1, keepdim=True)


def make_view(cfg, dd, name_data, view, maps=None):
    split, d_clean = name_data
    if view == "clean":
        assert maps is None, "ens maps are noisy-view only"
        return d_clean, None
    d = with_obs_wind(d_clean, np.load(dd / f"ch4tu_{split}_uobs.npy"))
    if maps == "ens":
        stats = torch.tensor(
            np.load(dd / f"ch4tu_{split}_maps_ens.npz")["maps"])
    else:
        stats = None
    return d, stats


def stage_train(cfg, device, views, seed, maps=None, arch="deepsets"):
    dd = resolve(cfg, "data_dir")
    rr = resolve(cfg, "results_dir")
    tr = cfg["training"]
    m = cfg["model"]
    alpha = cfg["conformal"]["alpha"]
    clean = {s: dict(np.load(dd / f"ch4t_{s}.npz", allow_pickle=True))
             for s in SPLITS}
    view_list = [views]
    if maps in NOISY_ONLY_MAPS:
        assert views == "noisy", f"{maps} configs are noisy-view only"
    V = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps)
             for v in view_list} for s in SPLITS}
    conds = (("noisy",) if maps in NOISY_ONLY_MAPS
             else ("clean", "noisy"))
    A = {s: {v: make_view(cfg, dd, (s, clean[s]), v, maps)
             for v in conds} for s in ("calib", "test")}
    torch.manual_seed(6000 + seed + 1300)
    rng = np.random.default_rng(seed)
    n_ch = V["train"][view_list[0]][1].shape[1] if maps else 0
    if arch == "gnn":
        base_cls = GNNT
    elif arch == "st":
        base_cls = SetTransformerT
    else:
        base_cls = DeepSetsT
    if not maps:
        model = base_cls(cfg).to(device)
    elif arch == "deepsets":
        model = PhysHeadNet(cfg, n_ch).to(device)
    else:
        model = phys_head_on(base_cls, n_ch)(cfg).to(device)
    print(f"  {type(model).__name__} parameters: "
          f"{sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=200, eta_min=tr["lr_final"])
    inv = d4_target_perms(N_GRID, device)
    n_train = len(clean["train"]["ids"])
    best, best_state = np.inf, None

    def fwd_loss(v, idx):
        d, stats = V["train"][v]
        batch = to_batch_t(d, idx, device)
        batch, xs, codes = d4_augment_t(batch, d["xs"][idx],
                                        batch["u_seq"], rng, device)
        gi = inv[codes]
        if maps:
            st = stats[idx].to(device)
            batch["stats"] = st.reshape(len(idx), n_ch, N_CELLS).gather(
                2, gi[:, None, :].expand(-1, n_ch, -1)).reshape(
                len(idx), n_ch, N_GRID, N_GRID)
        tc = torch.tensor(pos_to_cell(xs, N_GRID))
        tgt = smoothed_targets(tc, N_GRID, m["target_smooth_std_cells"],
                               m["target_trunc_sigmas"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        logp = torch.log_softmax(logits.float(), -1)
        return -(tgt * logp).sum(1).mean()

    @torch.no_grad()
    def probs(split_views, split, v):
        d, stats = split_views[split][v]
        out = []
        for lo in range(0, len(d["ids"]), 512):
            idx = np.arange(lo, min(lo + 512, len(d["ids"])))
            b = to_batch_t(d, idx, device)
            if stats is not None:
                b["stats"] = stats[idx].to(device).view(len(idx), -1, N_GRID,
                                                        N_GRID)
            out.append(torch.softmax(model(b).float(), -1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    n_epochs = int(os.environ.get("PW_EPOCHS", 200))
    warmup = int(os.environ.get("PW_HEAD_WARMUP", 0))
    head_params = list(model.phys.parameters()) if (maps and warmup) else []
    for ep in range(n_epochs):
        if head_params:
            for pp in head_params:
                pp.requires_grad_(ep >= warmup)
        model.train()
        perm = rng.permutation(n_train)
        for lo in range(0, n_train, tr["batch_size"]):
            idx = perm[lo: lo + tr["batch_size"]]
            loss = sum(fwd_loss(v, idx) for v in view_list) / len(view_list)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                tr["grad_clip"])
            if ep == 0 and not (torch.isfinite(loss) and torch.isfinite(gn)):
                raise RuntimeError("non-finite loss/grad on first epoch")
            opt.step()
        sched.step()
        model.eval()
        tot = 0.0
        for v in view_list:
            d, stats = V["val"][v]
            with torch.no_grad():
                for lo in range(0, len(d["ids"]), 512):
                    idx = np.arange(lo, min(lo + 512, len(d["ids"])))
                    b = to_batch_t(d, idx, device)
                    if stats is not None:
                        b["stats"] = stats[idx].to(device).view(
                            len(idx), -1, N_GRID, N_GRID)
                    logp = torch.log_softmax(model(b).float(), -1)
                    tc = torch.tensor(d["true_cell"][idx], device=device)
                    tot += float(-logp.gather(1, tc[:, None]).sum())
        crit = tot / (len(V["val"][view_list[0]][0]["ids"]) * len(view_list))
        if crit < best:
            best = crit
            best_state = {k: v_.detach().clone()
                          for k, v_ in model.state_dict().items()}
        if ep % 25 == 0:
            print(f"  ep {ep} val_nll {crit:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no epoch improved")
    model.load_state_dict(best_state)
    model.eval()

    mt = {"ens": "_ens", None: "_nomaps"}[maps]
    at = "" if arch == "deepsets" else f"_{arch}"
    tag = f"pw_{views}{mt}{at}_seed{seed}"
    out = {"tag": tag, "maps": maps or "off", "arch": arch,
           "val_nll": float(best)}
    if os.environ.get("PW_SAVE_CKPT"):
        ck = dd / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save({"model_state": model.state_dict(), "maps": maps or "off",
                    "arch": arch, "seed": seed, "val_nll": float(best)},
                   ck / f"{tag}.pt")
        print(f"saved checkpoint {ck / f'{tag}.pt'}", flush=True)
    for cond in conds:
        Pc = probs(A, "calib", cond)
        Pt = probs(A, "test", cond)
        rngc = np.random.default_rng(cfg["conformal"]["score_seed"])
        d_cal = A["calib"][cond][0]
        d_tst = A["test"][cond][0]
        th = tail_threshold(tail_scores(Pc, d_cal["true_cell"], rngc), alpha)
        reg = regions(Pt, th, d_tst["true_cell"])
        radii = np.sqrt(reg["sizes"] / N_CELLS / np.pi) * 500
        out[cond] = {"coverage": float(reg["covered"].mean()),
                     "median_radius_m": rad_m(reg["sizes"]),
                     "frac_below_50m": float((radii < 50).mean())}
        masks = np.stack([region_mask(Pt[i], th) for i in range(len(Pt))])
        savez_atomic(dd / f"pw_audit_{tag}_{cond}.npz",
                     sizes=reg["sizes"], covered=reg["covered"],
                     masks=np.packbits(masks, axis=1))
    res_path = rr / "paired_wind.json"
    with open(rr / "paired_wind.json.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        all_res = json.load(open(res_path)) if res_path.exists() else {}
        all_res[tag] = out
        tmp = str(res_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_res, f, indent=2)
        os.replace(tmp, res_path)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", choices=["ens", "off"], required=True)
    ap.add_argument("--arch", choices=["deepsets", "gnn", "st"],
                    default="deepsets")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    device = get_device()
    stage_train(cfg, device, "noisy", args.seed,
                maps=None if args.maps == "off" else args.maps,
                arch=args.arch)


if __name__ == "__main__":
    main()
