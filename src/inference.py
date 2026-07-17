"""Split loading and batched model inference shared by train/conformal/dose stages."""
import numpy as np
import torch

from model import DeepSetsLocalizer


def load_split(data_dir, name):
    """npz -> dict of numpy arrays (NaN sensor padding replaced by 0;
    the keep mask already excludes padded rows)."""
    d = dict(np.load(data_dir / f"{name}.npz", allow_pickle=True))
    d["sensors"] = np.nan_to_num(d["sensors"], nan=0.0)
    return d


def to_batch(d, idx, device):
    """Assemble a model input batch for scenario indices idx."""
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return {
        "sensors": t(d["sensors"][idx], torch.float32),
        "readings": t(d["readings"][idx], torch.float32),
        "keep": t(d["keep"][idx], torch.bool),
        "times": t(d["times"], torch.float32),
        "u": t(d["u"][idx], torch.float32),
        "D": t(d["D"][idx], torch.float32),
        "sigma": t(d["sigma"][idx], torch.float32),
        "n_sensors": t(d["n_sensors"][idx], torch.float32),
    }


def load_model(cfg, ckpt_path, device):
    model = DeepSetsLocalizer(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def heatmaps(model, d, device, batch_size=512):
    """Softmax heatmaps for a whole split -> (S, n_cells) float32 numpy."""
    S = len(d["ids"])
    out = []
    for lo in range(0, S, batch_size):
        idx = np.arange(lo, min(lo + batch_size, S))
        logits = model(to_batch(d, idx, device))
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)
