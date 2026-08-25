import os
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "config" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve(cfg, key):
    if key == "data_dir" and os.environ.get("CSR_DATA_DIR"):
        return Path(os.environ["CSR_DATA_DIR"])
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def pos_to_cell(pos, n):
    pos = np.asarray(pos)
    idx = np.clip(np.floor(pos * n).astype(np.int64), 0, n - 1)
    return idx[..., 1] * n + idx[..., 0]


def cell_centers(n, dtype=np.float64):
    c = (np.arange(n, dtype=dtype) + 0.5) / n
    xx, yy = np.meshgrid(c, c)
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def savez_atomic(path, **arrs):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrs)
    os.replace(tmp, path)
