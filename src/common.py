"""Shared utilities: config loading, grid conventions, seeding, device."""
import hashlib
import json
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
    """Resolve a paths: entry relative to the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------- grid
# Convention (used identically by model targets, conformal, oracle, figures):
#   ix = clip(floor(x * n), 0, n-1), iy = clip(floor(y * n), 0, n-1)
#   cell = iy * n + ix
#   center(cell) = ((ix + 0.5) / n, (iy + 0.5) / n)

def pos_to_cell(pos, n):
    """pos: (..., 2) array in [0,1]^2 -> int64 cell ids."""
    pos = np.asarray(pos)
    idx = np.clip(np.floor(pos * n).astype(np.int64), 0, n - 1)
    return idx[..., 1] * n + idx[..., 0]


def cell_centers(n, dtype=np.float64):
    """(n*n, 2) array of cell centers, ordered by cell id (iy*n + ix)."""
    c = (np.arange(n, dtype=dtype) + 0.5) / n
    xx, yy = np.meshgrid(c, c)          # row = iy, col = ix
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def cell_center_of(cells, n):
    cells = np.asarray(cells)
    ix = cells % n
    iy = cells // n
    return np.stack([(ix + 0.5) / n, (iy + 0.5) / n], axis=-1)


# ---------------------------------------------------------------- seeding

def scenario_rng(root_entropy, split_tag, index):
    """Deterministic per-scenario RNG. split_tag is a stable small int."""
    ss = np.random.SeedSequence([root_entropy, split_tag, int(index)])
    return np.random.default_rng(ss)


SPLIT_TAGS = {"train": 0, "val": 1, "calib": 2, "test": 3, "train_xl": 4,
              "dose_base": 10, "dose_geometry": 11, "dose_sigma": 12}


# ---------------------------------------------------------------- io

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def update_json(path, updates):
    """Merge updates into a JSON file (nested one level)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return data
