"""Array I/O and JSONL manifest helpers."""
import json
from pathlib import Path
import numpy as np
import torch


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected a JSON object')
            rows.append(row)
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return path.parent, rows


def resolve(root, value):
    return (Path(root) / value).resolve()


def load_array(path, keys):
    path = Path(path)
    if path.suffix == '.npy':
        result = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as pack:
            key = next((key for key in keys if key in pack), None)
            if key is None:
                raise ValueError(f'{path}: expected one of {keys}, found {pack.files}')
            result = pack[key].copy()
    else:
        raise ValueError(f'{path}: expected .npy or .npz')
    if not np.isfinite(result).all():
        raise ValueError(f'{path}: non-finite values')
    return result


def load_occupancy(path, resolution=64):
    grid = load_array(path, ('occupancy', 'ss'))
    if grid.shape == (1, resolution, resolution, resolution):
        grid = grid[0]
    if grid.shape != (resolution,) * 3:
        raise ValueError(f'{path}: expected {(resolution,) * 3}, got {grid.shape}')
    if not np.isin(grid, [0, 1]).all():
        raise ValueError(f'{path}: occupancy must contain only 0 and 1')
    return torch.from_numpy(grid.astype(np.float32))[None]


