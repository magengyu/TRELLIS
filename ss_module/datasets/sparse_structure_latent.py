"""Cached mean latents and image features used to train SS Flow."""
import numpy as np
import torch
from .components import ManifestDatasetBase
from ..utils.data import load_array, resolve


class ImageConditionedSparseStructureLatentDataset(ManifestDatasetBase):
    def __getitem__(self, index):
        row = self.rows[index]
        spec = self.config['models']['denoiser']['args']
        latent = load_array(resolve(self.root, row['latent']), ('latent', 'mean'))
        if latent.shape != (spec['in_channels'], *([spec['resolution']] * 3)):
            raise ValueError(f'Latent shape {latent.shape} does not match Flow config')
        cond = load_array(resolve(self.root, row['cond']), ('cond',))
        if cond.ndim == 3:
            # Deterministic per-epoch view selection; independent of worker RNG.
            rng = np.random.default_rng(self.config['seed'] + self.epoch * len(self) + index)
            cond = cond[rng.integers(len(cond))]
        if cond.ndim != 2 or not cond.shape[0] or cond.shape[-1] != spec['cond_channels']:
            raise ValueError(f'Expected condition [tokens,{spec["cond_channels"]}], got {cond.shape}')
        return {'latent': torch.from_numpy(latent.astype(np.float32)),
                'cond': torch.from_numpy(cond.astype(np.float32))}
