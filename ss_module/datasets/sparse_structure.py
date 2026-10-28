"""Binary occupancy grids used to train SS VAE."""
from .components import ManifestDatasetBase
from ..utils.data import load_occupancy, resolve


class SparseStructureDataset(ManifestDatasetBase):
    def __getitem__(self, index):
        row = self.rows[index]
        return {'ss': load_occupancy(resolve(self.root, row['occupancy']), self.config['resolution'])}
