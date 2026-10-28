"""Base manifest dataset; no original repository dependencies."""
from torch.utils.data import Dataset
from ..utils.data import read_manifest


class ManifestDatasetBase(Dataset):
    def __init__(self, manifest, config):
        self.root, self.rows = read_manifest(manifest)
        self.config, self.epoch = config, 0

    def __len__(self):
        return len(self.rows)
