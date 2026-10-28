from .sparse_structure import SparseStructureDataset
from .sparse_structure_latent import ImageConditionedSparseStructureLatentDataset


def build_dataset(manifest, stage, config):
    classes = {'vae': SparseStructureDataset, 'flow': ImageConditionedSparseStructureLatentDataset}
    return classes[stage](manifest, config)
