"""Standalone image-to-occupancy inference pipeline."""
import torch
from ..models import load_model
from ..utils.image import ImageConditioner
from ..utils.checkpoint import model_prefix
from ..utils.occupancy import occupancy_result
from .samplers import sample_flow


class OccupancyPipeline:
    def __init__(self, weights, device='cuda', decoder_weights=None, image_model='dinov2_vitl14_reg',
                 dino_repo=None, remove_background=True):
        self.device = torch.device(device)
        self.flow = load_model(model_prefix(weights, 'denoiser'), device)
        self.decoder = load_model(model_prefix(decoder_weights or weights, 'decoder'), device)
        self.conditioner = ImageConditioner(device, image_model, dino_repo, remove_background)

    @torch.inference_mode()
    def generate(self, images=None, cond=None, steps=12, seed=1, cfg_strength=7.5,
                 mode='stochastic', progress=None, cfg_interval=(.5, 1.), rescale_t=3.):
        if (images is None) == (cond is None):
            raise ValueError('Provide either images or precomputed cond')
        if images is not None:
            prepared = [self.conditioner.preprocess(image) for image in images]
            cond = self.conditioner.encode(prepared)
        latent = sample_flow(self.flow, cond, steps=steps, seed=seed, cfg_strength=cfg_strength,
                             mode=mode, progress=progress, cfg_interval=cfg_interval, rescale_t=rescale_t)
        return occupancy_result(self.decoder(latent))


