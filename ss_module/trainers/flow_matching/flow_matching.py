import torch
from torch.nn import functional as F


def flow_loss(model, latent, cond, sigma_min=1e-5, p_uncond=0.1, t_mean=1., t_std=1.):
    noise = torch.randn_like(latent)
    t = (torch.randn(len(latent), device=latent.device) * t_std + t_mean).sigmoid()
    t_grid = t[:, None, None, None, None]
    noisy = (1 - t_grid) * latent + (sigma_min + (1 - sigma_min) * t_grid) * noise
    drop = torch.rand(len(latent), 1, 1, device=latent.device) < p_uncond
    cond = torch.where(drop, torch.zeros_like(cond), cond)
    prediction = model(noisy, t * 1000, cond)
    target = (1 - sigma_min) * noise - latent
    loss = F.mse_loss(prediction.float(), target)
    return {'loss': loss, 'mse': loss}
