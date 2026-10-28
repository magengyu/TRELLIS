import torch
from torch.nn import functional as F


def vae_loss(encoder, decoder, occupancy, loss_type='dice', lambda_kl=0.001):
    z, mean, logvar = encoder(occupancy, sample_posterior=True, return_raw=True)
    logits = decoder(z).float()
    if loss_type == 'dice':
        prob = logits.sigmoid()
        reconstruction = 1 - (2 * (prob * occupancy).sum() + 1) / (prob.sum() + occupancy.sum() + 1)
    elif loss_type == 'bce':
        reconstruction = F.binary_cross_entropy_with_logits(logits, occupancy)
    else:
        raise ValueError('loss_type must be dice or bce')
    kl = 0.5 * (mean.float().square() + logvar.float().exp() - logvar.float() - 1).mean()
    return {'loss': reconstruction + lambda_kl * kl, 'reconstruction': reconstruction, 'kl': kl}


