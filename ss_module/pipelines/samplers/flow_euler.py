"""Euler flow sampling with TRELLIS-compatible interval CFG."""
import math
import torch
from tqdm import tqdm


@torch.no_grad()
def sample_flow(model, cond, steps=12, seed=1, cfg_strength=7.5,
                cfg_interval=(0.5, 1.), rescale_t=3., mode='stochastic',
                num_samples=1, progress=None, verbose=True):
    if steps < 1 or num_samples < 1 or mode not in ('stochastic', 'multidiffusion'):
        raise ValueError('Invalid sampling steps, sample count or mode')
    if not math.isfinite(rescale_t) or rescale_t <= 0 or not math.isfinite(cfg_strength):
        raise ValueError('Invalid time rescaling or CFG strength')
    if not 0 <= cfg_interval[0] <= cfg_interval[1] <= 1:
        raise ValueError('Invalid CFG interval')
    if cond.ndim != 3 or len(cond) == 0 or cond.shape[1] == 0 or cond.shape[-1] != model.cond_channels:
        raise ValueError('Expected condition [views,tokens,cond_channels]')
    if not torch.isfinite(cond).all():
        raise ValueError('Condition features must be finite')
    if mode == 'stochastic' and len(cond) > steps:
        raise ValueError('stochastic mode requires at least one step per image')
    device = model.device
    cond = cond.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_samples, model.in_channels, *([model.resolution] * 3),
                    device=device, generator=generator)
    times = torch.linspace(1, 0, steps + 1, dtype=torch.float64).tolist()
    times = [rescale_t * t / (1 + (rescale_t - 1) * t) for t in times]
    negative = torch.zeros_like(cond[:1]).expand(num_samples, -1, -1)
    for index in tqdm(range(steps), desc='SS Flow', disable=not verbose):
        t, next_t = times[index:index + 2]
        time = torch.full((num_samples,), 1000 * t, device=device)
        views = [index % len(cond)] if mode == 'stochastic' else range(len(cond))
        prediction = None
        for view in views:
            value = model(x, time, cond[view:view+1].expand(num_samples, -1, -1))
            prediction = value if prediction is None else prediction + value
        prediction = prediction / len(views)
        if cfg_interval[0] <= t <= cfg_interval[1] and cfg_strength != 0:
            unconditional = model(x, time, negative)
            prediction = (1 + cfg_strength) * prediction - cfg_strength * unconditional
        x = x - (t - next_t) * prediction
        if progress:
            progress(index + 1, steps)
    return x
