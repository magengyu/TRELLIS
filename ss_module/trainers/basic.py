"""Single-process CPU/CUDA training, AMP, EMA, accumulation, resume and validation."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..datasets import build_dataset
from .vae import vae_loss
from .flow_matching import flow_loss
from ..models import build_model, save_model
from ..utils.random import rng_state, restore_rng


def train(config, manifest, output, device='cuda', resume=None, init=None, val_manifest=None):
    cfg = copy.deepcopy(config)
    defaults = dict(resolution=64, batch_size=1, accumulation_steps=1, max_steps=100000,
                    lr=1e-4, weight_decay=0., precision='fp32', grad_clip=1., ema_decay=0.,
                    save_every=1000, log_every=10, seed=42, num_workers=0)
    cfg = {**defaults, **cfg}
    if cfg['stage'] not in ('vae', 'flow') or cfg['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Invalid stage or precision')
    expected_roles = {'encoder', 'decoder'} if cfg['stage'] == 'vae' else {'denoiser'}
    if set(cfg['models']) != expected_roles:
        raise ValueError(f'{cfg["stage"]} requires model roles {sorted(expected_roles)}')
    for key in ('batch_size', 'accumulation_steps', 'max_steps', 'save_every', 'log_every'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if not 0 <= cfg['ema_decay'] < 1:
        raise ValueError('ema_decay must be in [0,1)')
    if cfg['lr'] <= 0 or cfg['grad_clip'] <= 0 or cfg['num_workers'] < 0:
        raise ValueError('Invalid learning rate, gradient clipping or worker count')
    if cfg['stage'] == 'flow' and not (0 <= cfg['p_uncond'] <= 1 and 0 <= cfg['sigma_min'] < 1):
        raise ValueError('Invalid unconditional probability or minimum noise')
    device = torch.device(device)
    if device.type != 'cuda' and cfg['precision'] != 'fp32':
        raise ValueError('Use precision=fp32 for CPU training')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use fp16 or fp32')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    networks = {name: build_model(spec, training=True).to(device).train()
                for name, spec in cfg['models'].items()}
    if resume and init:
        raise ValueError('Use either --resume or --init, not both')
    if init:
        from ..models import load_model
        names = {'encoder': 'ss_enc_conv3d_16l8_fp16', 'decoder': 'ss_dec_conv3d_16l8_fp16',
                 'denoiser': 'ss_flow_img_dit_L_16l8_fp16'}
        for name, model in networks.items():
            path = Path(init) / name
            if not path.with_suffix('.json').exists():
                path = Path(init) / names[name]
            pretrained = load_model(path, training=True)
            model.load_state_dict(pretrained.state_dict(), strict=True)
            del pretrained
    parameters = [p for model in networks.values() for p in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    ema = {name: {k: v.detach().clone() for k, v in model.state_dict().items()}
           for name, model in networks.items()} if cfg['ema_decay'] else None
    step, epoch, batch_offset = 0, 0, 0
    if resume:
        # Full training checkpoints contain optimizer and Python/NumPy RNG states.
        # Only resume checkpoints you trust; inference weights use safetensors.
        state = torch.load(resume, map_location='cpu', weights_only=False)
        for key in ('stage', 'models', 'batch_size', 'accumulation_steps', 'seed', 'precision',
                    'resolution', 'lr', 'weight_decay', 'ema_decay', 'sigma_min', 'p_uncond',
                    't_mean', 't_std', 'lambda_kl', 'loss_type'):
            if state['config'].get(key) != cfg.get(key):
                raise ValueError(f'Resume config mismatch: {key}; use --init for fine-tuning')
        for name, model in networks.items():
            model.load_state_dict(state['models'][name], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        if ema is not None:
            ema = {name: {k: v.to(device) for k, v in weights.items()}
                   for name, weights in state['ema'].items()}
        step, epoch, batch_offset = state['step'], state['epoch'], state['batch_offset']
        restore_rng(state['rng'])
        del state
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    dataset = build_dataset(manifest, cfg['stage'], cfg)
    if len(dataset) < cfg['batch_size']:
        raise ValueError('Dataset must contain at least batch_size records')
    validation = build_dataset(val_manifest, cfg['stage'], cfg) if val_manifest else None
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[cfg['precision']]

    def loader_for_epoch():
        dataset.epoch = epoch
        # A separate generator makes resume independent of global model RNG.
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                          num_workers=cfg['num_workers'], generator=generator,
                          pin_memory=device.type == 'cuda')

    iterator = iter(loader_for_epoch())
    for _ in range(batch_offset):
        next(iterator)

    def next_batch():
        nonlocal iterator, epoch, batch_offset
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1; batch_offset = 0
            iterator = iter(loader_for_epoch()); batch = next(iterator)
        batch_offset += 1
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def losses(batch):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg['precision'] != 'fp32'):
            if cfg['stage'] == 'vae':
                return vae_loss(networks['encoder'], networks['decoder'], batch['ss'],
                                cfg.get('loss_type', 'dice'), cfg.get('lambda_kl', .001))
            return flow_loss(networks['denoiser'], batch['latent'], batch['cond'],
                             **{k: cfg[k] for k in ('sigma_min', 'p_uncond', 't_mean', 't_std')})

    def save():
        state = dict(config=cfg, step=step, epoch=epoch, batch_offset=batch_offset,
                     models={name: model.state_dict() for name, model in networks.items()},
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), ema=ema, rng=rng_state())
        temporary = output / 'latest.tmp'
        torch.save(state, temporary); temporary.replace(output / 'latest.pt')
        # Raw and EMA exports are directly consumable by the inference CLI.
        for name, model in networks.items():
            save_model(model, cfg['models'][name], output / 'weights' / name)
            if ema is not None:
                raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema[name])
                save_model(model, cfg['models'][name], output / 'ema_weights' / name)
                model.load_state_dict(raw)

    while step < cfg['max_steps']:
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for _ in range(cfg['accumulation_steps']):
            terms = losses(next_batch())
            if not torch.isfinite(terms['loss']):
                raise FloatingPointError(f'Non-finite loss at step {step + 1}')
            scaler.scale(terms['loss'] / cfg['accumulation_steps']).backward()
            for key, value in terms.items():
                metrics[key] = metrics.get(key, 0.) + value.detach().item() / cfg['accumulation_steps']
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'])
        if not torch.isfinite(norm):
            if not scaler.is_enabled():
                raise FloatingPointError('Non-finite gradients; lower the learning rate or use fp32')
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        updated = scaler.get_scale() >= old_scale
        if ema is not None and updated:
            with torch.no_grad():
                for name, model in networks.items():
                    for key, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema[name][key].lerp_(value, 1 - cfg['ema_decay'])
                        else:
                            ema[name][key].copy_(value)
        step += 1
        if step % cfg['log_every'] == 0:
            record = dict(step=step, epoch=epoch, optimizer_updated=updated, **metrics)
            print(json.dumps(record), flush=True)
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        if step % cfg['save_every'] == 0 or step == cfg['max_steps']:
            if validation:
                saved_rng = rng_state()
                for model in networks.values():model.eval()
                totals, count = {}, 0
                with torch.no_grad():
                    for batch in DataLoader(validation, batch_size=cfg['batch_size']):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        size = len(next(iter(batch.values())))
                        for key, value in losses(batch).items():
                            totals[key] = totals.get(key, 0.) + value.item() * size
                        count += size
                record = dict(step=step, validation={k: v/count for k,v in totals.items()})
                print(json.dumps(record), flush=True)
                with (output / 'metrics.jsonl').open('a') as f:f.write(json.dumps(record) + '\n')
                for model in networks.values():model.train()
                restore_rng(saved_rng)
            save()
    return output / 'latest.pt'


