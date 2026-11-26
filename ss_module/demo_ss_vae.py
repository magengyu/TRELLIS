"""验证 SS encoder -> decoder。

在仓库根目录运行：
    python ss_module/demo_ss_vae.py
    python ss_module/demo_ss_vae.py --input occupancy.npz --device cuda
    python ss_module/demo_ss_vae.py --device cpu

默认使用合成球面；实际数据支持二值 .npy 或包含 occupancy/ss 的 .npz。
合成数据用于验证推理链路，重建 IoU 不作为模型质量的通过门槛。
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ss_module.models import load_model
from ss_module.utils.data import load_occupancy


def check_tensor(name, tensor, shape):
    if tuple(tensor.shape) != shape:
        raise RuntimeError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{name}: contains NaN/Inf")
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    ckpts = ROOT / 'ckpts/TRELLIS-image-large/ckpts'
    parser.add_argument('--encoder', type=Path,
                        default=ckpts / 'ss_enc_conv3d_16l8_fp16.json')
    parser.add_argument('--decoder', type=Path,
                        default=ckpts / 'ss_dec_conv3d_16l8_fp16.json')
    parser.add_argument('--input', type=Path, help='64^3 binary occupancy grid')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'ss_module/outputs/ss_vae_demo.npz')
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    for path in (args.encoder, args.decoder):
        for suffix in ('.json', '.safetensors'):
            if not path.with_suffix(suffix).is_file():
                parser.error(f'Missing checkpoint: {path.with_suffix(suffix)}')

    encoder = load_model(args.encoder, device=args.device)
    decoder = load_model(args.decoder, device=args.device)
    print(f'Device: {args.device}; encoder/decoder internal dtype: '
          f'{encoder.dtype}/{decoder.dtype}')
    if args.input:
        x = load_occupancy(args.input, resolution=64).unsqueeze(0)
    else:
        # A thin surface resembles the sparse occupied voxels used by SS VAE.
        axis = torch.arange(64, dtype=torch.float32) - 31.5
        radius = sum(a.square() for a in torch.meshgrid(axis, axis, axis, indexing='ij')).sqrt()
        x = ((radius - 20).abs() < 0.75).float()[None, None]
    # The models keep input/output layers in FP32 even with an FP16 torso.
    x = x.to(device=args.device, dtype=torch.float32)
    with torch.inference_mode():
        check_tensor('input', x, (1, 1, 64, 64, 64))
        z, mean, logvar = encoder(x, sample_posterior=False, return_raw=True)
        for name, value in (('latent', z), ('mean', mean), ('logvar', logvar)):
            check_tensor(name, value, (1, 8, 16, 16, 16))
        logits = decoder(z)
        check_tensor('decoded logits', logits, tuple(x.shape))
        prediction, target = logits > 0, x > 0.5
        intersection = (prediction & target).sum().item()
        union = (prediction | target).sum().item()
        iou = intersection / union if union else 1.0
        bce = F.binary_cross_entropy_with_logits(logits, x).item()
        print(f'Occupied voxels: input={target.sum().item()}, '
              f'reconstructed={prediction.sum().item()}')
        print(f'Reconstruction: IoU={iou:.6f}, BCE={bce:.6f}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output, input=x[0, 0].cpu().numpy(),
            occupancy=prediction[0, 0].cpu().numpy(),
            logits=logits[0, 0].cpu().numpy(), latent=z.cpu().numpy(),
            mean=mean.cpu().numpy(), logvar=logvar.cpu().numpy(), iou=iou, bce=bce,
        )
    print(f'PASS: encoder/decoder forward validation completed. Saved {args.output}')


if __name__ == '__main__':
    main()
