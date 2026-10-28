"""Cache VAE mean latents and optional DINOv2 features for Flow training."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ss_module.utils.data import read_manifest, resolve, load_occupancy, load_array
from ss_module.utils.image import ImageConditioner
from ss_module.models import load_model
from ss_module.utils.checkpoint import model_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSONL: occupancy + images (or cond)')
    parser.add_argument('--encoder-weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--image-model', default='dinov2_vitl14_reg')
    parser.add_argument('--dino-repo')
    parser.add_argument('--no-rembg', action='store_true')
    args = parser.parse_args()
    root, rows = read_manifest(args.manifest)
    output = Path(args.output).resolve();output.mkdir(parents=True, exist_ok=True)
    encoder = load_model(model_prefix(args.encoder_weights, 'encoder'), args.device)
    conditioner = ImageConditioner(args.device, args.image_model, args.dino_repo, not args.no_rembg)
    records = []
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc='Caching')):
            grid = load_occupancy(resolve(root, row['occupancy']), args.resolution)[None].to(args.device)
            latent = encoder(grid, sample_posterior=False)[0].float().cpu().numpy()
            if 'cond' in row:
                cond = load_array(resolve(root, row['cond']), ('cond',)).astype(np.float32)
                if cond.ndim == 2:cond = cond[None]
            else:
                if not row.get('images'):
                    raise ValueError(f'Record {index} needs images or cond')
                images = []
                for path in row['images']:
                    with Image.open(resolve(root, path)) as image:
                        images.append(conditioner.preprocess(image))
                cond = conditioner.encode(images).float().cpu().numpy()
            if cond.ndim != 3 or min(cond.shape) < 1 or not np.isfinite(cond).all():
                raise ValueError(f'Invalid condition shape/values in record {index}')
            if not np.isfinite(latent).all():raise ValueError('Non-finite VAE latent')
            np.savez_compressed(output/f'{index:07d}.npz', latent=latent, cond=cond)
            records.append(dict(id=row.get('id', str(index)), latent=f'{index:07d}.npz',
                                cond=f'{index:07d}.npz'))
    temporary = output/'manifest.tmp'
    temporary.write_text(''.join(json.dumps(row)+'\n' for row in records))
    temporary.replace(output/'manifest.jsonl')
    prefix = model_prefix(args.encoder_weights, 'encoder')
    metadata = dict(encoder=str(prefix), encoder_config=json.loads(prefix.with_suffix('.json').read_text()),
                    image_model=args.image_model, remove_background=not args.no_rembg,
                    resolution=args.resolution, latent_policy='posterior_mean', normalized=False)
    # Record the exact encoder weight identity: changing VAE requires recaching Flow targets.
    digest=hashlib.sha256()
    with prefix.with_suffix('.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    metadata['encoder_sha256']=digest.hexdigest()
    (output/'cache_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Cached {len(records)} objects: {output / "manifest.jsonl"}')


if __name__ == '__main__':
    main()
