"""Generate occupancy from photos/features or reconstruct occupancy through a VAE."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import torch
from PIL import Image
from ss_module.utils.data import load_array, load_occupancy
from ss_module.models import load_model
from ss_module.pipelines import OccupancyPipeline
from ss_module.utils.checkpoint import model_prefix
from ss_module.utils.occupancy import occupancy_result, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--weights', required=True)
    gen.add_argument('--decoder-weights')
    source = gen.add_mutually_exclusive_group(required=True)
    source.add_argument('--images', nargs='+')
    source.add_argument('--cond', help='Cached .npy/.npz features [views,tokens,channels]')
    gen.add_argument('--steps', type=int, default=12)
    gen.add_argument('--seed', type=int, default=1)
    gen.add_argument('--cfg-strength', type=float, default=7.5)
    gen.add_argument('--mode', choices=['stochastic', 'multidiffusion'], default='stochastic')
    gen.add_argument('--rescale-t', type=float, default=3.)
    gen.add_argument('--no-rembg', action='store_true')
    gen.add_argument('--image-model', default='dinov2_vitl14_reg')
    gen.add_argument('--dino-repo', help='Optional local DINOv2 source checkout')
    rec = sub.add_parser('reconstruct')
    rec.add_argument('--weights', required=True)
    rec.add_argument('--occupancy', required=True)
    rec.add_argument('--resolution', type=int, default=64)
    for command in (gen, rec):
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with torch.inference_mode():
        if args.command == 'generate':
            pipe = OccupancyPipeline(args.weights, args.device, args.decoder_weights,
                                     args.image_model, args.dino_repo, not args.no_rembg)
            cond, images = None, None
            if args.cond:
                cond = torch.from_numpy(load_array(args.cond, ('cond',))).float()
                if cond.ndim == 2:cond = cond[None]
            else:
                images = []
                for path in args.images:
                    with Image.open(path) as image:images.append(image.copy())
            result = pipe.generate(images=images, cond=cond, steps=args.steps, seed=args.seed,
                                   cfg_strength=args.cfg_strength, mode=args.mode, rescale_t=args.rescale_t)
        else:
            encoder = load_model(model_prefix(args.weights, 'encoder'), args.device)
            decoder = load_model(model_prefix(args.weights, 'decoder'), args.device)
            grid = load_occupancy(args.occupancy, args.resolution)[None].to(args.device)
            result = occupancy_result(decoder(encoder(grid, sample_posterior=False)))
        save_result(result, args.output)


if __name__ == '__main__':
    main()
