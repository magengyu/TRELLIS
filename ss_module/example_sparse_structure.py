"""Generate 64^3 occupancy using the standalone ss_module package."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',nargs='+')
    parser.add_argument('--model',default=str(Path(__file__).resolve().parent/'weights'))
    parser.add_argument('--output',default='occupancy.npz')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--cfg-strength',type=float,default=7.5)
    parser.add_argument('--mode',choices=['stochastic','multidiffusion'],default='stochastic')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--no-rembg',action='store_true')
    args=parser.parse_args()
    from PIL import Image
    from ss_module.pipelines import OccupancyPipeline
    from ss_module.utils.occupancy import save_result
    images=[]
    for path in args.images:
        with Image.open(path) as image:images.append(image.copy())
    pipe=OccupancyPipeline(args.model,args.device,remove_background=not args.no_rembg)
    save_result(pipe.generate(images=images,steps=args.steps,seed=args.seed,
                             cfg_strength=args.cfg_strength,mode=args.mode),args.output)


if __name__=='__main__':
    main()
