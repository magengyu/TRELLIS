"""Command-line entry for standalone SS VAE / Flow training."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ss_module.trainers import train


def main(expected_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--init', help='Directory containing exported or official model prefixes')
    parser.add_argument('--val-manifest')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if expected_stage and config['stage'] != expected_stage:
        parser.error(f'This entry point requires stage={expected_stage}')
    if args.max_steps is not None:config['max_steps'] = args.max_steps
    train(config, args.manifest, args.output, args.device, args.resume, args.init, args.val_manifest)


if __name__ == '__main__':
    main()
