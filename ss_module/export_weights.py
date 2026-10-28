"""Copy the three SS checkpoints into a portable directory (no symlinks)."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import shutil
from ss_module.utils.checkpoint import NAMES, model_prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Local original model/ckpts directory')
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    # Validate all sources before copying any large files.
    prefixes={role:model_prefix(args.source,role) for role in NAMES}
    for prefix in prefixes.values():
        for suffix in ('.json','.safetensors'):
            if not prefix.with_suffix(suffix).is_file():raise FileNotFoundError(prefix.with_suffix(suffix))
    for role,prefix in prefixes.items():
        for suffix in ('.json','.safetensors'):
            source=prefix.with_suffix(suffix).resolve();destination=output/(role+suffix)
            if source!=destination:
                temporary=destination.with_name(destination.name+'.tmp')
                shutil.copy2(source,temporary);temporary.replace(destination)
        print(f'Exported {role}',flush=True)
    (output/'weights_manifest.json').write_text(json.dumps(dict(
        format='TRELLIS SS state dict',roles=list(prefixes),
        image_model='dinov2_vitl14_reg',resolution=64),indent=2)+'\n')


if __name__ == '__main__':
    main()
