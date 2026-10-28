"""Resolve standalone or original model weight prefixes."""
from pathlib import Path

NAMES = dict(encoder='ss_enc_conv3d_16l8_fp16', decoder='ss_dec_conv3d_16l8_fp16',
             denoiser='ss_flow_img_dit_L_16l8_fp16')


def model_prefix(directory, role):
    directory = Path(directory)
    for prefix in (directory/role, directory/NAMES[role], directory/'ckpts'/NAMES[role]):
        if prefix.with_suffix('.json').exists():
            return prefix
    raise FileNotFoundError(f'No {role} weights in {directory}; run ss-export or specify exported weights')


