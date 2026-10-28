"""Dense-only TRELLIS utilities; no sparse CUDA extensions are imported."""
import torch.nn as nn

PRIMITIVES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
              nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def convert_module_to_f16(module):
    if isinstance(module, PRIMITIVES):
        module.half()


def convert_module_to_f32(module):
    if isinstance(module, PRIMITIVES):
        module.float()


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
