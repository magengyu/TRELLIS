"""Networks preserve the original TRELLIS state-dict names."""
import json
from pathlib import Path
from safetensors.torch import load_file, save_file
from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
from .sparse_structure_flow import SparseStructureFlowModel

CLASSES = {cls.__name__: cls for cls in (
    SparseStructureEncoder, SparseStructureDecoder, SparseStructureFlowModel)}


def build_model(spec, training=False):
    args = dict(spec['args'])
    if training:
        args['use_fp16'] = False  # AMP uses FP32 master parameters.
    return CLASSES[spec['name']](**args)


def load_model(prefix, device='cpu', training=False):
    prefix = Path(prefix)
    spec = json.loads(prefix.with_suffix('.json').read_text())
    model = build_model(spec, training=training)
    model.load_state_dict(load_file(str(prefix.with_suffix('.safetensors'))), strict=True)
    if str(device) == 'cpu' and hasattr(model, 'convert_to_fp32'):
        model.convert_to_fp32()
    return model.to(device).eval()


def save_model(model, spec, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(spec, indent=2) + '\n')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(prefix.with_suffix('.safetensors')))
