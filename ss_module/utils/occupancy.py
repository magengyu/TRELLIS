"""Occupancy result conversion and NPZ export."""
from pathlib import Path
import numpy as np

def occupancy_result(logits):
    logits = logits.detach().float().cpu().numpy()
    if logits.ndim == 5 and logits.shape[:2] == (1, 1):
        logits = logits[0, 0]
    if logits.ndim != 3 or not np.isfinite(logits).all():
        raise ValueError('Expected a finite single occupancy grid')
    occupancy = logits > 0
    return dict(occupancy=occupancy, coords=np.argwhere(occupancy).astype(np.int32), logits=logits)



def save_result(result, output):
    output = Path(output)
    if output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f'Saved {output}: grid={result["occupancy"].shape}, occupied={len(result["coords"])}')
