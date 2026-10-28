"""TRELLIS tensor layout adapter for native PyTorch SDPA."""
from torch.nn.functional import scaled_dot_product_attention as sdpa


def scaled_dot_product_attention(*args):
    if len(args) == 1:
        q, k, v = args[0].unbind(dim=2)
    elif len(args) == 2:
        q = args[0]
        k, v = args[1].unbind(dim=2)
    elif len(args) == 3:
        q, k, v = args
    else:
        raise ValueError('Expected qkv, q+kv, or q+k+v')
    return sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
