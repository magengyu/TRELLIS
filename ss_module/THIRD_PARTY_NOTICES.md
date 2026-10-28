# Attribution

The sparse structure network definitions and dense transformer, attention,
normalization and spatial components are adapted from Microsoft TRELLIS,
copyright Microsoft Corporation, under the MIT license included in LICENSE.
State-dict names are retained for checkpoint interoperability. Sparse extensions
were removed and attention uses native PyTorch SDPA.

DINOv2 is an optional external image encoder loaded through Torch Hub. Its source,
weights and dependencies retain their own licenses. The optional rembg background
removal model also retains its upstream license. Pretrained weights copied by
ss-export retain the terms distributed with their original model repository.
