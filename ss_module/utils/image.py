"""Optional DINOv2 conditioning; core training can use cached features offline."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F


class ImageConditioner:
    def __init__(self, device='cuda', name='dinov2_vitl14_reg', repo=None, remove_background=True):
        self.device = torch.device(device)
        self.name = name
        self.repo = repo
        self.remove_background = remove_background
        self.model = None
        self.rembg_session = None

    def preprocess(self, image):
        image = ImageOps.exif_transpose(image)
        has_alpha = image.mode == 'RGBA' and np.any(np.array(image.getchannel('A')) < 255)
        if not has_alpha and not self.remove_background:
            return image.convert('RGB').resize((518, 518), Image.Resampling.LANCZOS)
        if not has_alpha:
            try:
                import rembg
            except ImportError as error:
                raise RuntimeError('Install ss-occupancy[image] or use --no-rembg with prepared images') from error
            image = image.convert('RGB')
            scale = min(1, 1024 / max(image.size))
            if scale < 1:
                image = image.resize(tuple(int(size * scale) for size in image.size), Image.Resampling.LANCZOS)
            if self.rembg_session is None:
                self.rembg_session = rembg.new_session('u2net')
            image = rembg.remove(image, session=self.rembg_session)
        array = np.array(image)
        positions = np.argwhere(array[:, :, 3] > 0.8 * 255)
        if not len(positions):
            raise ValueError('No foreground found in image; check transparency/background removal')
        low, high = positions.min(0), positions.max(0)
        cy, cx = (low + high) / 2
        size = max(1, int(max(high - low) * 1.2))
        image = image.crop((cx-size//2, cy-size//2, cx+size//2, cy+size//2))
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
        array = np.array(image).astype(np.float32) / 255
        return Image.fromarray((array[:, :, :3] * array[:, :, 3:4] * 255).astype(np.uint8))

    @torch.no_grad()
    def encode(self, images):
        if self.model is None:
            if self.repo:
                self.model = torch.hub.load(str(Path(self.repo)), self.name, source='local', pretrained=True)
            else:
                self.model = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=True)
            self.model = self.model.eval().to(self.device)
        features = []
        for image in images:
            image = image.convert('RGB').resize((518, 518), Image.Resampling.LANCZOS)
            x = torch.from_numpy(np.array(image).astype(np.float32)/255).permute(2, 0, 1)[None].to(self.device)
            mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
            std = x.new_tensor([.229, .224, .225])[None, :, None, None]
            tokens = self.model((x-mean)/std, is_training=True)['x_prenorm']
            features.append(F.layer_norm(tokens, tokens.shape[-1:]))
        return torch.cat(features)
