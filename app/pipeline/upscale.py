"""Real-ESRGAN upscaling (via spandrel) that stays seamless across the equirect wrap."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .gpu import DEVICE, download_file, free_gpu

WRAP_PAD = 32
TILE = 768
OVERLAP = 48


def upscale_pano(pano: np.ndarray, model_url: str, out_width: int) -> np.ndarray:
    out_height = out_width // 2
    if pano.shape[1] >= out_width:
        return _resize(pano, out_width, out_height)

    from spandrel import ImageModelDescriptor, ModelLoader

    model = ModelLoader(device=DEVICE).load_from_file(str(download_file(model_url, "upscalers")))
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(f"{model_url} is not a single-image upscaler")
    model.eval()
    dtype = torch.float16 if model.supports_half else torch.float32
    if dtype == torch.float16:
        model.half()
    try:
        while pano.shape[1] < out_width:
            pano = _upscale_wrapped(model, pano, dtype)
    finally:
        del model
        free_gpu()
    return _resize(pano, out_width, out_height)


def _upscale_wrapped(model, pano: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    scale = model.scale
    width = pano.shape[1]
    padded = np.concatenate([pano[:, -WRAP_PAD:], pano, pano[:, :WRAP_PAD]], axis=1)
    out = _tiled(model, padded, scale, dtype)
    return out[:, WRAP_PAD * scale : (WRAP_PAD + width) * scale]


def _tiled(model, image: np.ndarray, scale: int, dtype: torch.dtype) -> np.ndarray:
    height, width = image.shape[:2]
    src = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
    out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    with torch.inference_mode():
        for y in range(0, height, TILE):
            for x in range(0, width, TILE):
                y0, y1 = max(0, y - OVERLAP), min(height, y + TILE + OVERLAP)
                x0, x1 = max(0, x - OVERLAP), min(width, x + TILE + OVERLAP)
                tile = src[:, :, y0:y1, x0:x1].to(DEVICE, dtype).div_(255)
                result = model(tile)
                th, tw = min(TILE, height - y), min(TILE, width - x)
                oy, ox = (y - y0) * scale, (x - x0) * scale
                core = result[0, :, oy : oy + th * scale, ox : ox + tw * scale]
                core = (core.float().clamp_(0, 1) * 255).round_().to(torch.uint8)
                out[y * scale : (y + th) * scale, x * scale : (x + tw) * scale] = core.permute(1, 2, 0).cpu().numpy()
    return out


def _resize(pano: np.ndarray, width: int, height: int) -> np.ndarray:
    if pano.shape[1] == width and pano.shape[0] == height:
        return pano
    return np.asarray(Image.fromarray(pano).resize((width, height), Image.Resampling.LANCZOS))
