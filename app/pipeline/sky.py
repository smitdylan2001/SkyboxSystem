"""Sky segmentation (ADE20K semantic segmentation) for the equirectangular panorama.

MoGe marks a pixel as solid if *any* of its perspective views thinks so, and
views that look straight up at featureless sky sometimes do. A dedicated sky
mask keeps the sky at "infinitely far" (0 in the depth map) reliably.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .depth import resize_equirect
from .gpu import DEVICE, free_gpu

WORK_HEIGHT = 512  # segmentation runs on a 1024x512 copy of the panorama
TILE = 512
STRIDE = 256


def segment_sky(pano: np.ndarray, model_id: str) -> np.ndarray:
    """Boolean (H, W) sky mask at the panorama's resolution."""
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    height, width = pano.shape[:2]
    work = resize_equirect(pano, WORK_HEIGHT * 2, WORK_HEIGHT)
    work_w = work.shape[1]

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(DEVICE).eval()
    sky_ids = [int(i) for i, name in model.config.id2label.items() if name.strip().lower() == "sky"]
    if not sky_ids:
        raise RuntimeError(f"{model_id} has no 'sky' class")

    prob = np.zeros((WORK_HEIGHT, work_w), np.float32)
    hits = np.zeros(work_w, np.float32)
    try:
        starts = list(range(0, work_w, STRIDE))
        tiles = [work[:, np.arange(x0, x0 + TILE) % work_w] for x0 in starts]
        inputs = processor(images=tiles, return_tensors="pt", do_resize=False).to(DEVICE)
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = F.interpolate(logits, size=(TILE, TILE), mode="bilinear", align_corners=False)
            sky = logits.softmax(dim=1)[:, sky_ids].sum(dim=1).float().cpu().numpy()
        for x0, tile_prob in zip(starts, sky):
            cols = np.arange(x0, x0 + TILE) % work_w
            prob[:, cols] += tile_prob
            hits[cols] += 1
    finally:
        del model
        free_gpu()

    prob /= np.maximum(hits, 1)
    raw = prob > 0.5
    cleaned = _clean(raw)
    # Apply the clean-up to the probabilities, then upsample them smoothly (no blocky edges).
    prob = np.where(cleaned & ~raw, 1.0, np.where(raw & ~cleaned, 0.0, prob)).astype(np.float32)
    return resize_equirect(prob, width, height) > 0.5


def _clean(sky: np.ndarray) -> np.ndarray:
    """Drop small sky specks (e.g. reflections) and fill small holes (birds, wires)."""
    min_area = sky.size // 2000
    labels, count = ndimage.label(sky)
    if count:
        sizes = ndimage.sum(sky, labels, index=range(1, count + 1))
        sky = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= min_area])
    holes, count = ndimage.label(~sky)
    if count:
        sizes = ndimage.sum(~sky, holes, index=range(1, count + 1))
        small = [i + 1 for i, s in enumerate(sizes) if s < min_area // 4]
        sky = sky | np.isin(holes, small)
    return sky
