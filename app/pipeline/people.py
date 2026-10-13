"""Find people in the panorama (Mask R-CNN, wrap-aware tiles) and plan inpainting crops."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .flux import Region
from .gpu import DEVICE, free_gpu

PERSON = 1  # COCO label id
MIN_PIXELS = 48  # ignore specks


def detect_people(pano: np.ndarray, score: float) -> np.ndarray:
    """Boolean (H, W) mask of people. Square tiles overlap by 50% and wrap past the seam."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, box_score_thresh=score
    ).eval().to(DEVICE)
    height, width = pano.shape[:2]
    tile = height
    mask = np.zeros((height, width), dtype=bool)
    try:
        with torch.inference_mode():
            for x0 in range(0, width, tile // 2):
                cols = np.arange(x0, x0 + tile) % width
                crop = torch.from_numpy(np.ascontiguousarray(pano[:, cols])).to(DEVICE)
                crop = crop.permute(2, 0, 1).float().div_(255)
                out = model([crop])[0]
                keep = (out["labels"] == PERSON) & (out["scores"] >= score)
                if keep.any():
                    hits = (out["masks"][keep, 0] >= 0.5).any(dim=0).cpu().numpy()
                    mask[:, cols] |= hits
    finally:
        del model
        free_gpu()
    return mask


def plan_regions(mask: np.ndarray, max_regions: int, max_side: int = 1024) -> list[Region]:
    """Group detections into square crops (multiples of 16 px) with feathered masks."""
    height, width = mask.shape
    if mask.sum() < MIN_PIXELS:
        return []

    grow = max(6, height // 80)
    # Dilate with horizontal wrap by padding the seam before filtering.
    padded = np.concatenate([mask[:, -grow:], mask, mask[:, :grow]], axis=1)
    padded = ndimage.binary_dilation(padded, structure=np.ones((3, 3), bool), iterations=grow)
    grown = padded[:, grow:-grow]

    labels, count = ndimage.label(grown)
    if count == 0:
        return []
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, count + 1))
    order = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] >= MIN_PIXELS]
    slices = ndimage.find_objects(labels)

    max_side = min(max_side, height) // 16 * 16
    regions: list[Region] = []
    covered: set[int] = set()
    for label in order:
        if label in covered:
            continue
        if len(regions) >= max_regions:
            break
        ys, xs = slices[label - 1]
        extent = max(ys.stop - ys.start, xs.stop - xs.start)
        size = int(np.clip(np.ceil(extent * 2.0 / 16) * 16, 512, max_side))
        cy, cx = (ys.start + ys.stop) // 2, (xs.start + xs.stop) // 2
        y0 = int(np.clip(cy - size // 2, 0, height - size))
        x0 = cx - size // 2
        # Every other person that fits completely inside this crop is painted out in the same pass.
        members = [
            other for other in order
            if other not in covered and _inside(slices[other - 1], y0, x0, size, width)
        ]
        covered.update(members)
        covered.add(label)
        cols = np.arange(x0, x0 + size) % width
        local = np.isin(labels[y0 : y0 + size][:, cols], members + [label]).astype(np.float32)
        feather = ndimage.gaussian_filter(local, sigma=grow / 2)
        soft = np.clip(np.maximum(local, feather * 1.5), 0, 1)
        regions.append(Region(y0=y0, x0=int(x0), size=size, mask=soft))
    return regions


def _inside(box: tuple[slice, slice], y0: int, x0: int, size: int, width: int) -> bool:
    """Does a component's bounding box lie within the (horizontally wrapping) square crop?"""
    ys, xs = box
    margin = 8  # keep the feathered edge inside the crop
    if ys.start < y0 + margin or ys.stop > y0 + size - margin:
        return False
    offset = (xs.start - x0) % width
    return offset >= margin and offset + (xs.stop - xs.start) <= size - margin
