"""Panoramic distance estimation with MoGe-2.

The panorama is split into 12 perspective views (icosahedron directions, 90°
FOV), MoGe-2 predicts a point map per view, and MoGe's Poisson merge stitches
the per-view distances back into one seamless equirectangular distance map.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch

from .gpu import DEVICE, free_gpu

VIEW_RESOLUTION = 512
BATCH = 4


def estimate_distance(
    pano: np.ndarray,
    sky: np.ndarray,
    model_id: str,
    merge_width: int,
    on_progress: Callable[[float], None],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distance, not_sky) at merge resolution.

    distance is in approx. meters; solid pixels MoGe could not resolve are +inf
    (treated as "very far", just in front of the sky). not_sky is False for sky.
    """
    import utils3d_moge as utils3d
    from moge.model.v2 import MoGeModel
    from moge.utils.panorama import get_panorama_cameras, merge_panorama_depth, split_panorama_image

    extrinsics, intrinsics = get_panorama_cameras()
    views = split_panorama_image(pano, extrinsics, intrinsics, VIEW_RESOLUTION)
    sky_views = split_panorama_image(sky.astype(np.uint8) * 255, extrinsics, intrinsics, VIEW_RESOLUTION)

    model = MoGeModel.from_pretrained(model_id).to(DEVICE).eval()
    distances: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for i in range(0, len(views), BATCH):
                batch = torch.from_numpy(np.stack(views[i : i + BATCH])).to(DEVICE)
                batch = batch.permute(0, 3, 1, 2).float().div_(255)
                fov_x, _ = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(intrinsics[i : i + BATCH])))
                out = model.infer(
                    batch,
                    fov_x=torch.tensor(fov_x, dtype=torch.float32, device=DEVICE),
                    apply_mask=False,
                    use_fp16=True,
                )
                distances.extend(out["points"].float().norm(dim=-1).cpu().numpy())
                masks.extend(out["mask"].cpu().numpy())
                on_progress(0.5 * min(1.0, (i + BATCH) / len(views)))
    finally:
        del model
        free_gpu()

    # Sky never constrains the merge (MoGe ORs the masks of all views).
    masks = [m & (s < 128) for m, s in zip(masks, sky_views)]

    width = min(merge_width, pano.shape[1])
    distance, solid = merge_panorama_depth(width, width // 2, distances, masks, extrinsics, intrinsics)
    on_progress(1.0)
    not_sky = resize_equirect(sky.astype(np.uint8), width, width // 2, nearest=True) == 0
    solid = solid.astype(bool) & not_sky
    distance = distance.astype(np.float32) * _metric_scale(distance, solid, distances, masks)
    distance = np.where(solid, distance, np.inf).astype(np.float32)
    return distance, not_sky


def _metric_scale(merged, valid, view_distances, view_masks) -> float:
    """The Poisson merge loses absolute scale; match its solid-angle-weighted log mean to the views'."""
    height, width = merged.shape
    ok = valid & (merged > 0)
    if ok.sum() < 100:
        return 1.0
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    w_pano = np.broadcast_to(np.cos(lat)[:, None], (height, width))[ok]
    log_pano = np.sum(w_pano * np.log(merged[ok])) / np.sum(w_pano)

    n = view_distances[0].shape[0]
    uv = (np.arange(n) + 0.5) / n * 2 - 1
    u, v = np.meshgrid(uv, uv)
    w_view = (1 + u**2 + v**2) ** -1.5  # solid angle per pixel of a 90° pinhole view
    num = den = 0.0
    for d, m in zip(view_distances, view_masks):
        good = m & np.isfinite(d) & (d > 0)
        num += float(np.sum(w_view[good] * np.log(d[good])))
        den += float(np.sum(w_view[good]))
    if den == 0:
        return 1.0
    return float(np.exp(num / den - log_pano))


def resize_equirect(arr: np.ndarray, width: int, height: int, nearest: bool = False) -> np.ndarray:
    """Resize with wrap-around at the left/right edge (no seam) and clamping at the poles."""
    src_h, src_w = arr.shape[:2]
    xs = ((np.arange(width) + 0.5) * src_w / width - 0.5).astype(np.float32)
    ys = np.clip((np.arange(height) + 0.5) * src_h / height - 0.5, 0, src_h - 1).astype(np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return cv2.remap(
        arr,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
