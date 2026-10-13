"""Depth encoding and result-file writing (numpy + Pillow only, no GPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SKYBOX_PNG = "skybox.png"
SKYBOX_JPG = "skybox.jpg"
DEPTH_PNG = "depth.png"
DEPTH_R16 = "depth.r16"

# How Unity (or anything else) turns an encoded value v in [0, 1] back into a distance.
_DECODE_FORMULA = {
    "linear": "distance = far - v * (far - near)",
    "inverse": "distance = 1 / (1/far + v * (1/near - 1/far))",
    "log": "distance = far * pow(near / far, v)",
}


def encode_depth(
    distance: np.ndarray,
    valid: np.ndarray,
    mode: str,
    near_percentile: float,
    far_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a distance map to uint16 with near = white (65535) and sky = black (0).

    `distance` is the distance from the camera along each pixel's ray (not
    planar z-depth), which is what a vertex shader needs to push a skybox
    sphere's vertices along their direction. `valid` is False for sky only;
    non-sky pixels with a non-finite distance are encoded as "far" (1).
    Real geometry is always >= 1 so it can be told apart from sky.
    """
    distance = distance.astype(np.float64)
    valid = valid.astype(bool)
    geometry = valid & np.isfinite(distance) & (distance > 0)
    samples = distance[geometry]
    if samples.size == 0:
        meta = {"mode": mode, "near_distance": 0.0, "far_distance": 0.0}
        return valid.astype(np.uint16), _depth_meta(meta)

    near = float(np.percentile(samples, near_percentile))
    far = float(np.percentile(samples, far_percentile))
    far = max(far, near * 1.001 + 1e-6)
    x = np.where(geometry, np.clip(np.nan_to_num(distance, nan=far, posinf=far), near, far), far)

    if mode == "linear":
        v = (far - x) / (far - near)
    elif mode == "inverse":
        v = (1.0 / x - 1.0 / far) / (1.0 / near - 1.0 / far)
    elif mode == "log":
        v = (np.log(far) - np.log(x)) / (np.log(far) - np.log(near))
    else:
        raise ValueError(f"unknown depth mode {mode!r}")

    encoded = np.round(np.clip(v, 0.0, 1.0) * 65535.0).astype(np.uint16)
    encoded = np.where(valid, np.maximum(encoded, 1), 0).astype(np.uint16)
    return encoded, _depth_meta({"mode": mode, "near_distance": round(near, 4), "far_distance": round(far, 4)})


def _depth_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "encoding": "uint16",
        "near_is_white": True,
        "sky_value": 0,
        "units": "approx. meters (relative scale)",
        "decode": _DECODE_FORMULA.get(meta["mode"], ""),
        **meta,
    }


def write_outputs(
    pano: np.ndarray,
    depth: np.ndarray,
    out_dir: Path,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Write skybox (PNG + JPG) and depth (16-bit PNG + raw R16) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if pano.dtype != np.uint8 or pano.ndim != 3:
        raise ValueError("panorama must be HxWx3 uint8")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("depth must be HxW uint16")

    img = Image.fromarray(pano)
    img.save(out_dir / SKYBOX_PNG, compress_level=6)
    img.save(out_dir / SKYBOX_JPG, quality=jpeg_quality, subsampling=0, optimize=True)

    # Grayscale 16-bit PNG (Pillow maps a 2-D uint16 array to mode "I;16").
    Image.fromarray(np.ascontiguousarray(depth)).save(out_dir / DEPTH_PNG)

    # Raw little-endian uint16 in Unity row order (bottom row first), ready for
    # Texture2D(w, h, TextureFormat.R16, false, true).LoadRawTextureData(bytes).
    raw = np.ascontiguousarray(np.flipud(depth)).astype("<u2", copy=False)
    (out_dir / DEPTH_R16).write_bytes(raw.tobytes())

    return {
        "width": int(pano.shape[1]),
        "height": int(pano.shape[0]),
        "depth_width": int(depth.shape[1]),
        "depth_height": int(depth.shape[0]),
    }
