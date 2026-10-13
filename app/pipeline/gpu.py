"""Shared GPU helpers.

Memory strategy (12 GB VRAM shared with other containers, 16 GB system RAM):
every stage loads its model straight onto the GPU, runs, and frees it before
the next stage starts. Nothing large is ever parked in system RAM.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", "/models"))


def download_file(url: str, subdir: str) -> Path:
    """Download a plain URL (e.g. an upscaler .pth) into MODELS_DIR once."""
    dest = models_dir() / subdir / Path(urlparse(url).path).name
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        torch.hub.download_url_to_file(url, str(tmp), progress=False)
        tmp.replace(dest)
    return dest


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) tensor in [-1, 1] -> (H, W, 3) uint8."""
    t = (t[0].float() / 2 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
