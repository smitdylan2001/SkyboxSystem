"""GPU-free stand-in pipeline (MOCK_PIPELINE=true).

Renders a synthetic but geometrically exact scene (ground plane with a 2 m
checker grid, a ring of mountains at 150 m, sky) so you can test the API, the
queue and your Unity displacement shader before the real models are set up.
The ground grid lines should look straight and evenly spaced once displaced.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import Settings
from . import JobContext
from .outputs import encode_depth, write_outputs

CAMERA_HEIGHT = 1.7
MOUNTAIN_RADIUS = 150.0


class MockPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prefetch(self) -> None:
        pass

    def release(self) -> None:
        pass

    def run(self, ctx: JobContext) -> dict[str, Any]:
        s = self.settings
        ctx.progress("captioning", 0.02)
        time.sleep(0.5)
        caption = "A grassy plain surrounded by distant mountains under a clear blue sky."
        prompt = s.prompt_template.format(prompt=" ".join(p for p in (caption, ctx.params.get("prompt") or "") if p))
        ctx.set_text(caption, prompt)

        steps = int(ctx.params.get("steps") or s.steps)
        for i in range(steps):
            ctx.progress("generating", 0.05 + 0.6 * (i + 1) / steps)
            time.sleep(0.05)

        ctx.progress("upscaling", 0.7)
        pano, distance, valid = render_scene(s.out_width, s.out_height, seed=int(ctx.params["seed"]))
        ctx.progress("depth", 0.85)
        depth, depth_meta = encode_depth(
            distance,
            valid,
            ctx.params.get("depth_mode") or s.depth_mode,
            s.depth_near_percentile,
            s.depth_far_percentile,
        )
        ctx.progress("saving", 0.95)
        info = write_outputs(pano, depth, ctx.out_dir, s.jpeg_quality)
        return {**info, "depth": depth_meta, "people_removed": 0, "mock": True}


def render_scene(width: int, height: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=3)

    lon = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * np.pi  # -pi..pi, 0 = image centre
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi  # +pi/2 top .. -pi/2 bottom
    lon, lat = np.meshgrid(lon, lat)

    mountain_h = 25 + 14 * np.sin(3 * lon + phase[0]) + 8 * np.sin(7 * lon + phase[1]) + 4 * np.sin(17 * lon + phase[2])
    wall_top = mountain_h - CAMERA_HEIGHT  # relative to the camera
    wall_dist = MOUNTAIN_RADIUS / np.cos(lat)

    with np.errstate(divide="ignore", invalid="ignore"):
        ground_dist = np.where(lat < 0, CAMERA_HEIGHT / np.sin(-lat), np.inf)
        ground_horiz = ground_dist * np.cos(lat)

    hits_ground = (lat < 0) & (ground_horiz <= MOUNTAIN_RADIUS)
    hits_wall = ~hits_ground & (MOUNTAIN_RADIUS * np.tan(lat) <= wall_top)
    valid = hits_ground | hits_wall
    distance = np.where(hits_ground, ground_dist, np.where(hits_wall, wall_dist, 0.0))

    # --- colours ---------------------------------------------------------
    t = np.clip(lat / (np.pi / 2), 0, 1)[..., None]
    sky = (1 - t) * np.array([205, 222, 240]) + t * np.array([60, 110, 195])

    rel_h = np.clip((MOUNTAIN_RADIUS * np.tan(lat) + CAMERA_HEIGHT) / np.maximum(mountain_h, 1), 0, 1)[..., None]
    mountain = (1 - rel_h) * np.array([95, 110, 95]) + rel_h * np.array([150, 160, 175])
    mountain = mountain * 0.75 + np.array([205, 222, 240]) * 0.25  # aerial haze

    with np.errstate(invalid="ignore"):
        gx = ground_dist * np.cos(lat) * np.sin(lon)
        gz = ground_dist * np.cos(lat) * np.cos(lon)
        checker = ((np.floor(np.nan_to_num(gx) / 2) + np.floor(np.nan_to_num(gz) / 2)) % 2)[..., None]
    fade = np.clip(np.nan_to_num(ground_dist, posinf=1e9) / 120.0, 0, 1)[..., None]
    ground = checker * np.array([88, 140, 70]) + (1 - checker) * np.array([62, 112, 52])
    ground = ground * (1 - fade * 0.6) + np.array([150, 165, 140]) * fade * 0.6

    rgb = np.where(hits_ground[..., None], ground, np.where(hits_wall[..., None], mountain, sky))
    return np.clip(rgb, 0, 255).astype(np.uint8), distance.astype(np.float32), valid
