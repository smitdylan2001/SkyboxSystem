"""Checkerboard test pipeline: no AI, no GPU, finishes in a few seconds.

Used for POST /v1/test-jobs (always) and for every job when MOCK_PIPELINE=true.
It renders a synthetic but geometrically exact scene and then follows the same
path as a real job (progress stages, depth encoding, result files), so it tests
the queue, the API, the downloads and the Unity side without generation:

* the ground is a checkerboard of 2 m tiles (lines should stay straight once displaced),
* a ring of mountains stands at 150 m, the sky has grid lines every 30 degrees,
* the uploaded image hangs on a 8 m wide panel 6 m in front of the camera
  (at the centre of the equirect = "front"), with correct depth,
* N / E / S / W markers show the horizontal orientation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from . import JobContext
from .outputs import encode_depth, write_outputs

CAMERA_HEIGHT = 1.6
MOUNTAIN_RADIUS = 150.0
PANEL_DISTANCE = 6.0
PANEL_WIDTH = 8.0
STAGE_DELAY = 0.25  # seconds per fake stage, so clients can see the progress updates


class CheckerboardPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prefetch(self) -> None:
        pass

    def release(self) -> None:
        pass

    def run(self, ctx: JobContext) -> dict[str, Any]:
        s = self.settings
        started = time.time()
        ctx.progress("captioning", 0.05)
        source = Image.open(ctx.input_path).convert("RGB") if ctx.input_path.exists() else None
        ctx.set_text("Checkerboard test scene (no generation).", "test")
        time.sleep(STAGE_DELAY)

        for i in range(4):
            ctx.progress("generating", 0.1 + 0.15 * (i + 1))
            time.sleep(STAGE_DELAY)

        ctx.progress("estimating_depth", 0.75)
        pano, distance, valid = render_test_scene(s.out_width, s.out_height, source, seed=int(ctx.params.get("seed", 0)))
        depth, depth_meta = encode_depth(
            distance,
            valid,
            ctx.params.get("depth_mode") or s.depth_mode,
            s.depth_near_percentile,
            s.depth_far_percentile,
        )
        ctx.progress("saving", 0.9)
        info = write_outputs(pano, depth, ctx.out_dir, s.jpeg_quality)
        return {
            **info,
            "depth": depth_meta,
            "people_removed": 0,
            "test": True,
            "stages": {"checkerboard": {"seconds": round(time.time() - started, 1), "peak_vram_gb": 0.0}},
        }


# MOCK_PIPELINE=true uses the same pipeline for every job.
MockPipeline = CheckerboardPipeline


def render_test_scene(
    width: int, height: int, panel_image: Image.Image | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (rgb uint8 HxWx3, ray distance in meters, valid = not sky)."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=3)

    lon = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * np.pi  # -pi..pi, 0 = image centre (front)
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi  # +pi/2 top .. -pi/2 bottom
    lon, lat = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat)
    dir_x, dir_y, dir_z = cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)

    # Ground plane and mountain ring.
    mountain_h = 25 + 14 * np.sin(3 * lon + phase[0]) + 8 * np.sin(7 * lon + phase[1]) + 4 * np.sin(17 * lon + phase[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ground_dist = np.where(lat < 0, CAMERA_HEIGHT / np.sin(-lat), np.inf)
    hits_ground = (lat < 0) & (ground_dist * cos_lat <= MOUNTAIN_RADIUS)
    hits_wall = ~hits_ground & (MOUNTAIN_RADIUS * np.tan(lat) <= mountain_h - CAMERA_HEIGHT)
    distance = np.where(hits_ground, ground_dist, np.where(hits_wall, MOUNTAIN_RADIUS / cos_lat, np.inf))

    # Panel in front of the camera (plane z = PANEL_DISTANCE).
    aspect = (panel_image.height / panel_image.width) if panel_image is not None else 9 / 16
    panel_h = PANEL_WIDTH * aspect
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dir_z > 1e-6, PANEL_DISTANCE / dir_z, np.inf)
        px, py = dir_x * t, dir_y * t
    hits_panel = (np.abs(px) <= PANEL_WIDTH / 2) & (np.abs(py) <= panel_h / 2) & (t < distance)
    distance = np.where(hits_panel, t, distance)
    valid = np.isfinite(distance)

    # --- colours ---------------------------------------------------------
    sky_t = np.clip(lat / (np.pi / 2), 0, 1)[..., None]
    rgb = (1 - sky_t) * np.array([205.0, 222, 240]) + sky_t * np.array([60.0, 110, 195])
    grid = (np.abs(((np.degrees(lon) + 180) % 30) - 15) > 14.6) | (np.abs(((np.degrees(lat) + 90) % 30) - 15) > 14.6)
    rgb = np.where((grid & ~valid)[..., None], rgb * 0.82, rgb)

    rel_h = np.clip((MOUNTAIN_RADIUS * np.tan(lat) + CAMERA_HEIGHT) / np.maximum(mountain_h, 1), 0, 1)[..., None]
    mountain = (1 - rel_h) * np.array([95.0, 110, 95]) + rel_h * np.array([150.0, 160, 175])
    rgb = np.where(hits_wall[..., None], mountain * 0.75 + np.array([205.0, 222, 240]) * 0.25, rgb)

    with np.errstate(invalid="ignore"):
        gx = np.nan_to_num(ground_dist * cos_lat * np.sin(lon))
        gz = np.nan_to_num(ground_dist * cos_lat * np.cos(lon))
    checker = ((np.floor(gx / 2) + np.floor(gz / 2)) % 2)[..., None]
    fade = np.clip(np.nan_to_num(ground_dist, posinf=1e9) / 120.0, 0, 1)[..., None]
    ground = checker * np.array([235.0, 235, 235]) + (1 - checker) * np.array([40.0, 40, 40])
    ground = ground * (1 - fade * 0.6) + np.array([150.0, 160, 150]) * fade * 0.6
    rgb = np.where(hits_ground[..., None], ground, rgb)

    if hits_panel.any():
        if panel_image is None:
            panel_image = _placeholder_panel()
        img = np.asarray(panel_image.convert("RGB"), dtype=np.float32)
        u = np.clip((px[hits_panel] / PANEL_WIDTH + 0.5) * (img.shape[1] - 1), 0, img.shape[1] - 1)
        v = np.clip((0.5 - py[hits_panel] / panel_h) * (img.shape[0] - 1), 0, img.shape[0] - 1)
        rgb[hits_panel] = img[v.round().astype(int), u.round().astype(int)]
        border = hits_panel & ((np.abs(px) > PANEL_WIDTH / 2 - 0.08) | (np.abs(py) > panel_h / 2 - 0.08))
        rgb[border] = (255, 200, 0)

    out = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    _draw_compass(out)
    return np.asarray(out), distance.astype(np.float32), valid


def _draw_compass(img: Image.Image) -> None:
    """N/E/S/W letters on the horizon: N = front (image centre), E = 90 deg to the right."""
    draw = ImageDraw.Draw(img)
    size = max(16, img.height // 18)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    w, h = img.size
    for label, deg in (("N", 0), ("E", 90), ("S", 180), ("W", -90)):
        x = (deg / 360.0 + 0.5) % 1.0 * w
        for wrapped_x in (x - w, x, x + w):  # a label on the seam is drawn on both edges
            draw.text((wrapped_x, h * 0.43), label, fill=(255, 40, 40), font=font, anchor="mm",
                      stroke_width=max(2, size // 12), stroke_fill=(255, 255, 255))


def _placeholder_panel() -> Image.Image:
    img = Image.new("RGB", (640, 360), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(0, 360, 40):
        for x in range(0, 640, 40):
            if (x // 40 + y // 40) % 2:
                draw.rectangle([x, y, x + 39, y + 39], fill=(200, 30, 30))
    return img
