"""The real pipeline: photo -> caption -> FLUX panorama -> people removal -> depth -> upscale."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import Settings
from . import JobContext
from .caption import build_prompt, caption_image
from .depth import estimate_distance, resize_equirect
from .flux import FLUX_FILES, decode, encode_prompts, generate_latents, inpaint_regions, load_vae, redux_tokens
from .gpu import download_file, free_gpu
from .outputs import encode_depth, write_outputs
from .people import detect_people, plan_regions
from .sky import segment_sky
from .upscale import upscale_pano

log = logging.getLogger("skybox.pipeline")

INPAINT_STEPS = 24
INPAINT_GUIDANCE = 3.5


class SkyboxPipeline:
    def __init__(self, settings: Settings) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU visible in the container. Check the NVIDIA runtime and the "
                "deploy.resources.reservations.devices section in docker-compose.yml."
            )
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging

        # CLIP's 77-token truncation warning etc. are expected; keep container logs readable.
        diffusers_logging.set_verbosity_error()
        diffusers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.s = settings
        self.stage_stats: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------

    def prefetch(self) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        s = self.s
        log.info("prefetching models (first start downloads ~20 GB)...")
        snapshot_download(s.flux_quant_repo, allow_patterns=FLUX_FILES)
        if s.lora_repo:
            if s.lora_weight:
                hf_hub_download(s.lora_repo, s.lora_weight)
            else:
                snapshot_download(s.lora_repo, allow_patterns=["*.safetensors", "*.json"])
        snapshot_download(s.caption_model)
        snapshot_download(s.depth_model)
        snapshot_download(s.sky_model, allow_patterns=["*.json", "*.safetensors", "*.bin"])
        if s.redux_strength > 0:
            snapshot_download(s.redux_repo)
        MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=False)
        download_file(s.upscaler_url, "upscalers")
        log.info("all models present")

    def release(self) -> None:
        free_gpu()

    # ------------------------------------------------------------------

    def run(self, ctx: JobContext) -> dict[str, Any]:
        s, p = self.s, ctx.params
        self.stage_stats = {}
        source = Image.open(ctx.input_path).convert("RGB")
        seed = int(p["seed"])
        steps = int(p["steps"])

        # 1. Describe the photo (the panorama is newly generated, inspired by it).
        ctx.progress("captioning", 0.01)
        caption = ""
        if p.get("use_caption", True):
            with self._stage("caption"):
                caption = caption_image(source, s.caption_model, s.caption_max_tokens)
        description = " ".join(x for x in (caption, p.get("prompt", "")) if x).strip()
        prompt = build_prompt(s.prompt_template, description)
        inpaint_prompt = f"{description} The place is completely empty, nobody is there."
        ctx.set_text(caption, prompt)
        log.info("job %s prompt: %s", ctx.id, prompt)

        # 2. Text conditioning (both prompts at once; the encoders are loaded only once).
        ctx.progress("encoding_prompt", 0.08)
        with self._stage("text_encoders"):
            embeds, inpaint_embeds = encode_prompts(s.flux_quant_repo, [prompt, inpaint_prompt])

        style = None
        strength = float(p.get("style_strength", 0) or 0)
        if strength > 0:
            ctx.progress("style", 0.12)
            with self._stage("redux"):
                style = redux_tokens(s.redux_repo, source) * strength

        # 3. Panorama.
        def on_step(i: int) -> None:
            ctx.progress("generating", 0.15 + 0.5 * (i + 1) / steps)

        ctx.progress("generating", 0.15)
        with self._stage("transformer"):
            latents = generate_latents(
                s.flux_quant_repo, s.lora_repo, s.lora_weight, s.lora_scale, embeds, style,
                s.gen_width, s.gen_height, steps, s.guidance, seed, s.seam_roll_tokens, s.seam_ensemble_steps, on_step,
            )

        ctx.progress("decoding", 0.66)
        with self._stage("vae_decode"):
            vae = load_vae(s.flux_quant_repo)
            pano = decode(vae, latents)

        # 4. People removal.
        removed = 0
        if p.get("remove_people", True) and s.remove_people:
            ctx.progress("detecting_people", 0.69)
            with self._stage("people_detect"):
                regions = plan_regions(detect_people(pano, s.people_score), s.people_max_regions)
            if regions:
                log.info("job %s: painting out %d people region(s)", ctx.id, len(regions))
                with self._stage("inpaint"):
                    pano = inpaint_regions(
                        s.flux_quant_repo, vae, pano, regions, inpaint_embeds, INPAINT_STEPS,
                        INPAINT_GUIDANCE, s.inpaint_strength, seed,
                        lambda i: ctx.progress("removing_people", 0.71 + 0.07 * i / len(regions)),
                    )
                removed = len(regions)
        del vae
        free_gpu()

        # 5. Sky mask + depth (on the generation-resolution panorama; 512 px views are plenty).
        ctx.progress("segmenting_sky", 0.78)
        with self._stage("sky"):
            sky = segment_sky(pano, s.sky_model)
        ctx.progress("estimating_depth", 0.80)
        with self._stage("depth"):
            distance, _ = estimate_distance(
                pano, sky, s.depth_model, s.depth_merge_width,
                lambda f: ctx.progress("estimating_depth", 0.80 + 0.08 * f),
            )

        # 6. Upscale to the output size.
        ctx.progress("upscaling", 0.89)
        with self._stage("upscale"):
            pano_out = upscale_pano(pano, s.upscaler_url, s.out_width)

        # 7. Encode and save.
        ctx.progress("saving", 0.96)
        out_h, out_w = pano_out.shape[:2]
        distance = resize_equirect(distance, out_w, out_h)
        valid = resize_equirect(sky.astype(np.uint8), out_w, out_h, nearest=True) == 0
        depth16, depth_meta = encode_depth(
            distance, valid, p.get("depth_mode") or s.depth_mode,
            s.depth_near_percentile, s.depth_far_percentile,
        )
        info = write_outputs(pano_out, depth16, ctx.out_dir, s.jpeg_quality)
        log.info("job %s stages: %s", ctx.id, self.stage_stats)
        return {**info, "depth": depth_meta, "people_removed": removed, "stages": self.stage_stats}

    # ------------------------------------------------------------------

    def _stage(self, name: str):
        pipeline = self

        class _Stage:
            def __enter__(self):
                torch.cuda.reset_peak_memory_stats()
                self.t = time.time()

            def __exit__(self, *exc):
                pipeline.stage_stats[name] = {
                    "seconds": round(time.time() - self.t, 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
                return False

        return _Stage()
