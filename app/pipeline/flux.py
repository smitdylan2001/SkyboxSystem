"""FLUX.1-dev (NF4) panorama generation with the DiT360 LoRA, in memory-friendly stages.

Stage 1  encode_prompts()   CLIP + T5 (~4.5 GB VRAM)          -> embeddings, freed
Stage 2  redux_tokens()     SigLIP + Redux embedder (opt.)    -> image tokens, freed
Stage 3  generate_latents() transformer + LoRA (~7.3 GB VRAM) -> latents, freed
Stage 4  load_vae()/decode  VAE with wrap-around padding      -> seamless RGB panorama
Stage 5  inpaint_regions()  transformer without LoRA          -> people painted out (only if needed)

Seamlessness at the left/right edge comes from four things:
  * DiT360-style circular padding: one wrapped token column on each side of the
    latent grid (with matching position ids), kept in sync after every step;
  * rolling the latent horizontally between steps, so no content column sits at
    the transformer's position seam for long (layout forms without a seam);
  * a rotation ensemble for the last few steps: the transformer also predicts a
    copy rotated by half a panorama, and the two predictions are cross-faded so
    each one's seam region comes from the other (removes the final faint band);
  * a VAE decoder whose convolutions pad circularly along the width.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .gpu import DEVICE, DTYPE, free_gpu, to_uint8_image

FLUX_FILES = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "transformer/*",
    "vae/*",
]
VAE_SCALE = 8  # FLUX VAE downsamples 8x; packing groups 2x2 latents into one 16x16 px token.


@dataclass
class PromptEmbeds:
    prompt_embeds: torch.Tensor  # (1, seq, 4096) on CPU
    pooled: torch.Tensor  # (1, 768) on CPU


# --------------------------------------------------------------------------
# Stage 1/2: conditioning
# --------------------------------------------------------------------------


def encode_prompts(repo: str, prompts: list[str]) -> list[PromptEmbeds]:
    from diffusers import FluxPipeline
    from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", dtype=DTYPE).to(DEVICE)
    # Pre-quantized NF4 weights; bitsandbytes places them directly on the GPU.
    text_encoder_2 = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder_2", dtype=DTYPE)
    pipe = FluxPipeline(
        scheduler=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=None,
    )
    try:
        out = []
        with torch.inference_mode():
            for prompt in prompts:
                embeds, pooled, _ = pipe.encode_prompt(
                    prompt=prompt, prompt_2=None, device=DEVICE, max_sequence_length=512
                )
                out.append(PromptEmbeds(embeds.to("cpu", DTYPE), pooled.to("cpu", DTYPE)))
        return out
    finally:
        del pipe, text_encoder, text_encoder_2
        free_gpu()


def redux_tokens(repo: str, image: Image.Image) -> torch.Tensor:
    """FLUX Redux image tokens (1, 729, 4096) carrying the photo's look; scaled by the caller."""
    from diffusers import FluxPriorReduxPipeline

    redux = FluxPriorReduxPipeline.from_pretrained(repo, dtype=DTYPE).to(DEVICE)
    try:
        with torch.inference_mode():
            features = redux.encode_image(image, DEVICE, 1)
            tokens = redux.image_embedder(features).image_embeds
        return tokens.to("cpu", DTYPE)
    finally:
        del redux
        free_gpu()


# --------------------------------------------------------------------------
# Stage 3: denoising
# --------------------------------------------------------------------------


def _pano_pipeline_class():
    from diffusers import FluxPipeline

    class PanoFluxPipeline(FluxPipeline):
        """FluxPipeline + DiT360 circular padding (one wrapped token column per side)."""

        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            latents, ids = super().prepare_latents(
                batch_size, num_channels_latents, height, width, dtype, device, generator, latents
            )
            n_h = int(height) // (self.vae_scale_factor * 2)
            n_w = int(width) // (self.vae_scale_factor * 2)
            b, _, d = latents.shape
            grid = latents.reshape(b, n_h, n_w, d)
            latents = torch.cat([grid[:, :, -1:], grid, grid[:, :, :1]], dim=2).reshape(b, -1, d)
            id_grid = ids.reshape(n_h, n_w, ids.shape[-1])
            ids = torch.cat([id_grid[:, -1:], id_grid, id_grid[:, :1]], dim=1).reshape(-1, ids.shape[-1])
            return latents, ids

    return PanoFluxPipeline


class SeamKeeper:
    """Keeps the panorama seamless: step-end callback + transformer forward wrapper.

    Steps [0, ensemble_from): roll the content by `roll_tokens` after every step
    (the last of these rolls returns it to its original alignment).
    Steps [ensemble_from, steps): rotation ensemble (two transformer passes).
    """

    def __init__(
        self,
        height: int,
        width: int,
        steps: int,
        roll_tokens: int,
        ensemble_steps: int,
        on_step: Callable[[int], None],
    ) -> None:
        self.n_h = height // (VAE_SCALE * 2)
        self.n_w = width // (VAE_SCALE * 2)
        self.ensemble_from = max(0, steps - max(0, ensemble_steps))
        self.ensemble_active = self.ensemble_from == 0
        self.roll = roll_tokens % self.n_w if roll_tokens else 0
        self.roll_steps = self.ensemble_from if self.roll else 0
        self.total_roll = 0
        self.on_step = on_step
        # Cross-fade weight for the un-rotated prediction: 0 at the seam (edges), 1 in the middle half.
        c = torch.arange(self.n_w, dtype=torch.float32) + 0.5
        t = (torch.minimum(c, self.n_w - c) / (self.n_w / 4)).clamp(0, 1)
        self._w_center = (t * t * (3 - 2 * t)).view(1, 1, self.n_w, 1)

    # -- layout helpers (packed tokens incl. the two pad columns) --------------

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.n_h, self.n_w + 2, x.shape[-1])[:, :, 1:-1]

    @staticmethod
    def _padded(core: torch.Tensor) -> torch.Tensor:
        b, _, _, d = core.shape
        return torch.cat([core[:, :, -1:], core, core[:, :, :1]], dim=2).reshape(b, -1, d)

    # -- hooks -------------------------------------------------------------------

    def install(self, transformer) -> None:
        original = transformer.forward
        half = self.n_w // 2

        def forward(*args, **kwargs):
            out = original(*args, **kwargs)
            if not self.ensemble_active or args or not isinstance(out, tuple):
                return out
            rotated = self._padded(self._core(kwargs["hidden_states"]).roll(half, dims=2))
            out_rot = original(**{**kwargs, "hidden_states": rotated})
            pred = self._core(out[0])
            pred_rot = self._core(out_rot[0]).roll(-half, dims=2)
            w = self._w_center.to(pred.device, pred.dtype)
            return (self._padded(pred * w + pred_rot * (1 - w)),) + tuple(out[1:])

        transformer.forward = forward

    def __call__(self, pipe, step: int, timestep, kwargs: dict) -> dict:
        core = self._core(kwargs["latents"])
        if step < self.roll_steps:
            last = step == self.roll_steps - 1
            shift = (-self.total_roll) % self.n_w if last else self.roll
            if shift:
                core = core.roll(shift, dims=2)
                self.total_roll = (self.total_roll + shift) % self.n_w
        self.ensemble_active = step + 1 >= self.ensemble_from
        self.on_step(step)
        return {"latents": self._padded(core)}

    def finish(self, latents: torch.Tensor) -> torch.Tensor:
        """Drop the pad columns and undo any remaining roll -> (B, n_h*n_w, 64)."""
        core = self._core(latents)
        if self.total_roll:
            core = core.roll(-self.total_roll, dims=2)
        return core.reshape(core.shape[0], -1, core.shape[-1])


def generate_latents(
    repo: str,
    lora_repo: str,
    lora_weight: str | None,
    lora_scale: float,
    embeds: PromptEmbeds,
    style_tokens: torch.Tensor | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    roll_tokens: int,
    ensemble_steps: int,
    on_step: Callable[[int], None],
) -> torch.Tensor:
    """Run the denoiser; returns unpacked latents (1, 16, H/8, W/8) on CPU."""
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", dtype=DTYPE)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    pipe = _pano_pipeline_class()(
        scheduler=scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    try:
        if lora_repo:
            pipe.load_lora_weights(lora_repo, weight_name=lora_weight, adapter_name="pano")
            pipe.set_adapters(["pano"], adapter_weights=[lora_scale])

        prompt_embeds = embeds.prompt_embeds.to(DEVICE, DTYPE)
        if style_tokens is not None:
            prompt_embeds = torch.cat([prompt_embeds, style_tokens.to(DEVICE, DTYPE)], dim=1)

        keeper = SeamKeeper(height, width, steps, roll_tokens, ensemble_steps, on_step)
        keeper.install(transformer)
        with torch.inference_mode():
            packed = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=embeds.pooled.to(DEVICE, DTYPE),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator("cpu").manual_seed(seed),
                output_type="latent",
                callback_on_step_end=keeper,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
            latents = FluxPipeline._unpack_latents(keeper.finish(packed), height, width, VAE_SCALE)
        return latents.to("cpu")
    finally:
        del pipe, transformer
        free_gpu()


# --------------------------------------------------------------------------
# Stage 4: seamless VAE
# --------------------------------------------------------------------------


def load_vae(repo: str):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", dtype=DTYPE).to(DEVICE)
    _make_decoder_wrap_x(vae)
    return vae


def _wrap_conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    ph, pw = self._wrap_padding
    x = F.pad(x, (pw, pw, 0, 0), mode="circular" if self._wrap_x else "constant")
    x = F.pad(x, (0, 0, ph, ph))
    return self._conv_forward(x, self.weight, self.bias)


def _make_decoder_wrap_x(vae) -> None:
    """Make every padded conv in the decoder wrap around horizontally (equirect seam)."""
    for module in vae.decoder.modules():
        if isinstance(module, nn.Conv2d) and module.padding_mode == "zeros" and module.padding != (0, 0):
            module._wrap_padding = tuple(module.padding)
            module._wrap_x = True
            module.padding = (0, 0)
            module.forward = types.MethodType(_wrap_conv_forward, module)


@contextmanager
def vae_wrap_disabled(vae) -> Iterator[None]:
    """For crops (inpainting) the left/right edges are not neighbours."""
    convs = [m for m in vae.decoder.modules() if hasattr(m, "_wrap_x")]
    for m in convs:
        m._wrap_x = False
    try:
        yield
    finally:
        for m in convs:
            m._wrap_x = True


def decode(vae, latents: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        x = latents.to(DEVICE, DTYPE) / vae.config.scaling_factor + vae.config.shift_factor
        image = vae.decode(x, return_dict=False)[0]
    result = to_uint8_image(image)
    del image, x
    free_gpu()
    return result


# --------------------------------------------------------------------------
# Stage 5: inpainting (people removal)
# --------------------------------------------------------------------------


@dataclass
class Region:
    y0: int
    x0: int  # may be negative / wrap around
    size: int
    mask: np.ndarray  # (size, size) float32 0..1, feathered


def inpaint_regions(
    repo: str,
    vae,
    pano: np.ndarray,
    regions: list[Region],
    embeds: PromptEmbeds,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
    on_region: Callable[[int], None],
) -> np.ndarray:
    """Repaint each square crop (base FLUX, no panorama LoRA) and blend it back in."""
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxInpaintPipeline, FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_pretrained(repo, subfolder="transformer", dtype=DTYPE)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    pipe = FluxInpaintPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    pano = pano.copy()
    width = pano.shape[1]
    try:
        with vae_wrap_disabled(vae), torch.inference_mode():
            for i, region in enumerate(regions):
                on_region(i)
                cols = np.arange(region.x0, region.x0 + region.size) % width
                rows = slice(region.y0, region.y0 + region.size)
                crop = pano[rows][:, cols]
                hard_mask = (region.mask > 0.05).astype(np.uint8) * 255
                result = pipe(
                    prompt_embeds=embeds.prompt_embeds.to(DEVICE, DTYPE),
                    pooled_prompt_embeds=embeds.pooled.to(DEVICE, DTYPE),
                    image=Image.fromarray(crop),
                    mask_image=Image.fromarray(hard_mask),
                    height=region.size,
                    width=region.size,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=torch.Generator("cpu").manual_seed(seed + i),
                ).images[0]
                painted = np.asarray(result.convert("RGB").resize((region.size, region.size)), dtype=np.float32)
                alpha = region.mask[..., None]
                blended = painted * alpha + crop.astype(np.float32) * (1 - alpha)
                block = pano[rows]
                block[:, cols] = np.clip(blended, 0, 255).round().astype(np.uint8)
                pano[rows] = block
        return pano
    finally:
        del pipe, transformer
        free_gpu()
