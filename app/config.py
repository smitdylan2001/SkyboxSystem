"""Runtime configuration, read once from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _list(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # --- API -------------------------------------------------------------
    api_keys: tuple[str, ...]
    auth_disabled: bool
    data_dir: Path
    max_upload_mb: int
    max_queue: int
    result_ttl_hours: float
    cors_origins: tuple[str, ...]
    log_level: str

    # --- Worker ----------------------------------------------------------
    mock_pipeline: bool
    prefetch_on_start: bool

    # --- Panorama generation (FLUX.1-dev + 360 LoRA) -----------------------
    flux_repo: str
    flux_quant_repo: str
    lora_repo: str
    lora_weight: str
    lora_scale: float
    prompt_template: str
    gen_width: int
    gen_height: int
    steps: int
    guidance: float
    seam_roll_tokens: int
    seam_ensemble_steps: int
    redux_repo: str
    redux_strength: float

    # --- Captioning ------------------------------------------------------
    caption_model: str
    caption_max_tokens: int

    # --- People removal --------------------------------------------------
    remove_people: bool
    people_score: float
    people_max_regions: int
    inpaint_strength: float

    # --- Upscaling -------------------------------------------------------
    out_width: int
    upscaler_url: str

    # --- Depth -----------------------------------------------------------
    depth_model: str
    sky_model: str
    depth_mode: str
    depth_merge_width: int
    depth_near_percentile: float
    depth_far_percentile: float

    # --- Output ----------------------------------------------------------
    jpeg_quality: int

    @property
    def out_height(self) -> int:
        return self.out_width // 2

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            api_keys=_list("API_KEYS"),
            auth_disabled=_bool("AUTH_DISABLED", False),
            data_dir=Path(_str("DATA_DIR", "/data")),
            max_upload_mb=_int("MAX_UPLOAD_MB", 25),
            max_queue=_int("MAX_QUEUE", 50),
            result_ttl_hours=_float("RESULT_TTL_HOURS", 168),
            cors_origins=_list("CORS_ORIGINS"),
            log_level=_str("LOG_LEVEL", "INFO").upper(),
            mock_pipeline=_bool("MOCK_PIPELINE", False),
            prefetch_on_start=_bool("PREFETCH_ON_START", True),
            flux_repo=_str("FLUX_REPO", "black-forest-labs/FLUX.1-dev"),
            flux_quant_repo=_str("FLUX_QUANT_REPO", "diffusers/FLUX.1-dev-bnb-4bit"),
            lora_repo=_str("PANO_LORA_REPO", "Insta360-Research/DiT360-Panorama-Image-Generation"),
            lora_weight=_str("PANO_LORA_WEIGHT", "adapter_model.safetensors"),
            lora_scale=_float("PANO_LORA_SCALE", 1.0),
            prompt_template=_str("PANO_PROMPT_TEMPLATE", "This is a panorama. The image shows {prompt}"),
            gen_width=_int("GEN_WIDTH", 2048),
            gen_height=_int("GEN_HEIGHT", 1024),
            steps=_int("STEPS", 28),
            guidance=_float("GUIDANCE", 2.8),
            seam_roll_tokens=_int("SEAM_ROLL_TOKENS", 8),
            seam_ensemble_steps=_int("SEAM_ENSEMBLE_STEPS", 4),
            redux_repo=_str("REDUX_REPO", "black-forest-labs/FLUX.1-Redux-dev"),
            redux_strength=_float("REDUX_STRENGTH", 0.0),
            caption_model=_str("CAPTION_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
            caption_max_tokens=_int("CAPTION_MAX_TOKENS", 160),
            remove_people=_bool("REMOVE_PEOPLE", True),
            people_score=_float("PEOPLE_SCORE", 0.5),
            people_max_regions=_int("PEOPLE_MAX_REGIONS", 8),
            inpaint_strength=_float("INPAINT_STRENGTH", 0.92),
            out_width=_int("OUT_WIDTH", 4096),
            upscaler_url=_str(
                "UPSCALER_URL",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            ),
            depth_model=_str("DEPTH_MODEL", "Ruicheng/moge-2-vitl-normal"),
            sky_model=_str("SKY_MODEL", "nvidia/segformer-b5-finetuned-ade-640-640"),
            depth_mode=_str("DEPTH_MODE", "linear").lower(),
            depth_merge_width=_int("DEPTH_MERGE_WIDTH", 1024),
            depth_near_percentile=_float("DEPTH_NEAR_PERCENTILE", 0.5),
            depth_far_percentile=_float("DEPTH_FAR_PERCENTILE", 99.5),
            jpeg_quality=_int("JPEG_QUALITY", 92),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not self.auth_disabled and not self.api_keys:
            raise RuntimeError("Set API_KEYS (comma separated) or AUTH_DISABLED=true.")
        if self.gen_width % 16 or self.gen_height % 16:
            raise RuntimeError("GEN_WIDTH and GEN_HEIGHT must be multiples of 16.")
        if self.gen_width != 2 * self.gen_height:
            raise RuntimeError("GEN_WIDTH must be exactly 2 x GEN_HEIGHT (equirectangular).")
        if self.out_width % 2:
            raise RuntimeError("OUT_WIDTH must be even.")
        if self.depth_mode not in DEPTH_MODES:
            raise RuntimeError(f"DEPTH_MODE must be one of {sorted(DEPTH_MODES)}.")


DEPTH_MODES = {"linear", "inverse", "log"}

settings = Settings.from_env()
