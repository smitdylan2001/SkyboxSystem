"""HTTP API. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import io
import logging
import random
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from .auth import require_api_key
from .config import DEPTH_MODES, settings
from .jobs import DONE, FINAL_STATES, Job, JobStore, new_job_id
from .pipeline.outputs import DEPTH_PNG, DEPTH_R16, SKYBOX_JPG, SKYBOX_PNG
from .worker import Worker

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("skybox.api")
# huggingface_hub logs every cache check at INFO via httpx.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Inputs are only used for captioning/style, so they're stored downscaled.
MAX_STORED_INPUT_SIDE = 2048
Image.MAX_IMAGE_PIXELS = 80_000_000

store = JobStore(settings.data_dir / "jobs.sqlite3")
worker = Worker(store, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    requeued = store.requeue_interrupted()
    if requeued:
        log.info("re-queued %d job(s) interrupted by a restart", requeued)
    worker.start()
    cleanup = asyncio.create_task(_cleanup_loop())
    yield
    cleanup.cancel()
    worker.stop()


app = FastAPI(
    title="Skybox Generator",
    version=__version__,
    description="Queue an image, get back an equirectangular skybox and a matching depth map.",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    counts = store.counts()
    return {
        "status": "ok" if worker.is_alive() else "worker_down",
        "version": __version__,
        "worker": worker.state,
        "worker_error": worker.last_error,
        "mock": settings.mock_pipeline,
        "pipeline_ready": worker.pipeline_ready,
        "queued": counts.get("queued", 0),
        "running": counts.get("running", 0),
    }


@app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def create_job(
    image: UploadFile = File(..., description="JPG/PNG/WebP photo or video frame to take inspiration from."),
    prompt: str = Form("", max_length=600, description="Optional extra description, appended to the auto caption."),
    use_caption: bool = Form(True, description="false = skip auto captioning and use only `prompt`."),
    seed: int = Form(-1, description="-1 = random."),
    steps: int = Form(0, ge=0, le=60, description="0 = server default, otherwise 8..60."),
    depth_mode: str = Form("", description="linear | inverse | log. Empty = server default."),
    remove_people: bool = Form(True),
    style_strength: float = Form(-1.0, le=1.0, description="0..1 photo style transfer (FLUX Redux). -1 = server default."),
    style: str = Form(
        "", max_length=200,
        description="Visual style of the panorama, e.g. 'stylized 3D animated film render'. Empty = detected from the image.",
    ),
) -> JSONResponse:
    if not use_caption and not prompt.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "use_caption=false requires a prompt.")
    _check_queue()
    params = {
        "prompt": prompt.strip(),
        "use_caption": use_caption,
        "seed": seed if seed >= 0 else random.randint(0, 2**31 - 1),
        "steps": max(8, steps) if steps else settings.steps,
        "depth_mode": _depth_mode(depth_mode),
        "remove_people": remove_people,
        "style_strength": style_strength if style_strength >= 0 else settings.redux_strength,
        "style": " ".join(style.split()),
    }
    picture = await _read_image(image)
    return await _enqueue(params, picture, image.filename)


@app.post("/v1/test-jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def create_test_job(
    image: UploadFile | None = File(None, description="Optional; shown on a panel 6 m in front of the camera."),
    seed: int = Form(-1, description="-1 = random (only varies the mountains)."),
    depth_mode: str = Form("", description="linear | inverse | log. Empty = server default."),
) -> JSONResponse:
    """Test run without generation.

    Goes through the same queue, worker, depth encoding, result files and download URLs as
    /v1/jobs, but skips every AI model and returns a checkerboard test scene within seconds
    (works even when the GPU pipeline is down). Other form fields of /v1/jobs are accepted and ignored.
    """
    _check_queue()
    params = {
        "test": True,
        "prompt": "",
        "seed": seed if seed >= 0 else random.randint(0, 2**31 - 1),
        "depth_mode": _depth_mode(depth_mode),
    }
    picture = await _read_image(image) if image is not None and image.filename else None
    return await _enqueue(params, picture, image.filename if image is not None else "")


@app.get("/v1/jobs", dependencies=[Depends(require_api_key)])
def list_jobs(limit: int = 20) -> dict[str, Any]:
    return {"jobs": [_job_json(j) for j in store.list_recent(max(1, min(limit, 200)))]}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, Any]:
    return _job_json(_get_or_404(job_id))


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def delete_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued/running job, or delete a finished job and its files."""
    job = _get_or_404(job_id)
    if job.status in FINAL_STATES:
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
        store.delete(job_id)
        return {"id": job_id, "status": "deleted"}
    job = store.request_cancel(job_id)
    assert job is not None
    if job.status in FINAL_STATES:
        shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
    return _job_json(job)


_FILES = {
    SKYBOX_JPG: "image/jpeg",
    SKYBOX_PNG: "image/png",
    DEPTH_PNG: "image/png",
    DEPTH_R16: "application/octet-stream",
}


@app.get("/v1/jobs/{job_id}/{filename}", dependencies=[Depends(require_api_key)])
def get_job_file(job_id: str, filename: str) -> FileResponse:
    if filename not in _FILES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown file. Available: {sorted(_FILES)}")
    job = _get_or_404(job_id)
    if job.status != DONE:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Job is {job.status}, results not available.")
    path = settings.jobs_dir / job_id / "out" / filename
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Result file missing (expired?).")
    return FileResponse(
        path,
        media_type=_FILES[filename],
        filename=f"{job_id}_{filename}",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _check_queue() -> None:
    if store.pending_count() >= settings.max_queue:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Queue is full, try again later.")


def _depth_mode(value: str) -> str:
    mode = value.strip().lower() or settings.depth_mode
    if mode not in DEPTH_MODES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"depth_mode must be one of {sorted(DEPTH_MODES)}.")
    return mode


async def _read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image larger than {settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The uploaded image is empty.")
    try:
        return await asyncio.to_thread(_load_image, data)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Could not read image: {exc}") from exc


async def _enqueue(params: dict[str, Any], picture: Image.Image | None, filename: str | None) -> JSONResponse:
    params["source_filename"] = (filename or "")[:200]
    params["source_size"] = [picture.width, picture.height] if picture is not None else [0, 0]
    job_id = new_job_id()
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if picture is not None:
        await asyncio.to_thread(picture.save, job_dir / "input.png")
    job = store.create(params, job_id=job_id)
    worker.notify()
    log.info(
        "queued %sjob %s (%dx%d, prompt=%r)",
        "test " if params.get("test") else "", job.id, *params["source_size"], params.get("prompt", "")[:60],
    )
    return JSONResponse(_job_json(job), status_code=status.HTTP_202_ACCEPTED, headers={"Location": f"/v1/jobs/{job.id}"})


def _get_or_404(job_id: str) -> Job:
    job = store.get(job_id) if job_id.isalnum() else None
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return job


def _load_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_STORED_INPUT_SIDE, MAX_STORED_INPUT_SIDE), Image.Resampling.LANCZOS)
    return img


def _iso(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else ""


def _job_json(job: Job) -> dict[str, Any]:
    """Flat, null-free JSON so Unity's JsonUtility can parse it directly."""
    base = f"/v1/jobs/{job.id}"
    done = job.status == DONE
    depth = job.result.get("depth", {}) if done else {}
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": round(job.progress, 3),
        "position": job.position,
        "eta_seconds": job.eta_seconds,
        "error": job.error,
        "caption": job.caption,
        "style": str(job.result.get("style") or job.params.get("style") or ""),
        "prompt": job.prompt,
        "seed": int(job.params.get("seed", -1)),
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "status_url": base,
        "skybox_url": f"{base}/{SKYBOX_JPG}" if done else "",
        "skybox_png_url": f"{base}/{SKYBOX_PNG}" if done else "",
        "depth_url": f"{base}/{DEPTH_PNG}" if done else "",
        "depth_raw_url": f"{base}/{DEPTH_R16}" if done else "",
        "width": int(job.result.get("width", 0)),
        "height": int(job.result.get("height", 0)),
        "depth_width": int(job.result.get("depth_width", 0)),
        "depth_height": int(job.result.get("depth_height", 0)),
        "depth_mode": str(depth.get("mode", job.params.get("depth_mode", ""))),
        "depth_near": float(depth.get("near_distance", 0.0)),
        "depth_far": float(depth.get("far_distance", 0.0)),
        "depth_decode": str(depth.get("decode", "")),
        "people_removed": int(job.result.get("people_removed", 0)),
        "test": bool(job.params.get("test", False)),
    }


async def _cleanup_loop() -> None:
    while True:
        try:
            for job_id in store.expired(settings.result_ttl_hours * 3600):
                shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
                store.delete(job_id)
                log.info("expired job %s removed", job_id)
        except Exception:
            log.exception("cleanup failed")
        await asyncio.sleep(1800)
