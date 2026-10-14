"""Processing pipelines. The real one needs a CUDA GPU; the mock one runs anywhere."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..config import Settings


class JobCancelled(Exception):
    """Raised from a progress callback when the client cancelled the job."""


@dataclass
class JobContext:
    id: str
    params: dict[str, Any]
    input_path: Path
    out_dir: Path
    # progress(stage, fraction 0..1). Raises JobCancelled if the job was cancelled.
    progress: Callable[[str, float], None]
    # set_text(caption, final_prompt) so clients can see what was generated.
    set_text: Callable[[str, str], None]


class Pipeline(Protocol):
    def prefetch(self) -> None:
        """Download every model up-front so the first job doesn't stall."""

    def run(self, ctx: JobContext) -> dict[str, Any]:
        """Process one job, write outputs into ctx.out_dir, return result metadata."""

    def release(self) -> None:
        """Free GPU memory after a job (the GPU is shared with other containers)."""


def create_pipeline(settings: Settings) -> Pipeline:
    if settings.mock_pipeline:
        return create_test_pipeline(settings)
    from .skybox import SkyboxPipeline

    return SkyboxPipeline(settings)


def create_test_pipeline(settings: Settings) -> Pipeline:
    """Checkerboard pipeline for /v1/test-jobs: no models, no GPU."""
    from .mock import CheckerboardPipeline

    return CheckerboardPipeline(settings)
