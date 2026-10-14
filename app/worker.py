"""Single worker thread: takes jobs from the FIFO queue one at a time (real and test jobs alike)."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback

from .config import Settings
from .jobs import Job, JobStore
from .pipeline import JobCancelled, JobContext, Pipeline, create_pipeline, create_test_pipeline

log = logging.getLogger("skybox.worker")


class Worker(threading.Thread):
    def __init__(self, store: JobStore, settings: Settings) -> None:
        super().__init__(name="gpu-worker", daemon=True)
        self.store = store
        self.settings = settings
        self.state = "starting"  # starting | warming_up | idle | busy | stopped
        self.last_error = ""
        self.current_job: str | None = None
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._pipeline: Pipeline | None = None
        self._test_pipeline: Pipeline = create_test_pipeline(settings)

    def notify(self) -> None:
        """Wake the worker after a job was queued."""
        self._wake.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    def run(self) -> None:
        try:
            self._pipeline = create_pipeline(self.settings)
        except Exception as exc:
            # Keep serving test jobs and report the problem in /health and in every real job.
            self.last_error = f"pipeline init failed: {exc}"
            log.exception("pipeline init failed; only /v1/test-jobs will be processed")

        if self._pipeline is not None and self.settings.prefetch_on_start:
            self.state = "warming_up"
            try:
                self._pipeline.prefetch()
            except Exception as exc:
                self.last_error = f"model prefetch failed (will retry per job): {exc}"
                log.exception("model prefetch failed")

        self.state = "idle"
        while not self._stop_event.is_set():
            self._wake.clear()
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(timeout=5.0)
                continue
            self.state = "busy"
            self.current_job = job.id
            try:
                self._process(job)
            finally:
                self.current_job = None
                self.state = "idle"
        self.state = "stopped"

    @property
    def pipeline_ready(self) -> bool:
        return self._pipeline is not None

    def _process(self, job: Job) -> None:
        is_test = bool(job.params.get("test"))
        pipeline = self._test_pipeline if is_test else self._pipeline
        if pipeline is None:
            self.store.fail(job.id, f"Generation pipeline unavailable ({self.last_error}). Test jobs still work.")
            return

        job_dir = self.settings.jobs_dir / job.id
        out_dir = job_dir / "out"
        inputs = sorted(job_dir.glob("input.*"))
        if not inputs and not is_test:
            self.store.fail(job.id, "input image missing on disk")
            return

        def progress(stage: str, fraction: float) -> None:
            self.store.set_progress(job.id, stage, fraction)
            if self.store.is_cancel_requested(job.id):
                raise JobCancelled()

        ctx = JobContext(
            id=job.id,
            params=job.params,
            input_path=inputs[0] if inputs else job_dir / "input.png",
            out_dir=out_dir,
            progress=progress,
            set_text=lambda caption, prompt: self.store.set_text(job.id, caption, prompt),
        )

        started = time.time()
        log.info("%s %s started", "test job" if is_test else "job", job.id)
        try:
            result = pipeline.run(ctx)
            self.store.finish(job.id, result)
            log.info("job %s done in %.1fs", job.id, time.time() - started)
        except JobCancelled:
            shutil.rmtree(out_dir, ignore_errors=True)
            self.store.mark_cancelled(job.id)
            log.info("job %s cancelled", job.id)
        except Exception as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            message = _friendly_error(exc)
            self.store.fail(job.id, message)
            if not is_test:
                self.last_error = message
            log.error("job %s failed: %s\n%s", job.id, message, traceback.format_exc())
        finally:
            try:
                pipeline.release()
            except Exception:
                log.exception("releasing GPU memory failed")


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return (
            "CUDA out of memory. Another container is probably using the GPU; "
            "the job can be resubmitted, or lower GEN_WIDTH/GEN_HEIGHT."
        )
    return f"{name}: {exc}"
