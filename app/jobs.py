"""Persistent FIFO job queue backed by SQLite.

Jobs survive container restarts: anything that was 'running' when the process
died is put back at the front of the queue on startup (it keeps its original
sequence number, so first-in-first-out order is preserved).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
FINAL_STATES = {DONE, FAILED, CANCELLED}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    id               TEXT    NOT NULL UNIQUE,
    status           TEXT    NOT NULL,
    stage            TEXT    NOT NULL DEFAULT '',
    progress         REAL    NOT NULL DEFAULT 0,
    params           TEXT    NOT NULL,
    caption          TEXT    NOT NULL DEFAULT '',
    prompt           TEXT    NOT NULL DEFAULT '',
    error            TEXT    NOT NULL DEFAULT '',
    result           TEXT    NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at       REAL    NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_seq ON jobs(status, seq);
"""


@dataclass
class Job:
    seq: int
    id: str
    status: str
    stage: str
    progress: float
    params: dict[str, Any]
    caption: str
    prompt: str
    error: str
    result: dict[str, Any]
    cancel_requested: bool
    created_at: float
    started_at: float | None
    finished_at: float | None
    # Filled in by JobStore.get() for queued/running jobs.
    position: int = -1
    eta_seconds: int = -1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            seq=row["seq"],
            id=row["id"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            params=json.loads(row["params"]),
            caption=row["caption"],
            prompt=row["prompt"],
            error=row["error"],
            result=json.loads(row["result"]),
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


class JobStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)

    # --- creation / lookup -------------------------------------------------

    def create(self, params: dict[str, Any], job_id: str | None = None) -> Job:
        job_id = job_id or new_job_id()
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, status, params, created_at) VALUES (?, ?, ?, ?)",
                (job_id, QUEUED, json.dumps(params), time.time()),
            )
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = Job.from_row(row)
            self._annotate_queue_info(job)
            return job

    def list_recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
            jobs = [Job.from_row(r) for r in rows]
            for job in jobs:
                self._annotate_queue_info(job)
            return jobs

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def pending_count(self) -> int:
        c = self.counts()
        return c.get(QUEUED, 0) + c.get(RUNNING, 0)

    # --- worker side -------------------------------------------------------

    def claim_next(self) -> Job | None:
        """Atomically take the oldest queued job and mark it running."""
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY seq LIMIT 1", (QUEUED,)
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                "UPDATE jobs SET status = ?, stage = 'starting', progress = 0, started_at = ? WHERE id = ?",
                (RUNNING, time.time(), row["id"]),
            )
            return self.get(row["id"])

    def set_progress(self, job_id: str, stage: str, progress: float) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET stage = ?, progress = ? WHERE id = ?",
                (stage, max(0.0, min(1.0, progress)), job_id),
            )

    def set_text(self, job_id: str, caption: str, prompt: str) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET caption = ?, prompt = ? WHERE id = ?", (caption, prompt, job_id))

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, stage = 'done', progress = 1, result = ?, finished_at = ? WHERE id = ?",
                (DONE, json.dumps(result), time.time(), job_id),
            )

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (FAILED, error[:2000], time.time(), job_id),
            )

    def mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status = ?, stage = 'cancelled', finished_at = ? WHERE id = ?",
                (CANCELLED, time.time(), job_id),
            )

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row is None or bool(row["cancel_requested"])

    def requeue_interrupted(self) -> int:
        """Called on startup: jobs left 'running' by a crash go back in the queue."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status = ?, stage = '', progress = 0, started_at = NULL "
                "WHERE status = ? AND cancel_requested = 0",
                (QUEUED, RUNNING),
            )
            self._db.execute(
                "UPDATE jobs SET status = ?, finished_at = ? WHERE status = ? AND cancel_requested = 1",
                (CANCELLED, time.time(), RUNNING),
            )
            return cur.rowcount

    # --- client side -------------------------------------------------------

    def request_cancel(self, job_id: str) -> Job | None:
        """Queued jobs are cancelled immediately; running jobs stop at the next stage boundary."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            if job.status == QUEUED:
                self._db.execute(
                    "UPDATE jobs SET status = ?, cancel_requested = 1, finished_at = ? WHERE id = ?",
                    (CANCELLED, time.time(), job_id),
                )
            elif job.status == RUNNING:
                self._db.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
            return self.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def expired(self, max_age_seconds: float) -> list[str]:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?, ?) AND COALESCE(finished_at, created_at) < ?",
                (DONE, FAILED, CANCELLED, cutoff),
            ).fetchall()
        return [r["id"] for r in rows]

    # --- helpers -----------------------------------------------------------

    def _average_duration(self) -> float | None:
        rows = self._db.execute(
            "SELECT finished_at - started_at AS d FROM jobs WHERE status = ? AND started_at IS NOT NULL "
            "ORDER BY seq DESC LIMIT 10",
            (DONE,),
        ).fetchall()
        durations = [r["d"] for r in rows if r["d"] and r["d"] > 0]
        return sum(durations) / len(durations) if durations else None

    def _annotate_queue_info(self, job: Job) -> None:
        if job.status not in (QUEUED, RUNNING):
            return
        avg = self._average_duration()
        running = self._db.execute("SELECT started_at FROM jobs WHERE status = ?", (RUNNING,)).fetchone()
        remaining_running = 0.0
        if running is not None and avg is not None and running["started_at"]:
            remaining_running = max(0.0, avg - (time.time() - running["started_at"]))

        if job.status == RUNNING:
            job.position = 0
            if avg is not None:
                job.eta_seconds = int(remaining_running)
            return

        queued_ahead = self._db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ? AND seq < ?", (QUEUED, job.seq)
        ).fetchone()["n"]
        job.position = queued_ahead + (1 if running is not None else 0)
        if avg is not None:
            job.eta_seconds = int(remaining_running + (queued_ahead + 1) * avg)


def new_job_id() -> str:
    return uuid.uuid4().hex
