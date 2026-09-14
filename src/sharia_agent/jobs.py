"""In-process job store for the async assessment path.

Deliberately simple, and deliberately called out as a shortcut: this store dies
with the process and does not survive a second replica. Production replaces it
with Redis or a real queue — see the technical document, 'What we cut'. The
interface is kept narrow so that swap is a single-file change.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    trace_id: str = "-"
    principal_id: str = "-"


class JobStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._jobs: dict[str, Job] = {}
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def create(self, *, trace_id: str, principal_id: str) -> Job:
        async with self._lock:
            self._evict()
            job = Job(id=uuid.uuid4().hex, trace_id=trace_id, principal_id=principal_id)
            self._jobs[job.id] = job
            return job

    async def get(self, job_id: str, *, principal_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            # A job is readable only by the principal that created it. Without
            # this an assessment id is a capability anyone can guess at.
            if job is None or job.principal_id != principal_id:
                return None
            return job

    async def update(self, job_id: str, **fields: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            if job.status in ("succeeded", "failed"):
                job.finished_at = time.time()

    def _evict(self) -> None:
        cutoff = time.time() - self._ttl
        for job_id in [j.id for j in self._jobs.values() if j.created_at < cutoff]:
            self._jobs.pop(job_id, None)

    @property
    def size(self) -> int:
        return len(self._jobs)
