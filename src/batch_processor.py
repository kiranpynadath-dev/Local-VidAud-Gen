"""Queue-based batch processor for video + audio generation jobs."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .utils import get_memory_info

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    fn: Callable
    args: tuple
    kwargs: dict
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


@dataclass
class BatchResult:
    jobs: list[Job] = field(default_factory=list)

    @property
    def successful(self) -> list[Job]:
        return [j for j in self.jobs if j.status == JobStatus.DONE]

    @property
    def failed(self) -> list[Job]:
        return [j for j in self.jobs if j.status == JobStatus.FAILED]

    def summary(self) -> str:
        total = len(self.jobs)
        ok = len(self.successful)
        fail = len(self.failed)
        return f"Batch complete: {ok}/{total} succeeded, {fail} failed"


class BatchProcessor:
    """Run a list of generation jobs sequentially or in parallel threads.

    On constrained hardware (AMD Radeon 760M, 32 GB RAM) sequential mode
    is recommended because SVD inference already saturates available VRAM.
    Use parallel mode only for CPU-bound preprocessing / audio generation.
    """

    def __init__(
        self,
        max_workers: int = 1,
        retry_on_oom: bool = True,
        ram_threshold_gb: float = 4.0,
    ) -> None:
        """
        Args:
            max_workers: 1 = sequential (safe for GPU jobs), >1 = threaded.
            retry_on_oom: If OOM is detected, retry once after GC.
            ram_threshold_gb: Pause job queue when free RAM drops below this.
        """
        self.max_workers = max_workers
        self.retry_on_oom = retry_on_oom
        self.ram_threshold_gb = ram_threshold_gb
        self._jobs: list[Job] = []
        self._counter = 0

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_job(
        self,
        fn: Callable,
        *args,
        job_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Enqueue a callable job. Returns the assigned job_id."""
        self._counter += 1
        jid = job_id or f"job_{self._counter:04d}"
        self._jobs.append(Job(job_id=jid, fn=fn, args=args, kwargs=kwargs))
        logger.debug("Enqueued %s → %s(*%s)", jid, fn.__name__, [type(a).__name__ for a in args])
        return jid

    def add_video_job(self, processor, image_path: Path, output_path: Path, **kwargs) -> str:
        return self.add_job(processor.generate_from_image, image_path, output_path, **kwargs)

    def add_audio_job(self, generator, text: str, output_path: Path, **kwargs) -> str:
        return self.add_job(generator.generate_speech, text, output_path, **kwargs)

    def clear(self) -> None:
        self._jobs.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, progress_callback: Optional[Callable[[Job], None]] = None) -> BatchResult:
        """Execute all enqueued jobs and return a BatchResult."""
        if not self._jobs:
            logger.warning("No jobs enqueued")
            return BatchResult()

        logger.info("Starting batch: %d jobs, max_workers=%d", len(self._jobs), self.max_workers)

        if self.max_workers == 1:
            return self._run_sequential(progress_callback)
        return self._run_parallel(progress_callback)

    def _run_sequential(self, progress_callback) -> BatchResult:
        for job in self._jobs:
            self._wait_for_memory()
            self._execute_job(job)
            if progress_callback:
                progress_callback(job)
        return BatchResult(jobs=list(self._jobs))

    def _run_parallel(self, progress_callback) -> BatchResult:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._execute_job, job): job for job in self._jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    # _execute_job already catches exceptions; this is a safety net
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                if progress_callback:
                    progress_callback(job)
        return BatchResult(jobs=list(self._jobs))

    def _execute_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.monotonic()
        logger.info("▶ Running %s …", job.job_id)
        try:
            job.result = job.fn(*job.args, **job.kwargs)
            job.status = JobStatus.DONE
            logger.info("✓ %s completed in %.1fs", job.job_id, job.elapsed or 0)
        except RuntimeError as exc:
            if self.retry_on_oom and "out of memory" in str(exc).lower():
                logger.warning("OOM on %s, retrying after GC …", job.job_id)
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    job.result = job.fn(*job.args, **job.kwargs)
                    job.status = JobStatus.DONE
                    return
                except Exception as retry_exc:
                    exc = retry_exc
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.error("✗ %s failed: %s", job.job_id, exc)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.error("✗ %s failed: %s", job.job_id, exc)
        finally:
            job.finished_at = time.monotonic()

    def _wait_for_memory(self, poll_interval: float = 5.0) -> None:
        while True:
            info = get_memory_info()
            available = info.get("ram_available_gb", float("inf"))
            if available >= self.ram_threshold_gb:
                break
            logger.warning("Low RAM (%.1f GB free < %.1f GB threshold), pausing …", available, self.ram_threshold_gb)
            time.sleep(poll_interval)
