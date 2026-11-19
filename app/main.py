"""FastAPI surface for the checkpoint resharding service.

Flow (all interactive via /docs):
  POST /reshard                              plan & start a conversion
  GET  /jobs/{job_id}                        poll progress (intervals + step)
  POST /jobs/{job_id}/resume                 continue after failure/restart
  GET  /jobs/{job_id}/shards/{rank}/meta     interval + original step per rank
  GET  /jobs/{job_id}/shards/{rank}/download?part=weights|momentum
                                             gated: 409 until the whole job is done
  POST /diagnose                             diff two checkpoints (old vs new):
                                             where did finite values turn NaN/Inf?
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .diagnose import (
    DiagnoseError,
    DiagnoseReadError,
    LengthMismatchError,
    compare_checkpoints,
)
from .engine import DEFAULT_CHUNK_BYTES, execute_job
from .jobs import (
    MOMENTUM_NAME,
    WEIGHTS_NAME,
    Job,
    JobStore,
    RankState,
    compute_job_id,
)
from .manifest import ManifestError, load_manifest
from .plans import rank_intervals
from .readgate import ReadGate


class Settings(BaseModel):
    source_dir: Path = Path(os.environ.get("SOURCE_DIR", "/data/source"))
    next_dir: Path = Path(os.environ.get("NEXT_DIR", "/data/next"))
    output_root: Path = Path(os.environ.get("OUTPUT_ROOT", "/data/outputs"))
    chunk_bytes: int = int(os.environ.get("CHUNK_BYTES", DEFAULT_CHUNK_BYTES))
    read_concurrency: int = int(os.environ.get("READ_CONCURRENCY", "4"))
    sync_execution: bool = os.environ.get("RESHARD_SYNC", "").lower() in {"1", "true", "yes"}


class ReshardRequest(BaseModel):
    new_world_size: int = Field(..., ge=1, description="number of processes to reshard for")
    source_dir: str | None = Field(
        None, description="override the mounted source directory (defaults to SOURCE_DIR)"
    )


class DiagnoseRequest(BaseModel):
    old_dir: str | None = Field(
        None, description="earlier checkpoint directory (defaults to SOURCE_DIR)"
    )
    new_dir: str | None = Field(
        None, description="later checkpoint directory (defaults to NEXT_DIR)"
    )


class CategoryOut(BaseModel):
    count: int = Field(..., description="total new NaN/Inf parameters in this category")
    first_indices: list[int] = Field(
        ..., description="smallest global indices in this category (at most 5)"
    )


class DiagnoseOut(BaseModel):
    old_dir: str
    new_dir: str
    step_old: int
    step_new: int
    total_length: int
    weights_only: CategoryOut
    momentum_only: CategoryOut
    both: CategoryOut


class RankOut(BaseModel):
    rank: int
    start: int
    end: int
    status: str


class JobOut(BaseModel):
    job_id: str
    status: str
    error: str | None
    step: int
    total_length: int
    world_size: int
    done_ranks: int
    created_at: str
    updated_at: str
    ranks: list[RankOut]


def _job_out(job: Job) -> JobOut:
    return JobOut(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        step=job.step,
        total_length=job.total_length,
        world_size=job.world_size,
        done_ranks=job.done_ranks,
        created_at=job.created_at,
        updated_at=job.updated_at,
        ranks=[RankOut(rank=r.rank, start=r.start, end=r.end, status=r.status) for r in job.ranks],
    )


class Runner:
    """Serialises execution per job; threads for async, inline when sync_execution."""

    def __init__(self, store: JobStore, chunk_bytes: int, sync: bool, gate: ReadGate):
        self._store = store
        self._chunk = chunk_bytes
        self._sync = sync
        self._gate = gate
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, job_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def kick(self, job: Job) -> None:
        if job.status == "done":
            return  # nothing left to do
        lock = self._lock_for(job.job_id)
        if job.status == "running" and lock.locked():
            return  # already running elsewhere

        def _run() -> None:
            with lock:
                execute_job(self._store, job, self._chunk, gate=self._gate)

        if self._sync:
            _run()
        else:
            threading.Thread(target=_run, daemon=True, name=f"reshard-{job.job_id}").start()


def create_app(settings: Settings | None = None, gate: ReadGate | None = None) -> FastAPI:
    settings = settings or Settings()
    store = JobStore(settings.output_root)
    # One gate shared by reshard and diagnose chunk reads: READ_CONCURRENCY
    # caps how many reads are in flight across BOTH activities.
    read_gate = gate if gate is not None else ReadGate(settings.read_concurrency)
    runner = Runner(store, settings.chunk_bytes, settings.sync_execution, read_gate)

    app = FastAPI(
        title="Checkpoint Reshard API",
        version="1.1.0",
        description=(
            "Re-shard a float32 training checkpoint (weights + momentum) from a "
            "read-only source volume to a new process count, and diff two "
            "checkpoints to find parameters that newly turned NaN/Inf. Source "
            "segments with gaps or overlaps are rejected. Fully written rank "
            "shards are reused on resume; downloads open only when the whole "
            "job is complete. Reshard and diagnose chunk reads share one "
            "configurable concurrency cap, with priority to reshards."
        ),
    )
    app.state.store = store
    app.state.runner = runner
    app.state.settings = settings
    app.state.gate = read_gate

    def get_store() -> JobStore:
        return store

    def must_job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return job

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reshard", response_model=JobOut, status_code=202)
    def create_reshard(req: ReshardRequest) -> JobOut:
        source_dir = Path(req.source_dir) if req.source_dir else settings.source_dir
        if not source_dir.is_dir():
            raise HTTPException(status_code=422, detail=f"source directory {source_dir} not found")
        try:
            manifest = load_manifest(source_dir)
        except ManifestError as exc:
            # gaps / overlaps / size mismatches all land here
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            intervals = rank_intervals(manifest.total_length, req.new_world_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        job_id = compute_job_id(source_dir, manifest, req.new_world_size)
        job = store.get(job_id)
        if job is None:
            job = Job(
                job_id=job_id,
                source_dir=str(source_dir),
                world_size=req.new_world_size,
                step=manifest.step,
                total_length=manifest.total_length,
                ranks=[RankState(rank=i, start=s, end=e) for i, (s, e) in enumerate(intervals)],
            )
            store.add(job)
        runner.kick(job)
        return _job_out(job)

    @app.get("/jobs", response_model=list[JobOut])
    def list_jobs() -> list[JobOut]:
        return [_job_out(j) for j in store.all()]

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job: Job = Depends(must_job)) -> JobOut:
        return _job_out(job)

    @app.post("/jobs/{job_id}/resume", response_model=JobOut, status_code=202)
    def resume_job(job: Job = Depends(must_job)) -> JobOut:
        if job.status == "done":
            return _job_out(job)  # nothing to do
        runner.kick(job)
        return _job_out(job)

    @app.get("/jobs/{job_id}/shards/{rank}/meta")
    def shard_meta(rank: int, job: Job = Depends(must_job)) -> dict:
        if not 0 <= rank < job.world_size:
            raise HTTPException(status_code=404, detail=f"rank {rank} out of range")
        r = job.ranks[rank]
        return {
            "job_id": job.job_id,
            "rank": r.rank,
            "start": r.start,
            "end": r.end,
            "step": job.step,
            "status": r.status,
        }

    @app.get("/jobs/{job_id}/shards/{rank}/download")
    def download_shard(
        rank: int,
        part: Literal["weights", "momentum"] = Query(...),
        job: Job = Depends(must_job),
    ) -> FileResponse:
        if not 0 <= rank < job.world_size:
            raise HTTPException(status_code=404, detail=f"rank {rank} out of range")
        if job.status != "done":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"job is {job.status}: downloads open only when all "
                    f"{job.world_size} shards are complete ({job.done_ranks} done)"
                ),
            )
        name = WEIGHTS_NAME if part == "weights" else MOMENTUM_NAME
        path = store.rank_dir(job.job_id, rank) / name
        if not path.is_file():
            raise HTTPException(status_code=409, detail=f"shard {rank} {part} not on disk yet")
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{job.job_id}-rank{rank:05d}-{name}")

    @app.post("/diagnose", response_model=DiagnoseOut)
    def diagnose(req: DiagnoseRequest) -> DiagnoseOut:
        """Compare two checkpoints element-wise and report NEW NaN/Inf params.

        Reads both checkpoints chunk by chunk from the read-only volumes,
        aligned by global index regardless of how each of the four tensor
        groups is sharded.  All-or-nothing: any failure returns only an
        error, never partial counts.
        """
        old_dir = Path(req.old_dir) if req.old_dir else settings.source_dir
        new_dir = Path(req.new_dir) if req.new_dir else settings.next_dir
        for label, d in (("old_dir", old_dir), ("new_dir", new_dir)):
            if not d.is_dir():
                raise HTTPException(status_code=422, detail=f"{label} {d} not found")
        try:
            report = compare_checkpoints(old_dir, new_dir, read_gate, settings.chunk_bytes)
        except ManifestError as exc:
            # gaps / overlaps / size mismatches in either checkpoint
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LengthMismatchError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DiagnoseReadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except DiagnoseError as exc:  # pragma: no cover — defensive catch-all
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return DiagnoseOut(
            old_dir=str(old_dir),
            new_dir=str(new_dir),
            step_old=report.step_old,
            step_new=report.step_new,
            total_length=report.total_length,
            weights_only=CategoryOut(
                count=report.weights_only.count,
                first_indices=report.weights_only.first_indices,
            ),
            momentum_only=CategoryOut(
                count=report.momentum_only.count,
                first_indices=report.momentum_only.first_indices,
            ),
            both=CategoryOut(
                count=report.both.count,
                first_indices=report.both.first_indices,
            ),
        )

    return app


# Run with: uvicorn app.main:create_app --factory
# (no module-level app: importing this module must not touch the filesystem)
