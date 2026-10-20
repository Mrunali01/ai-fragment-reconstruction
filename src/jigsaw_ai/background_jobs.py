"""Bounded local worker. Persistent state, explicit interruption; no distributed queue."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import logging
from threading import BoundedSemaphore
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from jigsaw_ai.api_schemas import Scorer, Search
from jigsaw_ai.api_uploads import MAX_FILE_BYTES, decode_fragment, display_filename
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


class JobRunner:
    def __init__(self, storage, compute_lock, compute):
        self.storage, self.compute_lock, self.compute = storage, compute_lock, compute
        self.slots = BoundedSemaphore(2)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jigsaw-solver")
        offset = 0
        while True:
            records = storage.records("job", limit=100, offset=offset)
            for record in records:
                if record["status"] in ("queued", "running"):
                    storage.save("job", {**record, "status": "failed", "error": "Interrupted by server restart; please submit again", "finished_at": now()})
            if len(records) < 100:
                break
            offset += 100

    def submit(self, pieces, inputs, scorer, search, *, source_result_id=None, source_order=None):
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, "The local queue is full; retry shortly", headers={"Retry-After": "2"})
        record = {"id": str(uuid4()), "created_at": now(), "status": "queued", "scorer": scorer, "search": search,
                  "result_id": None, "error": None, "started_at": None, "finished_at": None,
                  "source_result_id": source_result_id, "source_order": source_order}
        try:
            self.storage.save("job", record)
            self.executor.submit(self.run, dict(record), pieces, inputs)
        except Exception:
            self.slots.release()
            raise
        return record

    def run(self, record, pieces, inputs):
        try:
            with self.compute_lock:
                record.update(status="running", started_at=now())
                self.storage.save("job", record)
                result = self.compute(pieces, inputs, record["scorer"], record["search"],
                    source_result_id=record.get("source_result_id"), source_order=record.get("source_order"))
                record.update(status="completed", result_id=str(result.id), finished_at=now())
                self.storage.save("job", record)
        except Exception:
            logger.exception("Background reconstruction failed: %s", record["id"])
            record.update(status="failed", error="Reconstruction or storage failed. Please retry.", finished_at=now())
            try:
                self.storage.save("job", record)
            except Exception:
                logger.exception("Could not persist failed background job")
        finally:
            self.slots.release()

    def close(self):
        self.executor.shutdown(wait=True)


def jobs_router(application):
    router = APIRouter()

    @router.get("/jobs")
    def recent_jobs(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
        return application.state.workspace.records("job", limit=limit, offset=offset)

    @router.post("/jobs", status_code=202)
    async def submit_job(
        pieces: Annotated[list[UploadFile], File(min_length=25, max_length=25)],
        scorer: Annotated[Scorer, Form()] = "cnn", search: Annotated[Search, Form()] = "beam",
    ):
        try:
            if scorer == "cnn" and application.state.model is None:
                raise HTTPException(503, "CNN checkpoint unavailable; select classical edge matching")
            decoded, inputs = [], []
            for index, upload in enumerate(pieces):
                data = await upload.read(MAX_FILE_BYTES + 1)
                decoded.append(decode_fragment(data, index))
                inputs.append({"filename": display_filename(upload.filename, index), "sha256": hashlib.sha256(data).hexdigest()})
            return await run_in_threadpool(application.state.jobs.submit, decoded, inputs, scorer, search)
        finally:
            for upload in pieces:
                await upload.close()

    @router.get("/jobs/{job_id}")
    def job_status(job_id: UUID):
        record = application.state.workspace.get("job", job_id)
        if record is None:
            raise HTTPException(404, "Job not found")
        return record

    return router
