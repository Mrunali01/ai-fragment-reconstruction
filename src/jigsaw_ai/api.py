"""Phase 9 FastAPI service. Start with: python -m uvicorn jigsaw_ai.api:app"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import pickle
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
import torch
import psycopg

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.api_schemas import HealthResponse, Placement, ReconstructionResponse, Scorer, Search
from jigsaw_ai.api_uploads import MAX_FILE_BYTES, UploadBodyLimit, decode_fragment, display_filename
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.pipeline import reconstruct
from jigsaw_ai.solve_boundary import load_boundary_checkpoint
from jigsaw_ai.database import PostgresStore
from jigsaw_ai.workspace_storage import WorkspaceStorage
from jigsaw_ai.workspace_routes import workspace_router
from jigsaw_ai.background_jobs import JobRunner, jobs_router
from jigsaw_ai.pipeline import score_pieces
from jigsaw_ai.diagnostics import compatibility_report
from jigsaw_ai.exports import export_router

logger = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = Path("artifacts/cnn_experiments/boundary_v1/ranking/best.pt")


def store_result(root: Path, result: ReconstructionResponse, image) -> None:
    directory = root / str(result.id)
    directory.mkdir()  # Refuse collisions rather than overwrite an existing result.
    try:
        image.save(directory / "image.png")
        write_json(directory / "result.json", result.model_dump(mode="json"))
    except Exception:
        # This directory was just created here and contains only these known files.
        for name in ("image.png", "result.json", "result.json.tmp"):
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()
        raise


def execute_reconstruction(application, decoded, inputs, scorer, search, solver, *, source_result_id=None, source_order=None):
    started = perf_counter()
    result_id = uuid4()
    store = application.state.store
    if store:
        store.start(result_id, inputs, scorer, search)
    try:
        model = application.state.model if scorer == "cnn" else None
        output = solver(decoded, model, search)
        result = ReconstructionResponse(
            id=result_id, created_at=datetime.now(timezone.utc), scorer=scorer, search=search,
            arrangement=[Placement(row=position // 5, column=position % 5, upload_index=index, **inputs[index])
                         for position, index in enumerate(output["order_indices"])],
            mean_border_cost=output["mean_border_cost"],
            checkpoint_sha256=application.state.checkpoint_sha256 if scorer == "cnn" else None,
            image_url=f"/reconstructions/{result_id}/image",
            diagnostics=compatibility_report(score_pieces(decoded, model), output["order_indices"], scorer),
            duration_seconds=perf_counter() - started,
            source_result_id=source_result_id, source_order=source_order,
        )
        store_result(application.state.results_dir, result, output["image"])
        if store:
            try:
                store.complete(result)
            except Exception:
                directory = application.state.results_dir / str(result_id)
                for name in ("image.png", "result.json"):
                    (directory / name).unlink(missing_ok=True)
                directory.rmdir()
                raise
        return result
    except Exception:
        if store:
            try:
                store.fail(result_id)
            except psycopg.Error:
                logger.exception("Could not mark failed job %s", result_id)
        raise


def create_app(checkpoint: Path | None = None, results_dir: Path | None = None,
               database_url: str | None = None) -> FastAPI:
    checkpoint = checkpoint if checkpoint is not None else Path(os.environ.get("JIGSAW_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    results_dir = results_dir if results_dir is not None else Path(os.environ.get("JIGSAW_RESULTS_DIR", "artifacts/api_results"))
    database_url = database_url if database_url is not None else os.environ.get("JIGSAW_DATABASE_URL")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        root = results_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        application.state.results_dir = root
        application.state.compute_lock = Lock()
        application.state.store = PostgresStore(database_url) if database_url else None
        if application.state.store:
            # A configured database must work; never silently lose persistence.
            application.state.store.initialize()
        application.state.workspace = WorkspaceStorage(root, application.state.store)
        application.state.model = None
        application.state.checkpoint_sha256 = None
        torch.set_num_threads(2)
        try:
            model, _ = load_boundary_checkpoint(checkpoint)
            application.state.model = model
            application.state.checkpoint_sha256 = sha256(checkpoint)
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as error:
            logger.warning("CNN checkpoint unavailable: %s", error)
        application.state.jobs = JobRunner(application.state.workspace, application.state.compute_lock,
            lambda pieces, inputs, scorer, search, **provenance: execute_reconstruction(application, pieces, inputs, scorer, search, reconstruct, **provenance))
        try:
            yield
        finally:
            await run_in_threadpool(application.state.jobs.close)

    application = FastAPI(title="Jigsaw AI", version="0.1.0", lifespan=lifespan,
                          description="Reconstruct 25 upright RGB 64x64 square fragments. Results are estimates, not guaranteed correct.")
    application.add_middleware(UploadBodyLimit)

    @application.exception_handler(psycopg.Error)
    async def database_unavailable(request, error):
        logger.error("Workspace database operation failed", exc_info=error)
        return JSONResponse({"detail": "Database temporarily unavailable"}, status_code=503)

    def load_result(result_id: UUID) -> ReconstructionResponse:
        if application.state.store:
            try:
                result = application.state.store.get(result_id)
            except psycopg.Error as error:
                logger.exception("Database read failed")
                raise HTTPException(503, "Database temporarily unavailable") from error
            if result is None:
                raise HTTPException(404, "Reconstruction not found")
            return result
        path = application.state.results_dir / str(result_id) / "result.json"
        try:
            result = ReconstructionResponse.model_validate_json(path.read_text(encoding="utf-8"))
            if result.id != result_id:
                raise ValueError("Result ID disagrees with stored record")
            return result
        except FileNotFoundError as error:
            raise HTTPException(404, "Reconstruction not found") from error
        except (OSError, ValueError, ValidationError) as error:
            logger.exception("Cannot read reconstruction %s", result_id)
            raise HTTPException(500, "Stored reconstruction could not be read") from error

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        if application.state.store:
            try:
                with application.state.store.connect() as connection:
                    connection.execute("SELECT 1")
            except psycopg.Error as error:
                raise HTTPException(503, "Database temporarily unavailable") from error
        return HealthResponse(cnn_available=application.state.model is not None)

    @application.post("/reconstructions", response_model=ReconstructionResponse, status_code=201,
                      responses={413: {"description": "File or request size limit exceeded"},
                                 503: {"description": "CNN unavailable or solver busy"},
                                 500: {"description": "Reconstruction or result storage failed"}})
    async def create_reconstruction(
        request: Request,
        pieces: Annotated[list[UploadFile], File(min_length=25, max_length=25, description="Exactly 25 RGB 64x64 PNG/JPEG fragments, in shuffled input order")],
        scorer: Annotated[Scorer, Form()] = "cnn",
        search: Annotated[Search, Form()] = "beam",
    ) -> ReconstructionResponse:
        if scorer == "cnn" and application.state.model is None:
            raise HTTPException(503, "CNN checkpoint unavailable. Configure the trained checkpoint or select the classical solver.")
        lock = application.state.compute_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Solver is busy; retry shortly", headers={"Retry-After": "1"})
        try:
            decoded, inputs = [], []
            for index, upload in enumerate(pieces):
                data = await upload.read(MAX_FILE_BYTES + 1)
                decoded.append(decode_fragment(data, index))
                inputs.append({"filename": display_filename(upload.filename, index),
                               "sha256": hashlib.sha256(data).hexdigest()})

            def compute() -> ReconstructionResponse:
                return execute_reconstruction(application, decoded, inputs, scorer, search, reconstruct)

            return await run_in_threadpool(compute)
        except HTTPException:
            raise
        except psycopg.Error as error:
            logger.exception("Database write failed")
            raise HTTPException(503, "Database temporarily unavailable; no completed result was saved") from error
        except Exception as error:
            logger.exception("Reconstruction failed")
            raise HTTPException(500, "Reconstruction failed; no completed result was saved") from error
        finally:
            lock.release()
            for upload in pieces:
                await upload.close()

    @application.get("/reconstructions/{result_id}", response_model=ReconstructionResponse,
                     responses={404: {"description": "Reconstruction not found"},
                                500: {"description": "Stored reconstruction could not be read"}})
    def result_metadata(result_id: UUID) -> ReconstructionResponse:
        return load_result(result_id)

    @application.get("/reconstructions/{result_id}/image", response_class=FileResponse, name="result_image",
                     responses={404: {"description": "Reconstruction or image not found"},
                                500: {"description": "Stored reconstruction could not be read"}})
    def result_image(result_id: UUID) -> FileResponse:
        load_result(result_id)
        path = application.state.results_dir / str(result_id) / "image.png"
        if not path.is_file():
            raise HTTPException(404, "Reconstruction image not found")
        return FileResponse(path, media_type="image/png", filename=f"{result_id}.png")

    application.include_router(workspace_router(application, load_result))
    application.include_router(jobs_router(application))
    application.include_router(export_router(application, load_result))
    return application


app = create_app()
