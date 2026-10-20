"""Human arrangements are separate versions, never overwrites of model results."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field, StrictInt, field_validator
from jigsaw_ai.api_schemas import Scorer, Search


class ArrangementInput(BaseModel):
    order: list[StrictInt] = Field(min_length=25, max_length=25)

    @field_validator("order")
    @classmethod
    def permutation(cls, value):
        if sorted(value) != list(range(25)):
            raise ValueError("Order must contain every upload index 0 through 24 exactly once")
        return value


class CorrectionInput(ArrangementInput):
    note: str = Field(default="", max_length=500)


class ReconstructInput(ArrangementInput):
    scorer: Scorer | None = None
    search: Search | None = None


def pieces_from_result(root: Path, result) -> dict[int, Image.Image]:
    """Recover the original decoded pixels from the lossless result mosaic."""
    path = root / str(result.id) / "image.png"
    try:
        with Image.open(path) as source:
            source.load()
            if source.size != (320, 320) or source.mode != "RGB":
                raise HTTPException(409, "Saved image does not match the expected RGB 320x320 puzzle")
            return {piece.upload_index: source.crop((i % 5 * 64, i // 5 * 64, i % 5 * 64 + 64, i // 5 * 64 + 64))
                    for i, piece in enumerate(result.arrangement)}
    except FileNotFoundError as error:
        raise HTTPException(404, "Original result image is missing") from error


def corrected_image(root: Path, result, order: list[int]) -> Image.Image:
    pieces = pieces_from_result(root, result)
    image = Image.new("RGB", (320, 320))
    for position, piece in enumerate(order):
        image.paste(pieces[piece], (position % 5 * 64, position // 5 * 64))
    return image


def workspace_router(application, load_result):
    router = APIRouter()

    @router.get("/history")
    def history(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
        rows = application.state.workspace.history(limit + 1, offset)
        return {"items": rows[:limit], "has_more": len(rows) > limit, "offset": offset, "limit": limit}

    @router.post("/reconstructions/{result_id}/reconstruct", status_code=202)
    def reconstruct_again(result_id: UUID, body: ReconstructInput):
        result = load_result(result_id)
        scorer, search = body.scorer or result.scorer, body.search or result.search
        if scorer == "cnn" and application.state.model is None:
            raise HTTPException(503, "CNN checkpoint unavailable; select classical edge matching")
        pieces = pieces_from_result(application.state.results_dir, result)
        metadata = {piece.upload_index: piece for piece in result.arrangement}
        # Preserve original upload names/hashes as source provenance, not as new uploads.
        inputs = [{"filename": metadata[index].filename, "sha256": metadata[index].sha256} for index in body.order]
        return application.state.jobs.submit([pieces[index] for index in body.order], inputs, scorer, search,
            source_result_id=str(result_id), source_order=body.order)

    @router.post("/reconstructions/{result_id}/corrections", status_code=201)
    def save_correction(result_id: UUID, body: CorrectionInput):
        result = load_result(result_id)
        correction_id = uuid4()
        image = corrected_image(application.state.results_dir, result, body.order)
        directory = application.state.results_dir / str(result_id) / "corrections"
        directory.mkdir(exist_ok=True)
        path = directory / f"{correction_id}.png"
        record = {"id": str(correction_id), "result_id": str(result_id), "created_at": datetime.now(timezone.utc).isoformat(),
                  "order": body.order, "note": body.note, "source": "human", "verified_accuracy": None,
                  "image_url": f"/reconstructions/{result_id}/corrections/{correction_id}/image"}
        try:
            image.save(path)
            application.state.workspace.save("correction", record)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return record

    @router.get("/reconstructions/{result_id}/corrections")
    def corrections(result_id: UUID, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        load_result(result_id)
        return application.state.workspace.records("correction", result_id, limit, offset)

    @router.get("/reconstructions/{result_id}/corrections/{correction_id}/image")
    def correction_image(result_id: UUID, correction_id: UUID):
        load_result(result_id)
        record = application.state.workspace.get("correction", correction_id)
        if record is None or record["result_id"] != str(result_id):
            raise HTTPException(404, "Correction not found")
        path = application.state.results_dir / str(result_id) / "corrections" / f"{correction_id}.png"
        if not path.is_file():
            raise HTTPException(404, "Correction image is missing")
        return FileResponse(path, media_type="image/png", filename=f"{correction_id}.png")

    return router
