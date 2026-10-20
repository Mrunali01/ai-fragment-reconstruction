"""Portable PNG, JSON and CSV reports with explicit unknown accuracy."""
import csv
from io import BytesIO, StringIO
import json
from typing import Literal
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from jigsaw_ai.analytics import application_statistics


def arrangement_csv(result: dict, order: list[int]) -> str:
    pieces = {piece["upload_index"]: piece for piece in result["arrangement"]}
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["row", "column", "upload_index", "filename", "sha256"])
    for position, index in enumerate(order):
        piece = pieces[index]
        name = piece["filename"]
        if name.lstrip().startswith(("=", "+", "-", "@")):
            name = "'" + name
        writer.writerow([position // 5, position % 5, index, name, piece["sha256"]])
    return stream.getvalue()


def export_router(application, load_result):
    router = APIRouter()

    @router.get("/analytics")
    def analytics():
        return application_statistics(application.state.workspace)

    @router.get("/reconstructions/{result_id}/export")
    def export_result(result_id: UUID, format: Literal["json", "csv", "zip"] = "zip", correction_id: UUID | None = None):
        result = load_result(result_id).model_dump(mode="json")
        correction = None
        path = application.state.results_dir / str(result_id) / "image.png"
        order = [piece["upload_index"] for piece in result["arrangement"]]
        if correction_id:
            correction = application.state.workspace.get("correction", correction_id)
            if correction is None or correction["result_id"] != str(result_id):
                raise HTTPException(404, "Correction not found")
            order = correction["order"]
            path = path.parent / "corrections" / f"{correction_id}.png"
        report = {"schema_version": 1, "result": result, "selected_correction": correction,
                  "exported_order": order, "ground_truth_available": False,
                  "position_accuracy": None, "exact_reconstruction": None,
                  "diagnostics_scope": "Original solver arrangement only; human corrections are not automatically verified."}
        metadata = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        csv_data = arrangement_csv(result, order).encode("utf-8")
        if format == "json":
            data, media = metadata, "application/json"
        elif format == "csv":
            data, media = csv_data, "text/csv"
        else:
            if not path.is_file():
                raise HTTPException(404, "Export image is missing")
            stream = BytesIO()
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("metadata.json", metadata)
                archive.writestr("arrangement.csv", csv_data)
                archive.writestr("reconstruction.png", path.read_bytes())
            data, media = stream.getvalue(), "application/zip"
        return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{correction_id or result_id}.{format}"'})

    return router
