"""HTTP validation, real inference, persistence, bounded uploads, and concurrency."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

import jigsaw_ai.api as api
from jigsaw_ai.api_uploads import MAX_BODY_BYTES, MAX_FILE_BYTES
from jigsaw_ai.boundary_model import BoundaryCNN
from jigsaw_ai.pairs import DIRECTION_IDS
from jigsaw_ai.pipeline import reconstruct


def image_bytes(mode="RGB", size=(64, 64), format="PNG", color=None):
    image = Image.new(mode, size, color)
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def uploads(count=25):
    return [("pieces", (f"piece_{i:03}.png", image_bytes(color=(i * 10, 30, 60)), "image/png")) for i in range(count)]


@pytest.fixture
def configured(tmp_path):
    torch.manual_seed(42)
    checkpoint = tmp_path / "model.pt"
    torch.save({"architecture": "boundary_cnn_v1", "model_state": BoundaryCNN().state_dict(),
                "config": {"direction_ids": DIRECTION_IDS}}, checkpoint)
    application = api.create_app(checkpoint, tmp_path / "results")
    with TestClient(application) as client:
        yield client, application, checkpoint, tmp_path / "results"


def test_real_cnn_request_matches_direct_pipeline_and_preserves_weights(configured):
    client, application, _, root = configured
    assert client.get("/health").json() == {"status": "ok", "cnn_available": True, "classical_available": True}
    files = uploads()
    before = {k: v.clone() for k, v in application.state.model.state_dict().items()}
    response = client.post("/reconstructions", files=files)
    assert response.status_code == 201, response.text
    result = response.json()
    assert UUID(result["id"]).version == 4
    assert result["scorer"] == "cnn" and result["search"] == "beam"
    assert result["checkpoint_sha256"]
    assert "accuracy" not in result and "confidence" not in result
    placements = result["arrangement"]
    assert sorted(p["upload_index"] for p in placements) == list(range(25))
    assert [(p["row"], p["column"]) for p in placements] == [divmod(i, 5) for i in range(25)]
    pieces = [Image.open(BytesIO(file[1][1])).copy() for file in files]
    expected = reconstruct(pieces, application.state.model, "beam")
    assert [p["upload_index"] for p in placements] == expected["order_indices"]
    assert result["mean_border_cost"] == expected["mean_border_cost"]
    image_response = client.get(result["image_url"])
    assert image_response.status_code == 200 and image_response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(image_response.content)) as image:
        assert image.size == (320, 320) and image.tobytes() == expected["image"].tobytes()
    assert client.get(f"/reconstructions/{result['id']}").json() == result
    assert all(torch.equal(before[k], v) for k, v in application.state.model.state_dict().items())
    assert sorted(p.name for p in (root / result["id"]).iterdir()) == ["image.png", "result.json"]


@pytest.mark.parametrize("count", [24, 26])
def test_piece_count_validation(configured, count):
    client, _, _, root = configured
    assert client.post("/reconstructions", files=uploads(count)).status_code == 422
    assert not list(root.iterdir())


@pytest.mark.parametrize("data", [{"scorer": "unknown"}, {"search": "unknown"}])
def test_invalid_method_rejected(configured, data):
    client, _, _, _ = configured
    assert client.post("/reconstructions", files=uploads(), data=data).status_code == 422


@pytest.mark.parametrize("data", [b"", b"not an image", image_bytes(size=(32, 64)),
                                   image_bytes(mode="RGBA"), image_bytes(format="GIF"), image_bytes()[:30]])
def test_bad_image_rejected_and_lock_released(configured, data):
    client, _, _, root = configured
    files = uploads()
    files[0] = ("pieces", ("bad.png", data, "image/png"))
    response = client.post("/reconstructions", files=files)
    assert response.status_code == 422, response.text
    assert not list(root.iterdir())
    assert client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).status_code == 201


def test_file_and_request_limits_include_chunked_bodies(configured):
    client, _, _, _ = configured
    files = uploads()
    files[0] = ("pieces", ("large.png", b"x" * (MAX_FILE_BYTES + 1), "image/png"))
    assert client.post("/reconstructions", files=files).status_code == 413
    assert client.post("/reconstructions", content=b"x", headers={"content-length": str(MAX_BODY_BYTES + 1)}).status_code == 413
    chunks = iter([b"x" * (MAX_BODY_BYTES // 2), b"y" * (MAX_BODY_BYTES // 2 + 1)])
    assert client.post("/reconstructions", content=chunks).status_code == 413
    assert client.post("/reconstructions", content=b"x", headers={"content-length": "invalid"}).status_code == 400


def test_missing_or_corrupt_model_does_not_silently_change_scorer(tmp_path):
    corrupt = tmp_path / "bad.pt"
    corrupt.write_bytes(b"invalid checkpoint")
    for checkpoint in (tmp_path / "missing.pt", corrupt):
        with TestClient(api.create_app(checkpoint, tmp_path / checkpoint.stem)) as client:
            assert client.get("/health").json()["cnn_available"] is False
            assert client.post("/reconstructions", files=uploads()).status_code == 503
            result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"})
            assert result.status_code == 201
            assert result.json()["checkpoint_sha256"] is None


def test_duplicate_and_pathlike_names_are_labels_not_disk_paths(configured):
    client, _, _, root = configured
    files = [("pieces", ("../../same.png", image_bytes(), "application/octet-stream")) for _ in range(25)]
    response = client.post("/reconstructions", files=files, data={"scorer": "classical", "search": "greedy"})
    assert response.status_code == 201
    result = response.json()
    assert {p["filename"] for p in result["arrangement"]} == {"same.png"}
    assert len({p["upload_index"] for p in result["arrangement"]}) == 25
    assert not (root.parent / "same.png").exists()


def test_results_survive_application_restart_and_unknown_ids_are_clear(tmp_path):
    root, checkpoint = tmp_path / "results", tmp_path / "missing.pt"
    with TestClient(api.create_app(checkpoint, root)) as client:
        files = [("pieces", (f"{i}.jpg", image_bytes(format="JPEG"), "image/jpeg")) for i in range(25)]
        response = client.post("/reconstructions", files=files, data={"scorer": "classical", "search": "greedy"})
        assert response.status_code == 201
        result = response.json()
    with TestClient(api.create_app(checkpoint, root)) as client:
        assert client.get(f"/reconstructions/{result['id']}").json() == result
        assert client.get(result["image_url"]).status_code == 200
        assert client.get(f"/reconstructions/{uuid4()}").status_code == 404
        assert client.get("/reconstructions/not-a-uuid").status_code == 422


def test_storage_failure_removes_partial_result_and_allows_retry(configured, monkeypatch):
    client, _, _, root = configured
    original = api.write_json
    def broken_write(*args, **kwargs):
        raise OSError("private disk failure")
    monkeypatch.setattr(api, "write_json", broken_write)
    response = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"})
    assert response.status_code == 500 and "private disk failure" not in response.text
    assert not list(root.iterdir())
    monkeypatch.setattr(api, "write_json", original)
    assert client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).status_code == 201


def test_missing_image_and_corrupt_metadata_are_reported(configured):
    client, _, _, root = configured
    result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
    directory = root / result["id"]
    (directory / "image.png").unlink()
    assert client.get(result["image_url"]).status_code == 404
    (directory / "result.json").write_text("{broken", encoding="utf-8")
    response = client.get(f"/reconstructions/{result['id']}")
    assert response.status_code == 500 and "could not be read" in response.json()["detail"]


def test_health_stays_responsive_during_compute_and_busy_returns_retry(configured, monkeypatch):
    client, _, _, _ = configured
    started, finish = Event(), Event()
    original = api.reconstruct
    def slow_compute(*args, **kwargs):
        started.set()
        assert finish.wait(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "reconstruct", slow_compute)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"})
        try:
            assert started.wait(timeout=5)
            assert client.get("/health").status_code == 200
            busy = client.post("/reconstructions", files=uploads())
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
        finally:
            finish.set()
        assert pending.result(timeout=10).status_code == 201


def test_openapi_describes_upload_contract(configured):
    client, _, _, _ = configured
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/reconstructions" in schema["paths"]
    content = schema["paths"]["/reconstructions"]["post"]["requestBody"]["content"]
    reference = content["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    files = schema["components"]["schemas"][reference]["properties"]["pieces"]
    assert files["minItems"] == files["maxItems"] == 25
