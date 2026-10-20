from io import BytesIO
from uuid import uuid4
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from jigsaw_ai.api import create_app
from test_api import uploads
from threading import Event
import time
import numpy as np
from jigsaw_ai.diagnostics import compatibility_report
from zipfile import ZipFile
import csv
import json
from jigsaw_ai.exports import arrangement_csv
from test_database import store


def test_corrections_are_permutations_persist_and_preserve_original(tmp_path):
    root = tmp_path / "results"
    application = create_app(tmp_path / "missing.pt", root)
    with TestClient(application) as client:
        result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
        base = client.get(result["image_url"]).content
        order = [piece["upload_index"] for piece in result["arrangement"]]
        order[0], order[1] = order[1], order[0]
        response = client.post(f"/reconstructions/{result['id']}/corrections", json={"order": order, "note": "swap"})
        assert response.status_code == 201, response.text
        saved = response.json()
        assert saved["source"] == "human" and saved["verified_accuracy"] is None
        corrected = Image.open(BytesIO(client.get(saved["image_url"]).content))
        original = Image.open(BytesIO(base))
        assert corrected.crop((0, 0, 64, 64)).tobytes() == original.crop((64, 0, 128, 64)).tobytes()
        assert client.get(result["image_url"]).content == base
        for bad in ([0] * 25, list(range(24)), list(range(1, 26))):
            assert client.post(f"/reconstructions/{result['id']}/corrections", json={"order": bad}).status_code == 422
        assert client.get(f"/reconstructions/{uuid4()}/corrections/{saved['id']}/image").status_code == 404
    with TestClient(create_app(tmp_path / "missing.pt", root)) as client:
        assert client.get(f"/reconstructions/{result['id']}/corrections").json() == [saved]
        assert client.get(saved["image_url"]).status_code == 200
        history = client.get("/history?limit=1").json()
        assert len(history["items"]) == 1 and history["items"][0]["id"] == result["id"]
        assert history["has_more"] is False
        assert client.get("/history?offset=1").json()["items"] == []
        assert client.get("/history?limit=101").status_code == 422


def test_background_queue_bounded_failure_and_restart(tmp_path, monkeypatch):
    application = create_app(tmp_path / "missing.pt", tmp_path / "results")
    entered, release = Event(), Event()
    import jigsaw_ai.api as api
    original = api.reconstruct
    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)
    monkeypatch.setattr(api, "reconstruct", blocked)
    with TestClient(application) as client:
        try:
            first = client.post("/jobs", files=uploads(), data={"scorer": "classical", "search": "greedy"})
            assert first.status_code == 202
            assert entered.wait(5)
            second = client.post("/jobs", files=uploads(), data={"scorer": "classical", "search": "greedy"})
            assert second.status_code == 202
            assert client.post("/jobs", files=uploads(), data={"scorer": "classical"}).status_code == 429
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        for row in (first.json(), second.json()):
            deadline = time.monotonic() + 10
            while True:
                state = client.get(f"/jobs/{row['id']}").json()
                if state["status"] in ("completed", "failed"):
                    break
                assert time.monotonic() < deadline
                time.sleep(.02)
            assert state["status"] == "completed", state
            assert client.get(f"/reconstructions/{state['result_id']}").status_code == 200
        interrupted = {**first.json(), "id": str(uuid4()), "status": "running"}
        application.state.workspace.save("job", interrupted)
    with TestClient(create_app(tmp_path / "missing.pt", tmp_path / "results")) as client:
        state = client.get(f"/jobs/{interrupted['id']}").json()
        assert state["status"] == "failed" and "restart" in state["error"]


def test_relative_evidence_handles_ties_and_negative_margins():
    costs = {direction: np.ones((25, 25)) for direction in ("LEFT", "RIGHT", "UP", "DOWN")}
    tied = compatibility_report(costs, list(range(25)), "classical")
    assert tied["edge_count"] == 40
    assert all(edge["rank"] == 1 and edge["margin"] == 0 for edge in tied["edges"])
    costs["RIGHT"][0, 1] = 2
    report = compatibility_report(costs, list(range(25)), "classical")
    edge = report["edges"][0]
    assert edge["rank"] == 24 and edge["margin"] == pytest.approx(-1/3)
    assert "not calibrated" in report["warning"]


def test_exports_analytics_and_unknown_accuracy(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt", tmp_path / "results")) as client:
        result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
        assert result["duration_seconds"] > 0
        assert result["diagnostics"]["edge_count"] == 40
        assert result["diagnostics"]["mean_border_cost"] == pytest.approx(result["mean_border_cost"])
        path = f"/reconstructions/{result['id']}"
        saved = client.post(path + "/corrections", json={"order": list(range(25))}).json()
        stats = client.get("/analytics").json()
        assert stats["completed_reconstructions"] == 1 and stats["saved_corrections"] == 1
        assert stats["by_scorer"] == {"classical": 1}
        assert stats["timed_reconstructions"] == 1
        assert stats["position_accuracy"] is None and stats["exact_reconstruction_rate"] is None
        report = client.get(path + "/export?format=json").json()
        assert report["ground_truth_available"] is False and report["position_accuracy"] is None
        for correction in (None, saved):
            query = f"&correction_id={correction['id']}" if correction else ""
            response = client.get(path + "/export?format=zip" + query)
            assert response.status_code == 200
            with ZipFile(BytesIO(response.content)) as archive:
                assert sorted(archive.namelist()) == ["arrangement.csv", "metadata.json", "reconstruction.png"]
                meta = json.loads(archive.read("metadata.json"))
                assert meta["selected_correction"] == correction
                image_url = correction["image_url"] if correction else result["image_url"]
                assert archive.read("reconstruction.png") == client.get(image_url).content
                assert len(list(csv.reader(archive.read("arrangement.csv").decode().splitlines()))) == 26
        assert client.get(path + "/export?format=csv").status_code == 200
        assert client.get(path + "/export?format=exe").status_code == 422
        assert client.get(path + f"/export?correction_id={uuid4()}").status_code == 404
        result["arrangement"][0]["filename"] = "=HYPERLINK(bad)"
        assert "'=HYPERLINK" in arrangement_csv(result, [piece["upload_index"] for piece in result["arrangement"]])


def test_postgres_workspace_records_survive_restart(store, tmp_path):
    root = tmp_path / "results"
    with TestClient(create_app(tmp_path / "missing.pt", root, store.url)) as client:
        result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
        saved = client.post(f"/reconstructions/{result['id']}/corrections", json={"order": list(range(25))}).json()
        (root / result["id"] / "result.json").unlink()
        assert client.get("/history").json()["items"][0]["id"] == result["id"]
        assert client.get("/analytics").json()["saved_corrections"] == 1
    with TestClient(create_app(tmp_path / "missing.pt", root, store.url)) as client:
        assert client.get(f"/reconstructions/{result['id']}/corrections").json() == [saved]
        assert client.get(saved["image_url"]).status_code == 200
        assert client.get(f"/reconstructions/{result['id']}/export?correction_id={saved['id']}").status_code == 200
    with store.connect() as connection:
        assert connection.execute("SELECT version FROM jigsaw_schema").fetchall() == [(2,)]


def test_background_failure_is_reported_and_capacity_released(tmp_path, monkeypatch):
    import jigsaw_ai.api as api
    original = api.reconstruct
    def fail(*args):
        raise RuntimeError("private solver detail")
    with TestClient(create_app(tmp_path / "missing.pt", tmp_path / "results")) as client:
        monkeypatch.setattr(api, "reconstruct", fail)
        response = client.post("/jobs", files=uploads(), data={"scorer": "classical"})
        assert response.status_code == 202
        deadline = time.monotonic() + 5
        while True:
            state = client.get(f"/jobs/{response.json()['id']}").json()
            if state["status"] == "failed":
                break
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert "private solver" not in state["error"]
        monkeypatch.setattr(api, "reconstruct", original)
        assert client.post("/jobs", files=uploads(), data={"scorer": "classical", "search": "greedy"}).status_code == 202


def test_reconstruct_current_order_without_reupload_preserves_pixels_and_original(tmp_path, monkeypatch):
    import jigsaw_ai.api as api
    application = create_app(tmp_path / "missing.pt", tmp_path / "results")
    with TestClient(application) as client:
        original = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
        original_png = client.get(original["image_url"]).content
        source = {piece["upload_index"]: piece for piece in original["arrangement"]}
        shuffled = list(reversed(range(25)))
        solver = api.reconstruct
        def inspect_pixels(pieces, model, search):
            assert [piece.getpixel((0, 0)) for piece in pieces] == [(index * 10, 30, 60) for index in shuffled]
            return solver(pieces, model, search)
        monkeypatch.setattr(api, "reconstruct", inspect_pixels)
        response = client.post(f"/reconstructions/{original['id']}/reconstruct", json={"order": shuffled})
        assert response.status_code == 202, response.text
        assert response.json()["source_result_id"] == original["id"]
        deadline = time.monotonic() + 10
        while True:
            job = client.get(f"/jobs/{response.json()['id']}").json()
            if job["status"] in ("failed", "completed"):
                break
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert job["status"] == "completed", job
        result = client.get(f"/reconstructions/{job['result_id']}").json()
        assert result["id"] != original["id"]
        assert result["source_order"] == shuffled and result["source_result_id"] == original["id"]
        image = Image.open(BytesIO(client.get(result["image_url"]).content))
        for piece in result["arrangement"]:
            original_index = shuffled[piece["upload_index"]]
            assert piece["sha256"] == source[original_index]["sha256"]
            assert piece["filename"] == source[original_index]["filename"]
            assert image.getpixel((piece["column"] * 64, piece["row"] * 64)) == (original_index * 10, 30, 60)
        assert client.get(original["image_url"]).content == original_png
        assert len(client.get("/history").json()["items"]) == 2


def test_reconstruct_again_validates_order_method_and_saved_image(tmp_path):
    root = tmp_path / "results"
    with TestClient(create_app(tmp_path / "missing.pt", root)) as client:
        result = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"}).json()
        path = f"/reconstructions/{result['id']}/reconstruct"
        for order in ([0] * 25, list(range(24)), [False] + list(range(1, 25))):
            assert client.post(path, json={"order": order}).status_code == 422
        assert client.post(path, json={"order": list(range(25)), "scorer": "unknown"}).status_code == 422
        assert client.post(path, json={"order": list(range(25)), "scorer": "cnn"}).status_code == 503
        assert client.post(f"/reconstructions/{uuid4()}/reconstruct", json={"order": list(range(25))}).status_code == 404
        Image.new("RGB", (64, 64)).save(root / result["id"] / "image.png")
        assert client.post(path, json={"order": list(range(25))}).status_code == 409
        (root / result["id"] / "image.png").unlink()
        assert client.post(path, json={"order": list(range(25))}).status_code == 404
