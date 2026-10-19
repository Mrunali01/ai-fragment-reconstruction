"""Dataset checks use generated local fixtures, never public network requests."""

import json
from pathlib import Path
from urllib.error import URLError

from PIL import Image
import pytest

from jigsaw_ai import acquire_images as acquisition
from jigsaw_ai import audit_dataset as auditing
from jigsaw_ai.image_validation import inspect_image


def test_image_dimensions_and_decoded_duplicates(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    image = Image.new("RGB", (400, 320), "red")
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    a, b = inspect_image(first), inspect_image(second)
    assert (a["width"], a["height"]) == (400, 320)
    assert a["sha256"] != b["sha256"]
    assert a["pixel_sha256"] == b["pixel_sha256"]


def test_reject_small_and_corrupt_images(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (319, 500)).save(path)
    with pytest.raises(ValueError, match="too small"):
        inspect_image(path)
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        inspect_image(path)


def test_truncated_jpeg_rejected(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (400, 400)).save(path)
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(OSError):
        inspect_image(path)


def test_exif_orientation(tmp_path):
    path = tmp_path / "portrait.jpg"
    image = Image.new("RGB", (400, 320))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    result = inspect_image(path)
    assert (result["width"], result["height"]) == (320, 400)


def test_candidate_order_reproducible():
    rows = [{"ImageID": str(i), "Subset": "train"} for i in range(20)]
    assert acquisition.candidate_order(rows, 42) == acquisition.candidate_order(rows[::-1], 42)
    assert acquisition.candidate_order(rows, 42) != acquisition.candidate_order(rows, 43)


def test_license_requires_matching_photo_and_license():
    url = "https://www.flickr.com/photos/author/123"
    evidence = {"@type": "ImageObject", "acquireLicensePage": url,
                "license": acquisition.LICENSE, "author": {"name": "Artist"}}
    def page():
        return '<script type="application/ld+json">' + json.dumps({"@graph": [evidence]}) + '</script>'
    assert acquisition.license_evidence(page(), url + "/") == evidence
    with pytest.raises(ValueError, match="No matching"):
        acquisition.license_evidence(page(), url + "4")
    evidence["license"] = "all rights reserved"
    with pytest.raises(ValueError, match="no longer"):
        acquisition.license_evidence(page(), url)


def test_transient_download_retries_and_limits(monkeypatch):
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return b"abc"
    def open_url(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise URLError("temporary outage")
        return Response()
    monkeypatch.setattr(acquisition, "urlopen", open_url)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)
    assert acquisition.fetch("https://example.org/image") == b"abc"
    assert len(calls) == 3
    with pytest.raises(ValueError, match="exceeds"):
        acquisition.fetch("https://example.org/image", limit=2)


def test_invalid_candidate_logged_without_network(tmp_path):
    result = acquisition.acquire_one(tmp_path, {"ImageID": "../escape", "Subset": "train"}, 320)
    assert "Invalid image ID" in result["error"]


def test_manifest_split_separation(tmp_path):
    records = [{"image_id": "one", "split": "train"}, {"image_id": "two", "split": "test"}]
    acquisition.save_manifest(tmp_path, records)
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert splits == {"train": ["one"], "validation": [], "test": ["two"]}


def test_acquisition_duplicates_failures_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "TARGETS", {"train": 2, "validation": 1, "test": 1})
    monkeypatch.setattr(acquisition, "candidate_order", lambda rows, seed: rows)
    def metadata(root, split, size):
        return [{"ImageID": f"{split}{i}", "Subset": split} for i in range(5)]
    monkeypatch.setattr(acquisition, "snapshot_metadata", metadata)
    calls = []
    def one(root, row, minimum):
        image_id, split = row["ImageID"], row["Subset"]
        calls.append(image_id)
        if image_id == "train0":
            return {"image_id": image_id, "split": split, "error": "Simulated download failure"}
        # One validation candidate duplicates an accepted training image.
        pixel = "train1" if image_id == "validation0" else image_id
        return {"image_id": image_id, "split": split, "path": "fixture.png",
                "sha256": pixel, "pixel_sha256": pixel, "source_url": image_id,
                "bytes": 10}
    monkeypatch.setattr(acquisition, "acquire_one", one)
    monkeypatch.setattr(acquisition, "inspect_image", lambda path, minimum: {"bytes": 10})
    acquisition.acquire(tmp_path, pool_size=5, workers=2)
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert splits == {"train": ["train1", "train2"], "validation": ["validation1"], "test": ["test0"]}
    assert len(json.loads((tmp_path / "failures.json").read_text())) == 2
    original_manifest = (tmp_path / "manifest.jsonl").read_bytes()
    calls.clear()
    acquisition.acquire(tmp_path, pool_size=5, workers=2)
    assert calls == []
    assert (tmp_path / "manifest.jsonl").read_bytes() == original_manifest
    with pytest.raises(ValueError, match="different acquisition settings"):
        acquisition.acquire(tmp_path, seed=43, pool_size=5, workers=2)


def test_audit_detects_tampering_and_split_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(auditing, "TARGETS", {"train": 1})
    (tmp_path / "images").mkdir()
    (tmp_path / "licenses").mkdir()
    image_id = "0123456789abcdef"
    path = tmp_path / "images" / f"{image_id}.jpg"
    Image.new("RGB", (400, 320), "red").save(path)
    url = "https://www.flickr.com/photos/artist/123"
    evidence = {"image": {"@type": "ImageObject", "acquireLicensePage": url,
                          "license": acquisition.LICENSE, "author": {"name": "Artist"}}}
    acquisition.write_json(tmp_path / "licenses" / f"{image_id}.json", evidence)
    record = {"image_id": image_id, "split": "train", "path": f"images/{image_id}.jpg",
              "source_url": url, "license": acquisition.LICENSE, "author": "Artist",
              "license_evidence": f"licenses/{image_id}.json", **inspect_image(path)}
    acquisition.save_manifest(tmp_path, [record])
    acquisition.write_json(tmp_path / "config.json", {"minimum_side": 320})
    assert auditing.audit(tmp_path)["images"] == 1
    acquisition.write_json(tmp_path / "splits.json", {"train": ["wrong"]})
    with pytest.raises(ValueError, match="Split assignment mismatch"):
        auditing.audit(tmp_path)
    acquisition.save_manifest(tmp_path, [record])
    Image.new("RGB", (400, 320), "blue").save(path)
    with pytest.raises(ValueError, match="Image details changed"):
        auditing.audit(tmp_path)


def test_atomic_write_survives_brief_windows_file_lock(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text("old data", encoding="utf-8")
    original_replace = Path.replace
    attempts = []
    def locked_replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert target.read_text(encoding="utf-8") == "old data"
            raise PermissionError("Simulated transient Windows file lock")
        return original_replace(source, destination)
    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)
    acquisition.write_json(target, {"complete": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True}
    assert len(attempts) == 3
