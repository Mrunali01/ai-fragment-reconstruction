"""Synthetic images prove orientation, coverage, shuffling, and saved ground truth."""

import json
from pathlib import Path
import random

from PIL import Image
import pytest

from jigsaw_ai.fragments import (
    create_fragments, parse_rotation, prepare_image, rebuild_ground_truth,
)
from jigsaw_ai.generate_puzzles import (
    generate_dataset, generate_puzzle, puzzle_seed, sha256, verify_puzzle,
)


def patterned_image() -> Image.Image:
    image = Image.new("RGB", (320, 320))
    image.putdata([(x % 256, y % 256, (x * 13 + y * 7) % 256)
                   for y in range(320) for x in range(320)])
    return image


def test_fragments_cover_every_pixel_and_match_coordinates():
    original = patterned_image()
    state = random.getstate()
    pieces, positions = create_fragments(original, seed=42)
    assert random.getstate() == state
    assert len(pieces) == 25
    assert len(set(positions)) == 25
    assert positions != [(r, c) for r in range(5) for c in range(5)]
    for piece, (row, column) in zip(pieces, positions):
        assert piece.size == (64, 64)
        assert piece.tobytes() == original.crop((column * 64, row * 64, (column + 1) * 64, (row + 1) * 64)).tobytes()
    assert rebuild_ground_truth(pieces, positions).tobytes() == original.tobytes()
    assert create_fragments(original, 42)[1] == positions
    assert create_fragments(original, 43)[1] != positions


@pytest.mark.parametrize("value,expected", [("270.0", 270), (90, 90), ("0.0", 0), ("", None), ("nan", None), (None, None)])
def test_rotation_metadata(value, expected):
    assert parse_rotation(value) == expected


@pytest.mark.parametrize("value", ["invalid", "45", "inf", -90])
def test_invalid_rotation(value):
    with pytest.raises(ValueError):
        parse_rotation(value)


def test_proportional_resize_and_center_crop(tmp_path):
    path = tmp_path / "wide.png"
    image = Image.new("RGB", (640, 320), "red")
    image.paste("green", (160, 0, 480, 320))
    image.save(path)
    prepared, details = prepare_image(path, rotation=0)
    assert details["resized_size"] == [640, 320]
    assert details["crop_box"] == [160, 0, 480, 320]
    assert prepared.tobytes() == Image.new("RGB", (320, 320), "green").tobytes()
    image.resize((1280, 640)).save(path)
    _, details = prepare_image(path, rotation=0)
    assert details["resized_size"] == [640, 320]


def test_source_rotation_is_counterclockwise_and_does_not_double_apply_exif(tmp_path):
    path = tmp_path / "orientation.png"
    image = Image.new("RGB", (320, 320), "black")
    image.paste("red", (0, 0, 160, 320))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    prepared, _ = prepare_image(path, rotation=90)
    assert prepared.tobytes() == image.rotate(90).tobytes()
    exif_only, details = prepare_image(path)
    assert exif_only.tobytes() == image.rotate(270).tobytes()
    assert details["rotation_ccw"] is None


def test_small_and_wrong_format_images_rejected(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (319, 500)).save(path)
    with pytest.raises(ValueError, match="too small"):
        prepare_image(path)
    with pytest.raises(ValueError, match="prepared image"):
        create_fragments(Image.new("RGB", (321, 320)), 42)
    with pytest.raises(ValueError, match="prepared image"):
        create_fragments(Image.new("L", (320, 320)), 42)


def test_missing_duplicate_and_bad_pieces_rejected():
    pieces, positions = create_fragments(patterned_image(), 42)
    with pytest.raises(ValueError, match="exactly 25"):
        rebuild_ground_truth(pieces[:-1], positions[:-1])
    with pytest.raises(ValueError, match="exactly once"):
        rebuild_ground_truth(pieces, [positions[0]] * 25)
    with pytest.raises(ValueError, match="integer"):
        rebuild_ground_truth(pieces, [(0.0, 0)] * 25)
    pieces[0] = Image.new("RGB", (63, 64))
    with pytest.raises(ValueError, match="64x64"):
        rebuild_ground_truth(pieces, positions)


def source_fixture(root: Path) -> dict:
    (root / "images").mkdir(parents=True)
    records = []
    for index, split in enumerate(("train", "validation", "test")):
        image_id = f"{index:016x}"
        path = root / "images" / f"{image_id}.jpg"
        patterned_image().save(path)
        records.append({"image_id": image_id, "split": split, "path": f"images/{image_id}.jpg",
                        "sha256": sha256(path), "rotation": "0.0"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (root / "splits.json").write_text(json.dumps({r["split"]: [r["image_id"]] for r in records}), encoding="utf-8")
    return records[0]


def test_saved_dataset_reproducible_and_preserves_splits(tmp_path):
    source, output = tmp_path / "source", tmp_path / "puzzles"
    record = source_fixture(source)
    summary = generate_dataset(source, output)
    assert summary["puzzles"] == summary["round_trips_verified"] == 3
    assert summary["pieces"] == 75
    assert summary["splits"] == {"train": 1, "validation": 1, "test": 1}
    assert (output / "previews" / "train.png").exists()
    before = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    generate_dataset(source, output)
    assert before == {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
    second = tmp_path / "independent"
    generate_dataset(source, second)
    assert before == {str(p.relative_to(second)): sha256(p) for p in second.rglob("*") if p.is_file()}
    metadata = verify_puzzle(output / "train" / record["image_id"])
    assert metadata["seed"] == puzzle_seed(42, record["image_id"])
    assert puzzle_seed(42, record["image_id"]) != puzzle_seed(42, "another_image")
    with pytest.raises(ValueError, match="different configuration"):
        generate_dataset(source, output, seed=43)


def test_changed_source_or_piece_rejected(tmp_path):
    source = tmp_path / "source"
    record = source_fixture(source)
    directory = tmp_path / "puzzle"
    path = source / record["path"]
    generate_puzzle(path, directory, record, 42)
    piece_path = directory / "pieces" / "piece_000.png"
    piece_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_puzzle(directory)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source image checksum"):
        generate_puzzle(path, tmp_path / "other", record, 42)


def test_source_split_mismatch_and_output_overlap_rejected(tmp_path):
    source = tmp_path / "source"
    source_fixture(source)
    with pytest.raises(ValueError, match="must not overlap"):
        generate_dataset(source, source / "puzzles")
    output = tmp_path / "unrelated"
    output.mkdir()
    (output / "notes.txt").write_text("preserve this", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        generate_dataset(source, output)
    (source / "splits.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="assignments disagree"):
        generate_dataset(source, tmp_path / "puzzles")
