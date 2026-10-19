"""Generate and verify labeled puzzles from a Phase 1 image manifest."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re

from PIL import Image, ImageDraw, __version__ as pillow_version

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.fragments import (
    GRID_SIZE, IMAGE_SIZE, PIECE_SIZE, create_fragments, parse_rotation,
    prepare_image, rebuild_ground_truth,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def puzzle_seed(seed: int, image_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8], "big")


def verify_puzzle(directory: Path) -> dict:
    """Read saved PNGs and prove ground truth is a permutation with exact pixels."""
    metadata = json.loads((directory / "puzzle.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1 or metadata.get("grid") != [GRID_SIZE, GRID_SIZE]
            or metadata.get("piece_size") != [PIECE_SIZE, PIECE_SIZE]):
        raise ValueError(f"Unsupported puzzle format: {directory}")
    entries = metadata["pieces"]
    if [entry["piece_id"] for entry in entries] != [f"piece_{i:03}" for i in range(25)]:
        raise ValueError("Piece IDs must occur exactly once in shuffled order")
    pieces, positions = [], []
    for index, entry in enumerate(entries):
        if entry["shuffled_index"] != index:
            raise ValueError("Shuffled indices must match piece order")
        expected_path = f"pieces/{entry['piece_id']}.png"
        if entry["path"] != expected_path:
            raise ValueError("Unexpected piece path")
        path = directory / expected_path
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Piece checksum mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            pieces.append(image.copy())
        positions.append((entry["original_row"], entry["original_column"]))
    rebuilt = rebuild_ground_truth(pieces, positions)
    prepared_path = directory / "prepared.png"
    if sha256(prepared_path) != metadata["prepared_sha256"]:
        raise ValueError("Prepared image checksum mismatch")
    with Image.open(prepared_path) as image:
        image.load()
        if image.mode != "RGB" or image.size != rebuilt.size or image.tobytes() != rebuilt.tobytes():
            raise ValueError(f"Ground-truth round trip does not match: {directory}")
    return metadata


def generate_puzzle(source: Path, directory: Path, record: dict, seed: int) -> dict:
    if sha256(source) != record["sha256"]:
        raise ValueError(f"Source image checksum mismatch: {source}")
    derived_seed = puzzle_seed(seed, record["image_id"])
    if directory.exists():
        metadata = verify_puzzle(directory)
        if (metadata["seed"] != derived_seed or metadata["source_sha256"] != record["sha256"]
                or metadata["source_rotation"] != record.get("rotation", "")
                or metadata["split"] != record["split"] or metadata["image_id"] != record["image_id"]):
            raise ValueError(f"Existing puzzle has different inputs: {directory}")
        return metadata
    prepared, preprocessing = prepare_image(source, parse_rotation(record.get("rotation")))
    pieces, positions = create_fragments(prepared, derived_seed)
    directory.mkdir(parents=True)
    (directory / "pieces").mkdir()
    prepared.save(directory / "prepared.png")
    entries = []
    for index, (piece, (row, column)) in enumerate(zip(pieces, positions)):
        piece_id = f"piece_{index:03}"
        relative = f"pieces/{piece_id}.png"
        piece.save(directory / relative)
        entries.append({"piece_id": piece_id, "path": relative, "shuffled_index": index,
                        "original_row": row, "original_column": column,
                        "sha256": sha256(directory / relative)})
    metadata = {
        "schema_version": 1, "image_id": record["image_id"], "split": record["split"],
        "seed": derived_seed, "source_sha256": record["sha256"],
        "source_rotation": record.get("rotation", ""),
        "source_path": record["path"], "source_url": record.get("source_url"),
        "author": record.get("author"), "title": record.get("title"), "license": record.get("license"),
        "modifications": "Orientation handling, proportional resize, center crop, and 5x5 slicing",
        "grid": [GRID_SIZE, GRID_SIZE], "piece_size": [PIECE_SIZE, PIECE_SIZE],
        "preprocessing": preprocessing, "prepared_sha256": sha256(directory / "prepared.png"),
        "pieces": entries,
    }
    write_json(directory / "puzzle.json", metadata)
    return verify_puzzle(directory)


def save_preview(directory: Path, destination: Path) -> None:
    metadata = verify_puzzle(directory)
    pieces, positions = [], []
    for entry in metadata["pieces"]:
        with Image.open(directory / entry["path"]) as piece:
            pieces.append(piece.copy())
        positions.append((entry["original_row"], entry["original_column"]))
    rebuilt = rebuild_ground_truth(pieces, positions)
    shuffled = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    for i, piece in enumerate(pieces):
        shuffled.paste(piece, ((i % GRID_SIZE) * PIECE_SIZE, (i // GRID_SIZE) * PIECE_SIZE))
    canvas = Image.new("RGB", (1000, 370), "white")
    draw = ImageDraw.Draw(canvas)
    with Image.open(directory / "prepared.png") as prepared:
        for index, (title, image) in enumerate([
            ("Prepared target", prepared), ("Shuffled pieces", shuffled), ("Rebuilt using ground truth", rebuilt)
        ]):
            x = 8 + index * 332
            draw.text((x, 8), title, fill="black")
            canvas.paste(image, (x, 30))
    draw.text((8, 354), f"{metadata['split']}/{metadata['image_id']} | Label-based check, not an AI prediction", fill="black")
    canvas.save(destination)


def load_records(root: Path) -> list[dict]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("Source manifest is empty")
    ids = [row["image_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs in source manifest")
    for row in records:
        if not re.fullmatch(r"[0-9a-f]{16}", row["image_id"]):
            raise ValueError("Invalid source image ID")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid source split")
        if row["path"] != f"images/{row['image_id']}.jpg":
            raise ValueError("Unexpected source path")
        parse_rotation(row.get("rotation"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if splits != {split: [row["image_id"] for row in records if row["split"] == split]
                  for split in ("train", "validation", "test")}:
        raise ValueError("Source manifest and split assignments disagree")
    return records


def generate_dataset(root: Path, output: Path, seed: int = 42) -> dict:
    records = load_records(root)
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Puzzle output and source dataset directories must not overlap")
    config = {"schema_version": 1, "seed": seed, "grid": [GRID_SIZE, GRID_SIZE],
              "piece_size": [PIECE_SIZE, PIECE_SIZE], "pillow_version": pillow_version,
              "python_version": platform.python_version(),
              "source_manifest_sha256": sha256(root / "manifest.jsonl")}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing puzzles use different configuration; choose a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    index = []
    for number, record in enumerate(records, start=1):
        relative = f"{record['split']}/{record['image_id']}"
        generate_puzzle(root / record["path"], output / relative, record, seed)
        index.append({"image_id": record["image_id"], "split": record["split"],
                      "metadata_path": f"{relative}/puzzle.json"})
        if number % 50 == 0 or number == len(records):
            print(f"Generated and verified {number}/{len(records)} puzzles", flush=True)
    write_json(output / "index.json", index)
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = next((r for r in records if r["split"] == split), None)
        if selected:
            save_preview(output / split / selected["image_id"], previews / f"{split}.png")
    rotated = next((r for r in records if parse_rotation(r.get("rotation")) not in (None, 0)), None)
    if rotated:
        save_preview(output / rotated["split"] / rotated["image_id"], previews / "rotation_example.png")
    summary = {"puzzles": len(index), "pieces": len(index) * GRID_SIZE ** 2,
               "splits": dict(Counter(row["split"] for row in records)),
               "round_trips_verified": len(index), "image_size": [IMAGE_SIZE, IMAGE_SIZE],
               "piece_size": [PIECE_SIZE, PIECE_SIZE],
               "source_rotations_applied": sum(parse_rotation(r.get("rotation")) not in (None, 0) for r in records),
               "unknown_source_orientation": sum(parse_rotation(r.get("rotation")) is None for r in records)}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/open_images"))
    parser.add_argument("--output", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(generate_dataset(args.source, args.output, args.seed), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Puzzle generation failed: {error}\n")


if __name__ == "__main__":
    main()
