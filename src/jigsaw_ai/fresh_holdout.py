"""Acquire a precommitted, licensed, development-disjoint confirmation cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from PIL import Image
import numpy as np

from jigsaw_ai.acquire_images import acquire_one, candidate_order, save_manifest, write_json
from jigsaw_ai.fragments import parse_rotation, prepare_image
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256


def difference_hash(image: Image.Image) -> int:
    image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(image).reshape(-1)
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def acquire(originals: Path, development: Path, frozen: Path, output: Path,
            puzzle_output: Path, count: int = 50) -> dict:
    if count < 1:
        raise ValueError("Holdout count must be positive")
    paths = [originals.resolve(), development.resolve(), frozen.resolve(), output.resolve(), puzzle_output.resolve()]
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("Fresh data and existing inputs must not overlap")
    selection = json.loads((frozen / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads((frozen / "config.json").read_text(encoding="utf-8"))
    if selection["split"] != "validation":
        raise ValueError("Freeze validation selection first")
    old = [json.loads(line) for line in (originals / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    old_ids = {r["image_id"] for r in old}
    index = json.loads((development / "index.json").read_text(encoding="utf-8"))
    if old_ids != set(configuration["development_ids"]) or old_ids != {r["image_id"] for r in index}:
        raise ValueError("Development manifests and frozen selection disagree")
    snapshot = originals / "metadata/test.json"
    metadata = []
    for split in ("train", "validation", "test"):
        metadata.extend(json.loads((originals / "metadata" / f"{split}.json").read_text(encoding="utf-8"))["rows"])
    old_authors = {r["AuthorProfileURL"].rstrip("/") for r in metadata if r["ImageID"] in old_ids and r.get("AuthorProfileURL")}
    pool = json.loads(snapshot.read_text(encoding="utf-8"))["rows"]
    available = [r for r in candidate_order(pool, 20261019) if r["ImageID"] not in old_ids
                 and r.get("AuthorProfileURL", "").rstrip("/") not in old_authors]
    protocol = {"count": count, "selection_seed": 20261019, "source": "unused official Open Images test CSV prefix candidates",
                "metadata_snapshot_sha256": sha256(snapshot), "old_manifest_sha256": sha256(originals / "manifest.jsonl"),
                "validation_selection_sha256": sha256(frozen / "summary.json"),
                "checkpoint_sha256": configuration["checkpoint_sha256"], "selected_search": selection["selected_mode"],
                "excluded_development_ids": sorted(old_ids), "excluded_author_profiles": sorted(old_authors),
                "duplicate_rule": "exclude same ID, source URL, file/pixel SHA256, or prepared-image dHash Hamming distance <=4",
                "near_duplicate_threshold": 4, "model_scores_used_for_acquisition": False,
                "candidate_ids": [r["ImageID"] for r in available]}
    for root in (output, puzzle_output):
        if root.exists() and any(root.iterdir()):
            raise ValueError("Fresh acquisition directories must be new/empty")
        root.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", protocol)
    for name in ("images", "licenses", "attempts"):
        (output / name).mkdir()
    sources = {r["source_url"].rstrip("/") for r in old}
    files, pixels = {r["sha256"] for r in old}, {r["pixel_sha256"] for r in old}
    hashes = []
    for item in index:
        with Image.open((development / item["metadata_path"]).parent / "prepared.png") as image:
            hashes.append(difference_hash(image))
    accepted, rejected = [], []
    for start in range(0, len(available), 4):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda row: acquire_one(output, row, 320), available[start:start + 4]))
        for record in results:
            if len(accepted) >= count:
                break
            reason = record.get("error")
            image_hash = None
            if reason is None:
                if record["source_url"].rstrip("/") in sources or record["sha256"] in files or record["pixel_sha256"] in pixels:
                    reason = "duplicate source/file/pixels"
                else:
                    prepared, _ = prepare_image(output / record["path"], parse_rotation(record.get("rotation")))
                    image_hash = difference_hash(prepared)
                    if any((image_hash ^ previous).bit_count() <= 4 for previous in hashes):
                        reason = "near-duplicate prepared-image fingerprint"
            if reason:
                rejected.append({"image_id": record["image_id"], "reason": reason})
            else:
                accepted.append(record)
                sources.add(record["source_url"].rstrip("/")); files.add(record["sha256"]); pixels.add(record["pixel_sha256"])
                hashes.append(image_hash)
        save_manifest(output, accepted)
        write_json(output / "rejections.json", rejected)
        print(f"Fresh holdout: {len(accepted)}/{count} accepted, {len(rejected)} rejected", flush=True)
        if len(accepted) >= count:
            break
    if len(accepted) != count:
        raise ValueError(f"Only {len(accepted)} licensed, disjoint images available; frozen target was {count}")
    puzzle_index = []
    for record in accepted:
        path = f"test/{record['image_id']}/puzzle.json"
        generate_puzzle(output / record["path"], (puzzle_output / path).parent, record, 42)
        puzzle_index.append({"image_id": record["image_id"], "split": "test", "metadata_path": path})
    write_json(puzzle_output / "index.json", puzzle_index)
    summary = {"images": len(accepted), "rejected": len(rejected), "source_bytes": sum(r["bytes"] for r in accepted),
               "development_id_overlap": 0, "development_photographer_overlap": 0,
               "filtered_near_duplicates": True, "near_duplicate_detection_is_not_exhaustive": True,
               "protocol_sha256": sha256(output / "protocol.json"), "puzzle_index_sha256": sha256(puzzle_output / "index.json")}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, default=Path("data/open_images"))
    parser.add_argument("--development", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--frozen-validation", type=Path, default=Path("artifacts/search_experiments/validation_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/fresh_holdout_v1"))
    parser.add_argument("--puzzle-output", type=Path, default=Path("data/fresh_puzzles_v1"))
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    try:
        print(json.dumps(acquire(args.originals, args.development, args.frozen_validation, args.output, args.puzzle_output, args.count), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Fresh acquisition failed: {error}\n")
