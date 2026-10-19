"""Balanced, ordered fragment pairs with pixels loaded only on demand."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re

from PIL import Image

from jigsaw_ai.generate_puzzles import puzzle_seed, sha256, verify_puzzle

# Direction always describes the SECOND piece relative to the FIRST piece.
DIRECTIONS = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
DIRECTION_IDS = {name: index for index, name in enumerate(DIRECTIONS)}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PairRecord:
    puzzle_id: str
    split: str
    first_id: str
    second_id: str
    first_path: str
    second_path: str
    first_sha256: str
    second_sha256: str
    compatible: int
    direction: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def pairs_for_puzzle(metadata: dict, seed: int = 42) -> list[PairRecord]:
    """Keep all 80 directed neighbors and sample 80 unique ordered non-neighbors."""
    entries = metadata["pieces"]
    coordinates = [(p["original_row"], p["original_column"]) for p in entries]
    if (metadata["grid"] != [5, 5] or len(entries) != 25
            or any(type(v) is not int for position in coordinates for v in position)
            or set(coordinates) != {(r, c) for r in range(5) for c in range(5)}):
        raise ValueError("Expected a complete 5x5 ground-truth grid")
    if len({p["piece_id"] for p in entries}) != 25:
        raise ValueError("Piece IDs must be unique")
    if metadata["split"] not in SPLITS:
        raise ValueError("Unknown dataset split")
    lookup = dict(zip(coordinates, entries))
    prefix = f"{metadata['split']}/{metadata['image_id']}"

    def record(first: dict, second: dict, direction: str | None) -> PairRecord:
        return PairRecord(
            metadata["image_id"], metadata["split"], first["piece_id"], second["piece_id"],
            f"{prefix}/{first['path']}", f"{prefix}/{second['path']}",
            first["sha256"], second["sha256"], int(direction is not None), direction,
        )

    positives, negatives = [], []
    for first in entries:
        row, column = first["original_row"], first["original_column"]
        for direction, (dr, dc) in DIRECTIONS.items():
            second = lookup.get((row + dr, column + dc))
            if second is not None:
                positives.append(record(first, second, direction))
        for second in entries:
            distance = abs(row - second["original_row"]) + abs(column - second["original_column"])
            if distance > 1:
                negatives.append(record(first, second, None))
    rng = random.Random(puzzle_seed(seed, f"pairs:{metadata['image_id']}"))
    pairs = positives + rng.sample(negatives, len(positives))
    rng.shuffle(pairs)
    return pairs


def read_index(root: Path) -> list[dict]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if not index:
        raise ValueError("Puzzle index is empty")
    seen = set()
    for item in index:
        image_id, split = item["image_id"], item["split"]
        if not re.fullmatch(r"[0-9a-f]{16}", image_id) or split not in SPLITS:
            raise ValueError("Invalid puzzle ID or split")
        if image_id in seen:
            raise ValueError("Puzzle IDs must be unique across all splits")
        seen.add(image_id)
        if item["metadata_path"] != f"{split}/{image_id}/puzzle.json":
            raise ValueError("Unexpected puzzle metadata path")
    return index


class PairDataset:
    """A small Python sequence; PyTorch tensor conversion belongs to later phases."""

    def __init__(self, root: Path, split: str, seed: int = 42) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(root)
        self.records: list[PairRecord] = []
        for item in read_index(self.root):
            if item["split"] != split:
                continue
            directory = (self.root / item["metadata_path"]).parent
            metadata = verify_puzzle(directory)
            if metadata["image_id"] != item["image_id"] or metadata["split"] != split:
                raise ValueError("Puzzle metadata disagrees with index")
            self.records.extend(pairs_for_puzzle(metadata, seed))
        if not self.records:
            raise ValueError(f"No puzzles available for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        images = []
        for relative, expected_hash in ((record.first_path, record.first_sha256),
                                        (record.second_path, record.second_sha256)):
            path = self.root / relative
            if sha256(path) != expected_hash:
                raise ValueError(f"Pair image checksum mismatch: {path}")
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (64, 64):
                    raise ValueError("Pair images must be RGB 64x64 fragments")
                images.append(image.copy())
        # Positions, shuffle seeds, and the prepared target are never model inputs.
        return {"first": images[0], "second": images[1], "compatible": record.compatible,
                "direction": None if record.direction is None else DIRECTION_IDS[record.direction]}
