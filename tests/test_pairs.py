"""Check pair labels independently against grid coordinates and saved pixels."""

from collections import Counter
import json
import random

from PIL import Image
import pytest

from jigsaw_ai.generate_pairs import generate_pairs
from jigsaw_ai.audit_pairs import audit_pairs
from jigsaw_ai.generate_puzzles import generate_puzzle, sha256
from jigsaw_ai.pairs import DIRECTIONS, DIRECTION_IDS, PairDataset, pairs_for_puzzle


@pytest.fixture
def puzzle_root(tmp_path):
    root = tmp_path / "puzzles"
    index = []
    for i, split in enumerate(("train", "validation", "test")):
        image_id = f"{i:016x}"
        path = tmp_path / f"source_{i}.jpg"
        image = Image.new("RGB", (320, 320))
        image.putdata([((x + i * 50) % 256, y % 256, (x + y) % 256)
                       for y in range(320) for x in range(320)])
        image.save(path)
        record = {"image_id": image_id, "split": split, "sha256": sha256(path),
                  "path": f"images/{image_id}.jpg", "rotation": "0.0"}
        relative = f"{split}/{image_id}"
        generate_puzzle(path, root / relative, record, 42)
        index.append({"image_id": image_id, "split": split, "metadata_path": f"{relative}/puzzle.json"})
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_pair_labels_counts_and_reverse_directions(puzzle_root):
    metadata = json.loads(next((puzzle_root / "train").glob("*/puzzle.json")).read_text())
    state = random.getstate()
    pairs = pairs_for_puzzle(metadata)
    assert random.getstate() == state
    coordinates = {r["piece_id"]: (r["original_row"], r["original_column"]) for r in metadata["pieces"]}
    assert len(pairs) == 160
    assert Counter(r.compatible for r in pairs) == {0: 80, 1: 80}
    assert Counter(r.direction for r in pairs if r.compatible) == {d: 20 for d in DIRECTIONS}
    assert len({(r.first_id, r.second_id) for r in pairs}) == 160
    positives = {(r.first_id, r.second_id): r.direction for r in pairs if r.compatible}
    opposite = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
    for pair in pairs:
        assert pair.first_id != pair.second_id
        a, b = coordinates[pair.first_id], coordinates[pair.second_id]
        dr, dc = b[0] - a[0], b[1] - a[1]
        assert pair.compatible == int(abs(dr) + abs(dc) == 1)
        if pair.compatible:
            assert DIRECTIONS[pair.direction] == (dr, dc)
            assert positives[pair.second_id, pair.first_id] == opposite[pair.direction]
        else:
            assert pair.direction is None
    assert pairs == pairs_for_puzzle(metadata)
    changed = pairs_for_puzzle(metadata, seed=43)
    assert {(r.first_id, r.second_id) for r in changed if r.compatible} == set(positives)
    assert {(r.first_id, r.second_id) for r in changed if not r.compatible} != {
        (r.first_id, r.second_id) for r in pairs if not r.compatible}


def test_loader_pixels_labels_split_isolation_and_file_changes(puzzle_root):
    datasets = {split: PairDataset(puzzle_root, split) for split in ("train", "validation", "test")}
    ids = [{r.puzzle_id for r in dataset.records} for dataset in datasets.values()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    for split, dataset in datasets.items():
        assert len(dataset) == 160
        for label in (0, 1):
            i = next(i for i, r in enumerate(dataset.records) if r.compatible == label)
            record, sample = dataset.records[i], dataset[i]
            assert record.split == split
            assert set(sample) == {"first", "second", "compatible", "direction"}
            with Image.open(puzzle_root / record.first_path) as expected:
                assert sample["first"].tobytes() == expected.tobytes()
            assert sample["second"].size == (64, 64)
            assert sample["compatible"] == label
            assert sample["direction"] == (DIRECTION_IDS[record.direction] if label else None)
        with pytest.raises(IndexError):
            dataset[len(dataset)]
    dataset = datasets["train"]
    (puzzle_root / dataset.records[0].first_path).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset[0]


def test_export_reproducible_and_settings_protected(puzzle_root, tmp_path):
    output = tmp_path / "pairs"
    result = generate_pairs(puzzle_root, output)
    assert result["total_pairs"] == 480
    assert audit_pairs(puzzle_root, output)["total_pairs_checked"] == 480
    before = {p.name: sha256(p) for p in output.iterdir()}
    generate_pairs(puzzle_root, output)
    assert before == {p.name: sha256(p) for p in output.iterdir()}
    second = tmp_path / "independent"
    generate_pairs(puzzle_root, second)
    audit_pairs(puzzle_root, second)
    assert before == {p.name: sha256(p) for p in second.iterdir()}
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 160
    assert all(row["split"] == "train" for row in rows)
    with pytest.raises(ValueError, match="different settings"):
        generate_pairs(puzzle_root, output, seed=43)
    with pytest.raises(ValueError, match="must not overlap"):
        generate_pairs(puzzle_root, puzzle_root / "pairs")
    rows[0]["compatible"] = 1 - rows[0]["compatible"]
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Incorrect compatibility"):
        audit_pairs(puzzle_root, output)


def test_malformed_grid_and_index_rejected(puzzle_root):
    path = next((puzzle_root / "train").glob("*/puzzle.json"))
    metadata = json.loads(path.read_text())
    metadata["pieces"][0]["original_row"] = 99
    with pytest.raises(ValueError, match="complete 5x5"):
        pairs_for_puzzle(metadata)
    with pytest.raises(ValueError, match="Unknown split"):
        PairDataset(puzzle_root, "training")
    index_path = puzzle_root / "index.json"
    index = json.loads(index_path.read_text())
    index[1]["image_id"] = index[0]["image_id"]
    with index_path.open("w") as stream:
        json.dump(index, stream)
    with pytest.raises(ValueError, match="unique across all splits"):
        PairDataset(puzzle_root, "train")
