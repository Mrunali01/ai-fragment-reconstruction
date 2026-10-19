"""Independently check exported labels against saved original coordinates."""

import argparse
from collections import Counter
import json
from pathlib import Path

from jigsaw_ai.acquire_images import write_json
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.pairs import DIRECTIONS, SPLITS, read_index


def audit_pairs(root: Path, output: Path) -> dict:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config["source_index_sha256"] != sha256(root / "index.json"):
        raise ValueError("Source puzzle index changed")
    metadata = {}
    for item in read_index(root):
        path = root / item["metadata_path"]
        if config["puzzle_metadata_sha256"][item["metadata_path"]] != sha256(path):
            raise ValueError("Source puzzle metadata changed")
        puzzle = json.loads(path.read_text(encoding="utf-8"))
        metadata[item["image_id"]] = (item["split"], {p["piece_id"]: p for p in puzzle["pieces"]})
    summary = {}
    for split in SPLITS:
        seen, counts, directions = set(), Counter(), Counter()
        with (output / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                pair = json.loads(line)
                puzzle_id = pair["puzzle_id"]
                source_split, pieces = metadata[puzzle_id]
                if pair["split"] != split or source_split != split:
                    raise ValueError("Cross-split pair detected")
                key = (puzzle_id, pair["first_id"], pair["second_id"])
                if key in seen or pair["first_id"] == pair["second_id"]:
                    raise ValueError("Duplicate or self pair detected")
                seen.add(key)
                a, b = pieces[pair["first_id"]], pieces[pair["second_id"]]
                dr = b["original_row"] - a["original_row"]
                dc = b["original_column"] - a["original_column"]
                expected = int(abs(dr) + abs(dc) == 1)
                if type(pair["compatible"]) is not int or pair["compatible"] != expected:
                    raise ValueError("Incorrect compatibility label")
                if expected:
                    if pair["direction"] not in DIRECTIONS or DIRECTIONS[pair["direction"]] != (dr, dc):
                        raise ValueError("Incorrect relative direction")
                    directions[puzzle_id, pair["direction"]] += 1
                elif pair["direction"] is not None:
                    raise ValueError("Negative pair must not have a direction")
                for name, piece in (("first", a), ("second", b)):
                    if (pair[f"{name}_path"] != f"{split}/{puzzle_id}/{piece['path']}"
                            or pair[f"{name}_sha256"] != piece["sha256"]):
                        raise ValueError("Pair image reference mismatch")
                counts[puzzle_id, expected] += 1
        ids = [pid for pid, (source_split, _) in metadata.items() if source_split == split]
        for pid in ids:
            if counts[pid, 0] != 80 or counts[pid, 1] != 80:
                raise ValueError("Expected 80 positives and 80 negatives per puzzle")
            if any(directions[pid, d] != 20 for d in DIRECTIONS):
                raise ValueError("Expected 20 positives per direction per puzzle")
        summary[split] = {"pairs_checked": len(seen), "puzzles": len(ids)}
    result = {"splits": summary, "total_pairs_checked": sum(s["pairs_checked"] for s in summary.values())}
    write_json(output / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs"))
    args = parser.parse_args()
    try:
        print(json.dumps(audit_pairs(args.source, args.pairs), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Pair audit failed: {error}\n")


if __name__ == "__main__":
    main()
