"""Export Phase 3 pair manifests, count summaries, and labeled examples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform

from PIL import Image, ImageDraw

from jigsaw_ai.acquire_images import replace_file, write_json
from jigsaw_ai.generate_puzzles import sha256
from jigsaw_ai.pairs import DIRECTIONS, DIRECTION_IDS, PairDataset, SPLITS, read_index


def save_examples(dataset: PairDataset, path: Path) -> None:
    chosen = [next(i for i, r in enumerate(dataset.records) if r.direction == direction)
              for direction in DIRECTIONS]
    chosen += [i for i, r in enumerate(dataset.records) if not r.compatible][:4]
    canvas = Image.new("RGB", (660, 8 * 86 + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 3), "Second relative to first; negatives have no direction", fill="black")
    for row, index in enumerate(chosen):
        sample, record = dataset[index], dataset.records[index]
        y = 24 + row * 86
        canvas.paste(sample["first"], (8, y))
        canvas.paste(sample["second"], (88, y))
        label = f"POSITIVE: {record.direction}" if record.compatible else "NEGATIVE: not neighbors"
        draw.text((170, y + 8), label, fill="black")
        draw.text((170, y + 27), f"{record.first_id} -> {record.second_id}", fill="black")
        draw.text((170, y + 44), f"{record.split}/{record.puzzle_id}", fill="black")
    canvas.save(path)


def generate_pairs(root: Path, output: Path, seed: int = 42) -> dict:
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Pair output and puzzle directories must not overlap")
    index = read_index(root)
    config = {"schema_version": 1, "seed": seed, "negative_to_positive_ratio": 1,
              "python_version": platform.python_version(), "direction_ids": DIRECTION_IDS,
              "source_index_sha256": sha256(root / "index.json"),
              "puzzle_metadata_sha256": {r["metadata_path"]: sha256(root / r["metadata_path"]) for r in index}}
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing pair output uses different settings or source metadata; use a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Pair output directory is not empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
    summaries = {}
    for split in SPLITS:
        dataset = PairDataset(root, split, seed)
        temporary = output / f"{split}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in dataset.records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        replace_file(temporary, output / f"{split}.jsonl")
        positives = sum(r.compatible for r in dataset.records)
        summaries[split] = {"puzzles": len({r.puzzle_id for r in dataset.records}),
                            "pairs": len(dataset), "positive": positives,
                            "negative": len(dataset) - positives,
                            "directions": dict(Counter(r.direction for r in dataset.records if r.compatible))}
        save_examples(dataset, output / f"{split}_examples.png")
        print(f"{split}: {len(dataset)} verified-label pairs", flush=True)
    summary = {"splits": summaries, "total_pairs": sum(s["pairs"] for s in summaries.values())}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/puzzles"))
    parser.add_argument("--output", type=Path, default=Path("data/pairs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(generate_pairs(args.source, args.output, args.seed), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Pair generation failed: {error}\n")


if __name__ == "__main__":
    main()
