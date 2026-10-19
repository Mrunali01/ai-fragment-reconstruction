"""Audit a local dataset and create intact-image contact sheets."""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from jigsaw_ai.acquire_images import LICENSE, TARGETS, license_evidence, write_json
from jigsaw_ai.image_validation import inspect_image


def audit(root: Path) -> dict:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    counts = dict(Counter(row["split"] for row in records))
    if counts != TARGETS:
        raise ValueError(f"Split counts do not match targets: {counts}")
    for field in ("image_id", "sha256", "pixel_sha256", "source_url"):
        values = [r[field].rstrip("/") for r in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicates in {field}")
    for split in TARGETS:
        if splits[split] != [r["image_id"] for r in records if r["split"] == split]:
            raise ValueError(f"Split assignment mismatch: {split}")
    for row in records:
        if row["path"] != f"images/{row['image_id']}.jpg":
            raise ValueError("Unexpected image path")
        details = inspect_image(root / row["path"], config["minimum_side"])
        if any(row[key] != value for key, value in details.items()):
            raise ValueError(f"Image details changed: {row['image_id']}")
        evidence = json.loads((root / row["license_evidence"]).read_text(encoding="utf-8"))
        image = evidence["image"]
        # Reuse the parser to verify the stored image-specific structured evidence.
        page = '<script type="application/ld+json">' + json.dumps(image) + '</script>'
        license_evidence(page, row["source_url"])
        if row["license"] != LICENSE or row["author"] != image["author"]["name"]:
            raise ValueError("Attribution or license mismatch")
    summary = {"images": len(records), "splits": counts,
               "image_bytes": sum(r["bytes"] for r in records),
               "width_range": [min(r["width"] for r in records), max(r["width"] for r in records)],
               "height_range": [min(r["height"] for r in records), max(r["height"] for r in records)],
               "exact_duplicates": 0,
               "directory_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}
    write_json(root / "audit.json", summary)
    return summary


def contact_sheets(root: Path) -> None:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    preview_dir = root / "previews"
    preview_dir.mkdir(exist_ok=True)
    for split in TARGETS:
        selected = [row for row in records if row["split"] == split]
        for start in range(0, len(selected), 50):
            batch = selected[start:start + 50]
            canvas = Image.new("RGB", (1000, ((len(batch) + 4) // 5) * 145), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(batch):
                x, y = (index % 5) * 200, (index // 5) * 145
                with Image.open(root / row["path"]) as image:
                    thumbnail = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (196, 120))
                    canvas.paste(thumbnail, (x + (200 - thumbnail.width) // 2, y))
                draw.text((x + 2, y + 123), row["image_id"], fill="black")
            canvas.save(preview_dir / f"{split}_{start // 50 + 1:02}.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/open_images"))
    parser.add_argument("--previews", action="store_true")
    args = parser.parse_args()
    try:
        if args.previews:
            contact_sheets(args.root)
        print(json.dumps(audit(args.root), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Audit failed: {error}\n")


if __name__ == "__main__":
    main()
