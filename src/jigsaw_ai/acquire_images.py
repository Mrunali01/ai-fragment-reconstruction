"""Acquire a licensed, reproducible Open Images subset. Run with --help."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import UnidentifiedImageError

from jigsaw_ai.image_validation import inspect_image

SOURCES = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}
TARGETS = {"train": 350, "validation": 75, "test": 75}
LICENSE = "https://creativecommons.org/licenses/by/2.0/"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_file(temporary: Path, destination: Path) -> None:
    """Allow brief Windows scanner/indexer locks without discarding old output."""
    for attempt in range(5):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    replace_file(temporary, path)


def fetch(url: str, limit: int = 20_000_000) -> bytes:
    """Fetch with bounded memory, timeout, and retries for transient errors."""
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "JigsawAI/0.1 (educational dataset acquisition)"})
            with urlopen(request, timeout=20) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"Response exceeds {limit} bytes")
            return data
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)
    raise RuntimeError("Download retries exhausted")


class StructuredDataParser(HTMLParser):
    """Extract JSON-LD scripts rather than unrelated license links in a page."""

    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.parts: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.active = dict(attrs).get("type") == "application/ld+json"
            self.parts = []

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            self.scripts.append("".join(self.parts))
            self.active = False


def image_objects(value):
    if isinstance(value, dict):
        if value.get("@type") == "ImageObject":
            yield value
        for child in value.values():
            yield from image_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from image_objects(child)


def license_evidence(page: str, source_url: str) -> dict:
    parser = StructuredDataParser()
    parser.feed(page)
    for script in parser.scripts:
        try:
            document = json.loads(script)
        except json.JSONDecodeError:
            continue
        for image in image_objects(document):
            if image.get("acquireLicensePage", "").rstrip("/") != source_url.rstrip("/"):
                continue
            if image.get("license") != LICENSE:
                raise ValueError("Original photo page no longer lists CC BY 2.0")
            author = image.get("author", {})
            if not isinstance(author, dict) or not author.get("name"):
                raise ValueError("Photo page has no usable author attribution")
            return image
    raise ValueError("No matching image-specific structured license evidence")


def candidate_order(rows: list[dict], seed: int) -> list[dict]:
    """Stable order independent of thread completion or CSV row order."""
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row['Subset']}:{row['ImageID']}".encode()).hexdigest())


def snapshot_metadata(root: Path, split: str, pool_size: int) -> list[dict]:
    path = root / "metadata" / f"{split}.json"
    if path.exists():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot["pool_size"] != pool_size:
            raise ValueError("Existing metadata uses a different pool size; use a new output directory")
        return snapshot["rows"]
    # A bounded prefix avoids downloading the enormous full training CSV.
    # It is deliberately documented as a pilot candidate pool, not a uniform sample.
    for attempt in range(3):
        try:
            with urlopen(Request(SOURCES[split], headers={"User-Agent": "JigsawAI/0.1"}), timeout=30) as response:
                reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
                rows = []
                for row in reader:
                    rows.append(row)
                    if len(rows) == pool_size:
                        break
            break
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    write_json(path, {"url": SOURCES[split], "retrieved_at": now(), "pool_size": pool_size, "rows": rows})
    return rows


def acquire_one(root: Path, row: dict, minimum_side: int) -> dict:
    image_id, split = row["ImageID"], row["Subset"]
    try:
        if not re.fullmatch(r"[0-9a-f]{16}", image_id) or split not in SOURCES:
            raise ValueError("Invalid image ID or source split")
        if row.get("License") != LICENSE:
            raise ValueError("Metadata license is not CC BY 2.0")
        source_url = row["OriginalLandingURL"]
        if not re.fullmatch(r"https://www\.flickr\.com/photos/[^/]+/\d+/?", source_url):
            raise ValueError("Unsupported original photo page URL")
        evidence_path = root / "licenses" / f"{image_id}.json"
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if evidence["source_url"] != source_url or evidence["image"]["license"] != LICENSE:
                raise ValueError("Cached license evidence does not match")
        else:
            page = fetch(source_url, limit=3_000_000)
            image = license_evidence(page.decode("utf-8"), source_url)
            evidence = {"source_url": source_url, "checked_at": now(),
                        "page_sha256": hashlib.sha256(page).hexdigest(), "image": image}
            write_json(evidence_path, evidence)
        path = root / "images" / f"{image_id}.jpg"
        download_url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
        if not path.exists():
            data = fetch(download_url)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(data)
            try:
                inspect_image(temporary, minimum_side)
                replace_file(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        details = inspect_image(path, minimum_side)
        return {"image_id": image_id, "split": split, "path": f"images/{image_id}.jpg",
                "source_url": source_url, "original_url": row["OriginalURL"],
                "download_url": download_url, "license": LICENSE,
                "author": evidence["image"]["author"]["name"],
                "title": row["Title"], "rotation": row.get("Rotation", ""),
                "license_evidence": f"licenses/{image_id}.json", **details}
    except (OSError, ValueError, UnidentifiedImageError) as error:
        return {"image_id": image_id, "split": split, "error": f"{type(error).__name__}: {error}"}


def save_manifest(root: Path, records: list[dict]) -> None:
    temporary = root / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    replace_file(temporary, root / "manifest.jsonl")
    write_json(root / "splits.json", {split: [r["image_id"] for r in records if r["split"] == split]
                                     for split in SOURCES})


def acquire(root: Path, seed: int = 42, pool_size: int = 4000,
            minimum_side: int = 320, workers: int = 4) -> None:
    if workers < 1 or pool_size < max(TARGETS.values()) or minimum_side < 1:
        raise ValueError("Invalid workers, candidate pool size, or minimum side")
    for directory in (root, root / "images", root / "metadata", root / "licenses", root / "attempts"):
        directory.mkdir(parents=True, exist_ok=True)
    config = {"seed": seed, "pool_size": pool_size, "minimum_side": minimum_side,
              "targets": TARGETS, "selection": "SHA256 order of bounded CSV prefix per source split"}
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory has different acquisition settings; use a new directory")
    write_json(config_path, config)
    records, seen_files, seen_pixels, seen_sources = [], set(), set(), set()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split, target in TARGETS.items():
            rows = candidate_order(snapshot_metadata(root, split, pool_size), seed)
            accepted = 0
            start = 0
            while start < len(rows) and accepted < target:
                batch = rows[start:start + min(workers, target - accepted)]
                def process(row):
                    attempt_path = root / "attempts" / f"{row['ImageID']}.json"
                    if attempt_path.exists():
                        result = json.loads(attempt_path.read_text(encoding="utf-8"))
                        if "error" not in result:
                            details = inspect_image(root / result["path"], minimum_side)
                            if details != {key: result[key] for key in details}:
                                raise ValueError(f"Cached image changed: {result['image_id']}")
                        return result
                    result = acquire_one(root, row, minimum_side)
                    write_json(attempt_path, result)
                    return result
                for result in executor.map(process, batch):
                    if "error" in result:
                        failures.append(result)
                        continue
                    if (result["sha256"] in seen_files or result["pixel_sha256"] in seen_pixels
                            or result["source_url"].rstrip("/") in seen_sources):
                        failures.append({"image_id": result["image_id"], "split": split, "error": "Duplicate image"})
                        continue
                    seen_files.add(result["sha256"])
                    seen_pixels.add(result["pixel_sha256"])
                    seen_sources.add(result["source_url"].rstrip("/"))
                    records.append(result)
                    accepted += 1
                save_manifest(root, records)
                write_json(root / "failures.json", failures)
                print(f"{split}: {accepted}/{target}; candidates examined {start + len(batch)}", flush=True)
                start += len(batch)
                if accepted == target:
                    break
            if accepted != target:
                raise ValueError(f"Candidate pool exhausted for {split}: {accepted}/{target}")
    write_json(root / "acquisition_summary.json", {"finished_at": now(), "images": len(records),
               "image_bytes": sum(r["bytes"] for r in records), "rejected_candidates": len(failures)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/open_images"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--minimum-side", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        acquire(args.output, args.seed, args.pool_size, args.minimum_side, args.workers)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Acquisition failed: {error}\n")


if __name__ == "__main__":
    main()
