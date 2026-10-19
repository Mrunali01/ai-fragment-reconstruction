"""Prepare and cut fixed 5 x 5 puzzles. No model or solver is used here."""

import math
from pathlib import Path
import random

from PIL import Image, ImageOps

GRID_SIZE = 5
PIECE_SIZE = 64
IMAGE_SIZE = GRID_SIZE * PIECE_SIZE


def parse_rotation(value: str | float | None) -> int | None:
    """None means the source dataset did not recover the photo's orientation."""
    if value is None or value == "":
        return None
    try:
        angle = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid source rotation: {value!r}") from error
    if math.isnan(angle):
        return None
    if angle not in (0, 90, 180, 270):
        raise ValueError(f"Source rotation must be 0, 90, 180, or 270: {value!r}")
    return int(angle)


def prepare_image(path: Path, rotation: int | None = None) -> tuple[Image.Image, dict]:
    """Orient, resize proportionally, and center crop without upscaling."""
    if rotation not in (None, 0, 90, 180, 270):
        raise ValueError("Rotation must be None, 0, 90, 180, or 270")
    with Image.open(path) as source:
        source.load()
        original_size = source.size
        if rotation is None:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            policy = "EXIF only; source rotation unknown"
        else:
            # The CVDF metadata angle applies to stored pixels, not an EXIF-transposed image.
            oriented = source.convert("RGB").rotate(rotation, expand=True)
            policy = "source metadata counterclockwise; EXIF not applied"
        width, height = oriented.size
        if min(width, height) < IMAGE_SIZE:
            raise ValueError(f"Image is too small: {width}x{height}; minimum side {IMAGE_SIZE}")
        if width <= height:
            resized_size = (IMAGE_SIZE, round(height * IMAGE_SIZE / width))
        else:
            resized_size = (round(width * IMAGE_SIZE / height), IMAGE_SIZE)
        resized = oriented.resize(resized_size, Image.Resampling.LANCZOS)
        left, top = (resized.width - IMAGE_SIZE) // 2, (resized.height - IMAGE_SIZE) // 2
        box = (left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)
        prepared = resized.crop(box)
    return prepared, {
        "original_size": list(original_size), "oriented_size": [width, height],
        "rotation_ccw": rotation, "orientation_policy": policy,
        "resized_size": list(resized_size), "crop_box": list(box),
        "prepared_size": [IMAGE_SIZE, IMAGE_SIZE], "resampling": "LANCZOS",
    }


def create_fragments(image: Image.Image, seed: int) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    """Return shuffled images and the original (row, column) of each piece."""
    if image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
        raise ValueError(f"Expected an RGB {IMAGE_SIZE}x{IMAGE_SIZE} prepared image")
    positions = [(row, column) for row in range(GRID_SIZE) for column in range(GRID_SIZE)]
    random.Random(seed).shuffle(positions)
    pieces = []
    for row, column in positions:
        left, top = column * PIECE_SIZE, row * PIECE_SIZE
        pieces.append(image.crop((left, top, left + PIECE_SIZE, top + PIECE_SIZE)))
    return pieces, positions


def rebuild_ground_truth(pieces: list[Image.Image], positions: list[tuple[int, int]]) -> Image.Image:
    """Reassemble using known labels, solely to verify dataset correctness."""
    expected = {(row, column) for row in range(GRID_SIZE) for column in range(GRID_SIZE)}
    if len(pieces) != GRID_SIZE ** 2 or len(positions) != len(pieces):
        raise ValueError("Expected exactly 25 pieces and 25 positions")
    if any(len(position) != 2 or any(type(v) is not int for v in position) for position in positions):
        raise ValueError("Positions must contain integer row and column coordinates")
    if set(positions) != expected:
        raise ValueError("Positions must cover the 5x5 grid exactly once")
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    for piece, (row, column) in zip(pieces, positions):
        if piece.size != (PIECE_SIZE, PIECE_SIZE) or piece.mode != "RGB":
            raise ValueError("Each piece must be an RGB 64x64 image")
        canvas.paste(piece, (column * PIECE_SIZE, row * PIECE_SIZE))
    return canvas
