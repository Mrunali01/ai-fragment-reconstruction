"""Validate intact images without resizing or generating puzzle pieces."""

import hashlib
from pathlib import Path

from PIL import Image, ImageOps


def inspect_image(path: Path, minimum_side: int = 320) -> dict:
    """Fully decode an image and fingerprint its orientation-corrected RGB pixels."""
    if minimum_side < 1:
        raise ValueError("minimum_side must be positive")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb.size
        if min(width, height) < minimum_side:
            raise ValueError(f"Image is too small: {width}x{height}; minimum side {minimum_side}")
        pixels = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
    return {
        "width": width, "height": height,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pixel_sha256": pixels,
    }
