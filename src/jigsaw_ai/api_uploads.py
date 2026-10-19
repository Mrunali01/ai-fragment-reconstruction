"""Bound request bodies and decode only supported puzzle fragments."""

from io import BytesIO
import re

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from starlette.responses import JSONResponse

MAX_FILE_BYTES = 256 * 1024
MAX_BODY_BYTES = 8 * 1024 * 1024


class UploadBodyLimit:
    """Bound bytes before multipart parsing, including chunked requests.

    These small 25-piece requests are buffered once to avoid partially parsed
    temporary uploads when rejecting an oversized body.
    """

    def __init__(self, app, limit: int = MAX_BODY_BYTES) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            return
        if declared > self.limit:
            await JSONResponse({"detail": "Request body exceeds 8 MiB"}, status_code=413)(scope, receive, send)
            return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            size += len(data)
            if size > self.limit:
                await JSONResponse({"detail": "Request body exceeds 8 MiB"}, status_code=413)(scope, receive, send)
                return
            chunks.append(data)
            if not message.get("more_body", False):
                break
        body, delivered = b"".join(chunks), False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def display_filename(filename: str | None, index: int) -> str:
    # Names are labels only. Disk paths always use server-generated UUIDs.
    name = re.split(r"[\\/]", filename or "")[-1]
    name = "".join(character for character in name if character.isprintable())[:255]
    return name or f"piece_{index:03}.png"


def decode_fragment(data: bytes, index: int) -> Image.Image:
    if not data:
        raise HTTPException(422, f"Piece {index} is empty")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"Piece {index} exceeds 256 KiB")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise HTTPException(422, f"Piece {index} must be PNG or JPEG")
            if image.size != (64, 64) or image.mode != "RGB":
                raise HTTPException(422, f"Piece {index} must be an RGB 64x64 image")
            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(422, f"Piece {index} must be a single image frame")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as error:
        raise HTTPException(422, f"Piece {index} could not be decoded as a valid image") from error
