"""Public API response contracts; matching cost is not reconstruction confidence."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Scorer = Literal["cnn", "classical"]
Search = Literal["greedy", "swaps", "groups", "beam"]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    cnn_available: bool
    classical_available: bool = True


class Placement(BaseModel):
    row: int = Field(ge=0, le=4)
    column: int = Field(ge=0, le=4)
    upload_index: int = Field(ge=0, le=24)
    filename: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class ReconstructionResponse(BaseModel):
    schema_version: Literal[1] = 1
    id: UUID
    status: Literal["completed"] = "completed"
    created_at: datetime
    scorer: Scorer
    search: Search
    grid: tuple[int, int] = (5, 5)
    piece_size: tuple[int, int] = (64, 64)
    image_size: tuple[int, int] = (320, 320)
    arrangement: list[Placement] = Field(min_length=25, max_length=25)
    mean_border_cost: float = Field(description="Optimization cost; not a confidence or accuracy estimate", allow_inf_nan=False)
    checkpoint_sha256: str | None
    image_url: str
