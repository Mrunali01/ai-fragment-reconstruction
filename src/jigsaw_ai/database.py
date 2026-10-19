"""Small PostgreSQL repository. Images stay in the configured filesystem root."""

from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from jigsaw_ai.api_schemas import ReconstructionResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS jigsaw_schema (version integer PRIMARY KEY);
INSERT INTO jigsaw_schema VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS puzzles (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    rows integer NOT NULL CHECK (rows = 5),
    columns integer NOT NULL CHECK (columns = 5),
    inputs jsonb NOT NULL CHECK (jsonb_array_length(inputs) = 25)
);
CREATE TABLE IF NOT EXISTS reconstruction_jobs (
    id uuid PRIMARY KEY REFERENCES puzzles(id),
    scorer text NOT NULL CHECK (scorer IN ('cnn', 'classical')),
    search text NOT NULL CHECK (search IN ('greedy', 'swaps', 'groups', 'beam')),
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    error text
);
CREATE TABLE IF NOT EXISTS reconstruction_results (
    job_id uuid PRIMARY KEY REFERENCES reconstruction_jobs(id),
    image_path text NOT NULL,
    metadata jsonb NOT NULL
);
"""


class PostgresStore:
    def __init__(self, url: str):
        self.url = url

    def connect(self):
        return psycopg.connect(self.url, connect_timeout=5)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(SCHEMA)
            versions = connection.execute("SELECT version FROM jigsaw_schema").fetchall()
            if versions != [(1,)]:
                raise RuntimeError("Unsupported Jigsaw database schema version")

    def start(self, job_id: UUID, inputs: list[dict], scorer: str, search: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO puzzles (id, rows, columns, inputs) VALUES (%s, 5, 5, %s)",
                (job_id, Jsonb(inputs)),
            )
            connection.execute(
                "INSERT INTO reconstruction_jobs (id, scorer, search, status) VALUES (%s, %s, %s, 'running')",
                (job_id, scorer, search),
            )

    def complete(self, result: ReconstructionResponse) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO reconstruction_results (job_id, image_path, metadata) VALUES (%s, %s, %s)",
                (result.id, f"{result.id}/image.png", Jsonb(result.model_dump(mode="json"))),
            )
            cursor = connection.execute(
                "UPDATE reconstruction_jobs SET status = 'completed', finished_at = now() "
                "WHERE id = %s AND status = 'running'", (result.id,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Reconstruction job is not running")

    def fail(self, job_id: UUID) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE reconstruction_jobs SET status = 'failed', finished_at = now(), "
                "error = 'Reconstruction or result storage failed' WHERE id = %s AND status = 'running'",
                (job_id,),
            )

    def get(self, job_id: UUID) -> ReconstructionResponse | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT r.metadata FROM reconstruction_results r JOIN reconstruction_jobs j ON j.id = r.job_id "
                "WHERE j.id = %s AND j.status = 'completed'", (job_id,),
            ).fetchone()
        return ReconstructionResponse.model_validate(row[0]) if row else None
