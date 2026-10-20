"""Persist append-only corrections and background job state in PostgreSQL or JSON."""
from pathlib import Path
import json
from threading import RLock
from uuid import UUID

from jigsaw_ai.acquire_images import write_json


class WorkspaceStorage:
    def __init__(self, root: Path, store):
        self.root, self.store = root, store
        self.lock = RLock()

    def save(self, kind: str, record: dict) -> None:
        with self.lock:
            if self.store:
                self.store.save_record(kind, record)
            else:
                (self.root / "workspace").mkdir(exist_ok=True)
                write_json(self.root / "workspace" / f"{kind}_{UUID(record['id'])}.json", record)

    def get(self, kind: str, record_id: UUID) -> dict | None:
        if self.store:
            return self.store.get_record(kind, record_id)
        # Serialize readers with atomic replacements: Windows can deny an open
        # while another thread is replacing this same status file.
        with self.lock:
            try:
                return json.loads((self.root / "workspace" / f"{kind}_{record_id}.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None

    def records(self, kind: str, result_id: UUID | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        if self.store:
            return self.store.records(kind, result_id, limit, offset)
        with self.lock:
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in (self.root / "workspace").glob(f"{kind}_*.json")]
        rows = [row for row in rows if result_id is None or row.get("result_id") == str(result_id)]
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=True)[offset:offset + limit]

    def history(self, limit: int, offset: int) -> list[dict]:
        if self.store:
            return self.store.history(limit, offset)
        rows = []
        for path in self.root.glob("*/result.json"):
            try:
                UUID(path.parent.name)
            except ValueError:
                continue
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(rows, key=lambda row: (row["created_at"], row["id"]), reverse=True)[offset:offset + limit]
