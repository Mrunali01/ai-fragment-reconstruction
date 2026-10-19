"""Real PostgreSQL tests; set JIGSAW_TEST_DATABASE_URL to an expendable database."""
import os
from uuid import uuid4

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import pytest

from jigsaw_ai.api import create_app
from jigsaw_ai.database import PostgresStore
from test_api import uploads


@pytest.fixture
def store():
    url = os.environ.get("JIGSAW_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set JIGSAW_TEST_DATABASE_URL for real PostgreSQL integration tests")
    schema = "test_jigsaw_" + uuid4().hex
    with psycopg.connect(url) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    repository = PostgresStore(make_conninfo(url, options=f"-c search_path={schema}"))
    try:
        repository.initialize()
        yield repository
    finally:
        with psycopg.connect(url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_database_roundtrip_restart_and_failed_jobs(store, tmp_path, monkeypatch):
    application = create_app(tmp_path / "missing.pt", tmp_path / "results", store.url)
    with TestClient(application) as client:
        response = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"})
        assert response.status_code == 201, response.text
        result = response.json()
        # Retrieval is from PostgreSQL, independent of the legacy JSON copy.
        (tmp_path / "results" / result["id"] / "result.json").unlink()
        assert client.get(f"/reconstructions/{result['id']}").json() == result
        import jigsaw_ai.api as api
        monkeypatch.setattr(api, "reconstruct", lambda *args: (_ for _ in ()).throw(RuntimeError("test failure")))
        failed = client.post("/reconstructions", files=uploads(), data={"scorer": "classical"})
        assert failed.status_code == 500
    with TestClient(create_app(tmp_path / "missing.pt", tmp_path / "results", store.url)) as client:
        assert client.get(f"/reconstructions/{result['id']}").json() == result
        assert client.get(result["image_url"]).status_code == 200
        assert client.get(f"/reconstructions/{uuid4()}").status_code == 404
    with store.connect() as connection:
        row = connection.execute("SELECT status, finished_at FROM reconstruction_jobs WHERE id = %s", (result["id"],)).fetchone()
        assert row[0] == "completed" and row[1] is not None
        assert connection.execute("SELECT count(*) FROM reconstruction_jobs WHERE status = 'failed'").fetchone()[0] >= 1


def test_completion_transaction_rolls_back_and_files_cleaned(store, tmp_path, monkeypatch):
    application = create_app(tmp_path / "missing.pt", tmp_path / "results", store.url)
    with TestClient(application) as client:
        def fail(result):
            raise psycopg.OperationalError("simulated unavailable database")
        monkeypatch.setattr(application.state.store, "complete", fail)
        response = client.post("/reconstructions", files=uploads(), data={"scorer": "classical", "search": "greedy"})
        assert response.status_code == 503
        assert list((tmp_path / "results").iterdir()) == []
    job_id = uuid4()
    store.start(job_id, [{}] * 25, "classical", "greedy")
    store.fail(job_id)
    assert store.get(job_id) is None


def test_configured_unreachable_database_fails_startup(tmp_path):
    with pytest.raises(psycopg.OperationalError):
        with TestClient(create_app(results_dir=tmp_path, database_url="postgresql://jigsaw@127.0.0.1:1/no_database")):
            pass


def test_database_outage_returns_service_unavailable(store, tmp_path, monkeypatch):
    application = create_app(tmp_path / "missing.pt", tmp_path / "results", store.url)
    with TestClient(application) as client:
        def unavailable():
            raise psycopg.OperationalError("simulated connection outage")
        monkeypatch.setattr(application.state.store, "connect", unavailable)
        assert client.get("/health").status_code == 503
        assert client.get(f"/reconstructions/{uuid4()}").status_code == 503
        response = client.post("/reconstructions", files=uploads(), data={"scorer": "classical"})
        assert response.status_code == 503
        assert list((tmp_path / "results").iterdir()) == []
