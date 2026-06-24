from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

if os.getenv("RUN_INTEGRATION_TESTS") != "1":
    pytest.skip("Integration tests are disabled. Set RUN_INTEGRATION_TESTS=1", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
DB_URL = os.getenv("DATABASE_URL", "postgresql://bmp2svg:bmp2svg@127.0.0.1:5432/bmp2svg")
os.environ["DATABASE_URL"] = DB_URL

from backend import main as appmod  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def migrate_database() -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", "upgrade", "head"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def db_conn() -> psycopg.Connection:
    with psycopg.connect(DB_URL) as conn:
        yield conn


@pytest.fixture(autouse=True)
def cleanup_tables(db_conn: psycopg.Connection) -> None:
    with db_conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE presets, jobs, refresh_tokens, users RESTART IDENTITY CASCADE")
    db_conn.commit()


@pytest.fixture(autouse=True)
def disable_worker_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appmod, "enqueue_job", lambda _job_id: "test-queue-job")


@pytest.fixture(autouse=True)
def reset_runtime_state() -> None:
    appmod.reset_runtime_state_for_tests()


@pytest.fixture()
def client() -> TestClient:
    with TestClient(appmod.app) as c:
        yield c


def register_user(client: TestClient, email: str, password: str = "password123") -> dict[str, str]:
    resp = client.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def auth_headers(tokens: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def insert_finished_log_job(db_conn: psycopg.Connection, user_id: int, *, log_file_key: str) -> str:
    job_id = str(uuid.uuid4())
    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                id, user_id, guest_token, status, input_file_key, svg_file_key, log_file_key,
                settings_json, error_text, created_at, started_at, finished_at, duration_ms
            ) VALUES (%s, %s, NULL, 'success', %s, NULL, %s, %s::jsonb, NULL, NOW(), NOW(), NOW(), 10)
            """,
            (job_id, user_id, "storage/uploads/dummy.bmp", log_file_key, "{}"),
        )
    db_conn.commit()
    return job_id


def test_presets_crud_and_permissions(client: TestClient) -> None:
    u1 = register_user(client, "u1@example.com")
    u2 = register_user(client, "u2@example.com")

    create_resp = client.post(
        "/presets",
        headers=auth_headers(u1),
        json={
            "name": "My Preset",
            "settings_json": {"method": "rle", "pre_scale_factor": 2.0},
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    preset = create_resp.json()

    list_resp = client.get("/presets", headers=auth_headers(u1))
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) == 1
    assert items[0]["id"] == preset["id"]

    forbidden_update = client.put(
        f"/presets/{preset['id']}",
        headers=auth_headers(u2),
        json={"name": "Other", "settings_json": {"method": "rle"}},
    )
    assert forbidden_update.status_code == 404

    invalid_update = client.put(
        f"/presets/{preset['id']}",
        headers=auth_headers(u1),
        json={"name": "Broken", "settings_json": {"unknown_key": 1}},
    )
    assert invalid_update.status_code == 400

    delete_resp = client.delete(f"/presets/{preset['id']}", headers=auth_headers(u1))
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "ok"


def test_settings_validation_negative_cases(client: TestClient) -> None:
    tokens = register_user(client, "settings@example.com")

    invalid_unknown = client.post(
        "/jobs",
        headers=auth_headers(tokens),
        files={"file": ("test.bmp", b"BMxx", "image/bmp")},
        data={"settings": '{"unknown_key": true}'},
    )
    assert invalid_unknown.status_code == 400
    assert "Unsupported settings keys" in invalid_unknown.text

    invalid_range = client.post(
        "/jobs",
        headers=auth_headers(tokens),
        files={"file": ("test.bmp", b"BMxx", "image/bmp")},
        data={"settings": '{"pre_scale_factor": 0}'},
    )
    assert invalid_range.status_code == 400
    assert "pre_scale_factor" in invalid_range.text


def test_signed_download_link_single_use(client: TestClient, db_conn: psycopg.Connection) -> None:
    tokens = register_user(client, "signed@example.com")
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", ("signed@example.com",))
        user_id = int(cur.fetchone()[0])

    job_id = str(uuid.uuid4())
    result_dir = ROOT / "storage" / "results" / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / f"{job_id}.log"
    log_path.write_text("hello\n", encoding="utf-8")

    rel_log_key = str(log_path.relative_to(ROOT)).replace("\\", "/")
    insert_finished_log_job(db_conn, user_id, log_file_key=rel_log_key)

    # We inserted a random job_id in helper; get latest job id for this user.
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM jobs WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,))
        actual_job_id = str(cur.fetchone()[0])

    link_resp = client.post(
        f"/jobs/{actual_job_id}/download-link?kind=log&ttl_seconds=120&single_use=true",
        headers=auth_headers(tokens),
    )
    assert link_resp.status_code == 200, link_resp.text
    signed_url = link_resp.json()["url"]

    first = client.get(signed_url)
    assert first.status_code == 200
    assert "hello" in first.text

    second = client.get(signed_url)
    assert second.status_code == 410


def test_rate_limit_auth_and_upload(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appmod, "AUTH_RATE_LIMIT_COUNT", 1)
    monkeypatch.setattr(appmod, "AUTH_RATE_LIMIT_WINDOW_SEC", 60)
    monkeypatch.setattr(appmod, "UPLOAD_RATE_LIMIT_COUNT", 1)
    monkeypatch.setattr(appmod, "UPLOAD_RATE_LIMIT_WINDOW_SEC", 60)

    register_user(client, "limit@example.com")
    appmod.reset_runtime_state_for_tests()

    login_1 = client.post("/auth/login", json={"email": "limit@example.com", "password": "password123"})
    assert login_1.status_code == 200
    login_2 = client.post("/auth/login", json={"email": "limit@example.com", "password": "password123"})
    assert login_2.status_code == 429

    # Reset bucket to isolate upload checks from auth checks.
    appmod.reset_runtime_state_for_tests()

    tokens = client.post("/auth/login", json={"email": "limit@example.com", "password": "password123"}).json()
    up_1 = client.post(
        "/jobs",
        headers=auth_headers(tokens),
        files={"file": ("a.bmp", b"BMxx", "image/bmp")},
        data={"settings": "{}"},
    )
    assert up_1.status_code == 200, up_1.text

    up_2 = client.post(
        "/jobs",
        headers=auth_headers(tokens),
        files={"file": ("b.bmp", b"BMxx", "image/bmp")},
        data={"settings": "{}"},
    )
    assert up_2.status_code == 429
