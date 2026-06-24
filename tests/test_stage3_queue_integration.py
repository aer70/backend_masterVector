from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

if os.getenv("RUN_STAGE3_TESTS") != "1":
    pytest.skip("Stage 3 tests are disabled. Set RUN_STAGE3_TESTS=1", allow_module_level=True)


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


def get_user_id(db_conn: psycopg.Connection, email: str) -> int:
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def insert_job(db_conn: psycopg.Connection, *, user_id: int, status: str = "queued", max_retries: int = 2) -> str:
    job_id = str(uuid.uuid4())
    upload_path = ROOT / "storage" / "uploads" / f"{job_id}.bmp"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(b"BMxx")

    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO jobs (
                id, user_id, guest_token, status, input_file_key, svg_file_key, log_file_key,
                settings_json, error_text, created_at, started_at, finished_at, duration_ms,
                retry_count, max_retries, input_size_bytes, queue_job_id
            ) VALUES (
                %s, %s, NULL, %s, %s, NULL, NULL,
                %s::jsonb, NULL, NOW(), NULL, NULL, NULL,
                0, %s, %s, NULL
            )
            """,
            (job_id, user_id, status, str(upload_path.relative_to(ROOT)).replace("\\", "/"), "{}", max_retries, 4),
        )
    db_conn.commit()
    return job_id


def test_queue_execute_retry_fail(client: TestClient, db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = register_user(client, "queue@example.com")
    user_id = get_user_id(db_conn, "queue@example.com")

    enqueue_calls: list[str] = []

    def fake_enqueue(job_id: str) -> str:
        enqueue_calls.append(job_id)
        return f"rq-{len(enqueue_calls)}"

    monkeypatch.setattr(appmod, "enqueue_job", fake_enqueue)

    # API enqueue path
    create_resp = client.post(
        "/jobs",
        headers=auth_headers(tokens),
        files={"file": ("sample.bmp", b"BMxx", "image/bmp")},
        data={"settings": "{}"},
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = create_resp.json()["id"]

    # Worker fail path with retry policy: max_retries=1 => fail, retry once, then final fail.
    with db_conn.cursor() as cur:
        cur.execute("UPDATE jobs SET max_retries = 1 WHERE id = %s", (job_id,))
    db_conn.commit()

    def fail_process_job(**_kwargs):
        out_dir = ROOT / "storage" / "results" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{job_id}.log"
        log_path.write_text("failed\n", encoding="utf-8")
        return {
            "status": "failed",
            "svg_path": None,
            "log_path": str(log_path),
            "duration_ms": 11,
            "error_message": "boom",
        }

    monkeypatch.setattr(appmod, "process_job", fail_process_job)

    appmod.run_job_once(job_id)
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, retry_count FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row[0] == "queued"
    assert int(row[1]) == 1

    appmod.run_job_once(job_id)
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, retry_count, error_text FROM jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    assert row[0] == "failed"
    assert int(row[1]) == 2
    assert "boom" in str(row[2])

    # Ensure a retry enqueue happened.
    assert enqueue_calls.count(job_id) >= 2

    # Worker success path on a new queued job.
    job_ok = insert_job(db_conn, user_id=user_id, status="queued", max_retries=1)

    def success_process_job(**_kwargs):
        out_dir = ROOT / "storage" / "results" / job_ok
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = out_dir / f"{job_ok}.svg"
        log_path = out_dir / f"{job_ok}.log"
        svg_path.write_text("<svg/>", encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return {
            "status": "success",
            "svg_path": str(svg_path),
            "log_path": str(log_path),
            "duration_ms": 9,
            "error_message": None,
        }

    monkeypatch.setattr(appmod, "process_job", success_process_job)
    appmod.run_job_once(job_ok)
    with db_conn.cursor() as cur:
        cur.execute("SELECT status, svg_size_bytes, log_size_bytes FROM jobs WHERE id = %s", (job_ok,))
        row = cur.fetchone()
    assert row[0] == "success"
    assert int(row[1]) > 0
    assert int(row[2]) > 0


def test_recovery_block_user_and_metrics(client: TestClient, db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    register_user(client, "admincase@example.com")
    user_id = get_user_id(db_conn, "admincase@example.com")

    job_id = insert_job(db_conn, user_id=user_id, status="running", max_retries=2)
    with db_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET started_at = NOW() - INTERVAL '20 minutes',
                last_heartbeat_at = NOW() - INTERVAL '20 minutes'
            WHERE id = %s
            """,
            (job_id,),
        )
    db_conn.commit()

    monkeypatch.setattr(appmod, "enqueue_job", lambda _job_id: "rq-recovered")
    admin_headers = {"X-Admin-Token": appmod.ADMIN_TOKEN}

    stuck = client.get("/admin/jobs/stuck?older_than_sec=300", headers=admin_headers)
    assert stuck.status_code == 200, stuck.text
    assert any(item["id"] == job_id for item in stuck.json())

    recovered = client.post("/admin/jobs/recover-stuck?older_than_sec=300", headers=admin_headers)
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["recovered"] >= 1

    block = client.patch(
        f"/admin/users/{user_id}/block",
        headers=admin_headers,
        json={"blocked": True, "reason": "abuse"},
    )
    assert block.status_code == 200

    login = client.post("/auth/login", json={"email": "admincase@example.com", "password": "password123"})
    assert login.status_code == 403

    metrics = client.get("/admin/metrics", headers=admin_headers)
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["jobs_total"] >= 1
    assert "queue_depth" in payload


def test_smoke_load_processing(db_conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (%s, %s, NOW()) RETURNING id",
            ("smoke@example.com", "x"),
        )
        user_id = int(cur.fetchone()[0])
    db_conn.commit()

    job_ids = [insert_job(db_conn, user_id=user_id, status="queued", max_retries=0) for _ in range(20)]

    def success_process_job(input_path: str, settings: dict, output_dir: str, output_basename: str, job_id: str):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        svg_path = out_dir / f"{output_basename}.svg"
        log_path = out_dir / f"{output_basename}.log"
        svg_path.write_text("<svg/>", encoding="utf-8")
        log_path.write_text("ok\n", encoding="utf-8")
        return {
            "status": "success",
            "svg_path": str(svg_path),
            "log_path": str(log_path),
            "duration_ms": 1,
            "error_message": None,
        }

    monkeypatch.setattr(appmod, "process_job", success_process_job)
    monkeypatch.setattr(appmod, "enqueue_job", lambda _job_id: "rq-skip")

    started = time.perf_counter()
    for job_id in job_ids:
        appmod.run_job_once(job_id)
    elapsed = time.perf_counter() - started

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs WHERE status = 'success'")
        ok_count = int(cur.fetchone()[0])
    assert ok_count == len(job_ids)
    assert elapsed < 10.0
