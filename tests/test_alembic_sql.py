from __future__ import annotations

from pathlib import Path

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "20260515_0001_initial_schema.py"
)


def migration_text() -> str:
    return MIGRATION_FILE.read_text(encoding="utf-8").lower()


def test_initial_migration_contains_core_tables() -> None:
    sql = migration_text()
    assert "create table if not exists users" in sql
    assert "create table if not exists refresh_tokens" in sql
    assert "create table if not exists jobs" in sql


def test_initial_migration_contains_presets_table() -> None:
    sql = migration_text()
    assert "create table if not exists presets" in sql
    assert "uq_presets_owner_name" in sql


def test_initial_migration_contains_job_integrity_constraints() -> None:
    sql = migration_text()
    assert "status in ('queued', 'running', 'success', 'failed')" in sql
    assert "check (user_id is not null or guest_token is not null)" in sql
    assert "idx_jobs_guest_created" in sql
