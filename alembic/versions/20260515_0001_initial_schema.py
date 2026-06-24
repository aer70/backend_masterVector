"""initial schema

Revision ID: 20260515_0001
Revises:
Create Date: 2026-05-15 00:00:00

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260515_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL,
            revoked BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id UUID PRIMARY KEY,
            user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            guest_token TEXT,
            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'success', 'failed')),
            input_file_key TEXT NOT NULL,
            svg_file_key TEXT,
            log_file_key TEXT,
            settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            error_text TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            duration_ms INTEGER,
            CHECK (user_id IS NOT NULL OR guest_token IS NOT NULL)
        )
        """
    )

    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS guest_token TEXT")
    op.execute("ALTER TABLE jobs ALTER COLUMN user_id DROP NOT NULL")

    # Stage 1: presets table from product plan.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS presets (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            owner_user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_presets_owner_name UNIQUE (owner_user_id, name)
        )
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_guest_created ON jobs(guest_token, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tokens_user ON refresh_tokens(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_presets_owner ON presets(owner_user_id)")


def downgrade() -> None:
    # Non-destructive baseline migration.
    pass
