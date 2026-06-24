"""stage 3 queue hardening columns

Revision ID: 20260515_0002
Revises: 20260515_0001
Create Date: 2026-05-15 01:00:00

"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260515_0002"
down_revision = "20260515_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS blocked_at TIMESTAMPTZ")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS blocked_reason TEXT")

    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS max_retries INTEGER NOT NULL DEFAULT 2")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS queue_job_id TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS worker_id TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS input_size_bytes BIGINT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS svg_size_bytes BIGINT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS log_size_bytes BIGINT")

    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_running_heartbeat ON jobs(status, last_heartbeat_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_blocked ON users(is_blocked)")


def downgrade() -> None:
    # Non-destructive incremental migration.
    pass
