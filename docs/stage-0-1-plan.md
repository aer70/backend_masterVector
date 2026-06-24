# Stage 0 and Stage 1 Execution

## Stage 0 (Process + Quality Gate)

Definition of done for each task:
- Feature/code is implemented.
- Tests for the task pass locally.
- Regressions are checked.
- Small changelog entry is written.

Quality gate introduced:
- CI workflow at `.github/workflows/ci.yml`.
- Lint: `ruff check .`
- Tests: `pytest`

## Stage 1 (DB + Migrations)

Implemented:
- Alembic initialized (`alembic.ini`, `alembic/env.py`, `alembic/versions/*`).
- Initial migration `20260515_0001` adds:
  - `users`
  - `refresh_tokens`
  - `jobs`
  - `presets`
  - indexes and constraints
- Backend startup no longer mutates schema.
- Runtime now validates schema readiness and asks to run Alembic if missing.
- Legacy bootstrap SQL script is marked deprecated.

## Commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run migrations:

```powershell
python -m alembic -c alembic.ini upgrade head
```

Run checks:

```powershell
ruff check .
pytest
```

## Notes

- Since schema is now migration-managed, `docker-compose.yml` no longer mounts `db/init` scripts.
- Existing environments that were created before Alembic should be handled once with:
  1. backup DB
  2. align schema manually if needed
  3. `alembic stamp 20260515_0001`
  4. continue with new migrations
