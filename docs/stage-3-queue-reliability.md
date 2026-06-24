# Stage 3 - Queue and Operational Reliability

## Implemented

- Worker execution moved from in-process thread to Redis + RQ.
- Queue retry policy implemented in worker task flow (`retry_count`, `max_retries`).
- Admin operations added:
  - list stuck jobs: `GET /admin/jobs/stuck`
  - recover stuck jobs: `POST /admin/jobs/recover-stuck`
  - manual retry: `POST /admin/jobs/{job_id}/retry`
  - block/unblock user: `PATCH /admin/users/{user_id}/block`
- Centralized operational metrics:
  - `GET /admin/metrics` (errors, durations, file sizes, queue depth)

## New Runtime Components

- Redis service in docker-compose.
- RQ worker entrypoint: `backend/worker.py`.
- API process enqueues jobs; worker process executes `backend.main.run_job_once`.

## New Environment Variables

- `REDIS_URL` (default `redis://127.0.0.1:6379/0`)
- `BMP2SVG_QUEUE_NAME` (default `bmp2svg-jobs`)
- `BMP2SVG_JOB_MAX_RETRIES` (default `2`)
- `BMP2SVG_ADMIN_TOKEN`
- `BMP2SVG_STUCK_JOB_SEC` (default `600`)

## DB Migration

Apply latest migrations:

```powershell
python -m alembic -c alembic.ini upgrade head
```

## Run Services

```powershell
docker compose up -d postgres redis
```

Run API:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run worker:

```powershell
python -m backend.worker
```

## Tests

Base checks:

```powershell
python -m ruff check backend tests alembic
python -m pytest -q
```

Stage 3 integration tests:

```powershell
$env:RUN_STAGE3_TESTS='1'; python -m pytest -q tests/test_stage3_queue_integration.py
```
