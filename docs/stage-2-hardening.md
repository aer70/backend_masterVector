# Stage 2 - Backend Core Hardening

Implemented areas:
- Presets API: `GET/POST/PUT/DELETE /presets`
- Strict settings validation:
  - whitelist keys only
  - strict types by config schema
  - numeric ranges and enum checks
  - unknown fields are rejected
- Rate limiting:
  - auth endpoints (`/auth/register`, `/auth/login`)
  - upload endpoint (`/jobs`)
- Signed temporary download links:
  - `POST /jobs/{job_id}/download-link`
  - `GET /downloads/signed?token=...`
  - TTL bounded by config, optional single-use

Environment variables:
- `BMP2SVG_RATE_LIMIT_AUTH_COUNT` (default: 20)
- `BMP2SVG_RATE_LIMIT_AUTH_WINDOW_SEC` (default: 60)
- `BMP2SVG_RATE_LIMIT_UPLOAD_COUNT` (default: 10)
- `BMP2SVG_RATE_LIMIT_UPLOAD_WINDOW_SEC` (default: 60)
- `BMP2SVG_SIGNED_DOWNLOAD_DEFAULT_TTL_SEC` (default: 300)
- `BMP2SVG_SIGNED_DOWNLOAD_MAX_TTL_SEC` (default: 3600)

Run checks:

```powershell
python -m ruff check backend vectorization_service.py tests
python -m pytest -q
$env:RUN_INTEGRATION_TESTS='1'; python -m pytest -q tests/test_api_integration.py
```
