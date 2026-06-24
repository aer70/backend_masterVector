# Stage 3.1 - Redis-backed Runtime State

Implemented:
- Rate limiting state moved from process memory to Redis keys.
- Signed download single-use token state moved from process memory to Redis keys.
- In-memory fallback retained for temporary Redis outages (best-effort resilience).

Details:
- Rate limit key format:
  - `rl:{bucket}:{client_ip}:{window_slot}`
- Single-use token key format:
  - `dl:used:{jti}`
- `BMP2SVG_STATE_USE_REDIS=1` controls whether Redis-backed state is enabled.

Testing support:
- Added `reset_runtime_state_for_tests()` in backend to clear both in-memory and Redis state.
- Integration tests updated to use this helper for isolation.

Validation commands:

```powershell
python -m ruff check backend tests
$env:RUN_INTEGRATION_TESTS='1'; python -m pytest -q tests/test_api_integration.py
$env:RUN_STAGE3_TESTS='1'; python -m pytest -q tests/test_stage3_queue_integration.py
```
